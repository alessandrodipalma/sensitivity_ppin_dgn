import ray
from ray import air, tune
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

import os
import torch

import wandb

from src.model_development.dgn.dgn import DGN
from src.model_development.datamodule import GraphDataModule
from src.model_development.cli import get_args
from src.model_development.kfold import create_fold_pointers
from src.model_development.dgn.deepsets import DeepSetsModule

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

project = args.wandb_project

models_dict = {
    'gcn': DGN,
    'deepsets': DeepSetsModule,
}

search_space = {
    'seed': args.seed,

    # model related
    'model': args.model,
    "conv": tune.grid_search(args.conv),
    "aggr": tune.grid_search(args.aggr),
    "layers": tune.grid_search(args.layers),
    "hidden_dim": tune.grid_search(args.hidden_dim),
    "dropout": tune.grid_search(args.dropout),
    "bn": tune.grid_search(args.bn),
    "pooling": tune.grid_search(args.pooling),
    "pool_from": tune.grid_search(args.pool_from),
    "weight_initializer": tune.grid_search(args.w_init),
    "uniform_bound": tune.grid_search(args.uniform_bound),

    # optimizer related
    'weight_decay': tune.grid_search(args.weight_decay),
    "lr": tune.grid_search(args.lr),
    "scheduler": tune.grid_search(args.scheduler),
    'loss': 'bce',
    'optimizer': tune.grid_search(args.optimizer),
    'warmup_steps': tune.grid_search(args.warmup),

    # stopping related
    'max_epochs': args.max_epochs,
    'early_stop': args.early_stop,
    'patience': args.patience,
    'es_eps': args.es_eps,
    'ckpt': args.ckpt,

    # data related
    'workers': args.workers,
    'batch_size': tune.grid_search(args.bs),
    'val_fold': 0,
    'test_fold': tune.grid_search(list(range(4))) if args.fold is None else tune.grid_search(args.fold),
    'shuffle': True,    
    'weighted_sampler': tune.grid_search(args.weighted_sampler),
    'dataloader': tune.grid_search(args.dataloader),
    'num_neighbors': tune.grid_search(args.num_neighbors),
    'embeddings_len': tune.grid_search(args.embeddings_len),
    'use_case': tune.grid_search(args.use_case),

    # DirGNNConv related
    'dirgnn_conv': tune.grid_search(args.dirgnn_conv),
    'dirgnn_alpha': tune.grid_search(args.dirgnn_alpha),    
}

ray.shutdown()
ray.init(ignore_reinit_error=True, _temp_dir="/data/adipalma/tmp")

data_dict = create_fold_pointers(search_space)

data_dict_pointer = ray.put(data_dict)

    
def train_gcn(config):
    seed_everything(search_space['seed'], workers=True)
    
    wandb.init(project=project, name = ray.tune.get_trial_name(), config=config)
    logger = WandbLogger(project=project, config=config, settings=wandb.Settings(start_method="fork"))
    logger.log_hyperparams(config)
    
    data_dict = ray.get(data_dict_pointer)

    folds, df_pointer = data_dict[config['embeddings_len']]
    
    fold = ray.get(folds[config['use_case']][config['test_fold']])
    test = fold['test']
    outer_train = fold['train']
    train = outer_train[config["val_fold"]]['train']
    val = outer_train[config["val_fold"]]['val']
    
    if df_pointer is not None:
        sensitivity_df = ray.get(df_pointer)
    else:
        sensitivity_df = None

    data = GraphDataModule(train, val, test, config, property_df=sensitivity_df)
    data.setup()

    input_dim = train[0].x.shape[1]
    output_dim = 1
    
    model = models_dict[config['model']](input_dim, output_dim, config)
    
    callbacks=[]
    if config['early_stop']:
        callbacks.append(EarlyStopping(monitor="val_loss", mode='min', patience=config['patience'], min_delta=config['es_eps']))
    if config['ckpt']:
        callbacks.append(ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1, save_last=True, every_n_epochs=200))

    if torch.cuda.is_available():
        cuda_args = {"devices": 1, "accelerator": "gpu", "strategy": "ddp"}
    else:
        cuda_args = {"accelerator": "cpu"}

    trainer = Trainer(max_epochs=config['max_epochs'], logger=logger, enable_progress_bar = False, 
                            callbacks=callbacks, log_every_n_steps=args.log_every_n_steps, **cuda_args)
  
    logger.watch(model, log="all", log_freq=args.log_every_n_steps)

    trainer.fit(model, data)

    trainer.test(model, data.test_dataloader())

    wandb.finish()


os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
print("Gpu available: ", torch.cuda.is_available(), ray.get_gpu_ids())

if args.gpus_per_trial > 0:
    max_concurrent_trials = int(torch.cuda.device_count() / args.gpus_per_trial)
else:
    max_concurrent_trials = 8

tuner = tune.Tuner(tune.with_resources(train_gcn, {"cpu": search_space["workers"], "gpu": args.gpus_per_trial}),
                   tune_config=tune.TuneConfig(
                       metric="loss",
                       mode="min",
                       max_concurrent_trials=max_concurrent_trials,
                   ),
                   run_config=air.RunConfig(local_dir="/data/ray_results"),
                   param_space=search_space,
                   
)

tuner.fit()