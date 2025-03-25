from src.model_development.dgn.dgn import DGN
from src.model_development.datamodule import GraphDataModule
from lightning import Trainer
import torch

def load_and_predict(config, ckpt, datalist):

    if 'aggr' not in config.keys():
        config['aggr'] = config['SAGE_aggr']
    if 'uniform_bound' not in config.keys():
        config['uniform_bound'] = None
    if 'weight_initializer' not in config.keys():
        config['weight_initializer'] = 'kaiming_uniform'
    
    config['weighted_sampler'] = None
    
    input_dim = datalist[0].x.shape[1]
    output_dim = 1
    data = GraphDataModule(datalist, datalist, datalist, config)
    data.setup()

    model = DGN.load_from_checkpoint(ckpt, input_dim=input_dim, output_dim=output_dim, config=config,  map_location=torch.device('cuda'), strict=False)
    
    cuda_args = {"accelerator": "gpu"}

    trainer = Trainer(enable_progress_bar = False, logger=False, **cuda_args)
    print("Model type: ", type(model.eval()))
    
    predictions = trainer.predict(model, data.test_dataloader())
    predictions = torch.cat(predictions)
    
    return predictions