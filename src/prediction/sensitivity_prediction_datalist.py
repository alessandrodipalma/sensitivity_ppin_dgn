import argparse
import torch

from src.prediction.build_graphs import *
from src.model_development.dgn.dgn import DGN
from src.model_development.datamodule import GraphDataModule
from lightning import Trainer
import os
import pandas as pd

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict sensitivity between pair of proteins')
    parser.add_argument('--datalist', help='file containing the protein list', required=True)
    parser.add_argument('--outpath', help='output file, if not specified prints to shell', required=True)
    parser.add_argument('--gpu', help='gpu to use for prediction, if not specified uses CPU', required=False, nargs='+', type=int)
    parser.add_argument('--data-dir', help='directory containing the dataset', required=False, default='data')
    parser.add_argument('--all-present', help='if specified, all the proteins have to be in the graph', action='store_true')
    parser.add_argument('--batch-size', help='batch size for prediction', required=False, default=256, type=int)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu)) if args.gpu else "-1"
    datalist = pickle.load(open(args.datalist, "rb"))

    # load the DGN model
    
    ckpts={}
    for fold in range(4):
        dir = get_data_path() / f'prediction_data/ckpts/io/{fold}'
        config = pickle.load((dir/"params.pkl").open("rb"))
        config['batch_size'] = args.batch_size
        ckpts[fold] = {
            'config': config,
            'ckpt': dir/"last.ckpt"
        }
        
    predictions = []    
    for k, v in ckpts.items():
        try:
            pred = load_and_predict(v['config'], v['ckpt'], datalist)
            predictions.append(pred)
        except Exception as e:
            print(f"Error in fold {k}: {e}")
            continue

    predictions = torch.sigmoid(torch.stack(predictions))        
    predictions_mean = torch.mean(predictions, dim=0)
    predictions_maj, _ = torch.mode((predictions>0.5), dim=0)

    print("Mean prediction: ", predictions_mean.shape)
    print("Majority prediction: ", predictions_maj.shape)

    torch.save(predictions, 'predictions.pt')
    df = pd.DataFrame({
            'biomodel': [data['biomodel'] for data in datalist],
            'input_species': [data['input_species'] for data in datalist],
            'output_species': [data['output_species'] for data in datalist],
            'distance': [data['distance'] for data in datalist],
            'distance_ind': [data['distance_ind'] for data in datalist],
            'prediction_mean': predictions_mean,
            'prediction_mean_bin': predictions_mean>0.5,
            'prediction_maj': predictions_maj,
            'prediction_max': torch.max(predictions, dim=0).values,
            'prediction_min': torch.min(predictions, dim=0).values,
            'prediction_std': torch.std(predictions, dim=0),
        })
    df.to_csv(args.outpath, sep='\t')

