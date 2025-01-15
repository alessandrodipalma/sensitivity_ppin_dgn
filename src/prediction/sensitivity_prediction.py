# This code provides a command line tool to perform prediction of sensitivity between pair of proteins
# The user should call python sensitivity_prediction.py 
#   --p <file containing the protein list> 
#   --p-in <input protein or file containing the input proteins> 
#   --p-out <output proteins or file containing the output proteins> 
#   --all <predicts sensitivity for any protein pair>
#   --out <output file, if not specified prints to shell>
#   --gpu <gpu to use for prediction, if not specified uses CPU>
#   --ckpt <file containing the DGN model, if not specified uses the default model>
#   --timing <if specified, measures the inference time for the prediction and produces a csv file with the results>

import sys
import argparse
import torch

from src.prediction.build_graphs import *
from src.model_development.dgn.dgn import DGN
from src.model_development.datamodule import GraphDataModule
from lightning import Trainer


def load_and_predict(config, ckpt, datalist):

    if 'aggr' not in config.keys():
        config['aggr'] = config['SAGE_aggr']
    if 'uniform_bound' not in config.keys():
        config['uniform_bound'] = None
    if 'weight_initializer' not in config.keys():
        config['weight_initializer'] = 'kaiming_uniform'

    config['batch_size'] = 5000
    
    
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
    parser.add_argument('--p', help='file containing the protein list', required=True)
    parser.add_argument('--p-in', help='input protein or file containing the input proteins', required=True)
    parser.add_argument('--p-out', help='output proteins or file containing the output proteins', required=True)
    parser.add_argument('--all', help='predicts sensitivity for any protein pair. If true, p_in and p_out arguments are ignored.', action='store_true')
    parser.add_argument('--outpath', help='output file, if not specified prints to shell', required=False)
    parser.add_argument('--gpu', help='gpu to use for prediction, if not specified uses CPU', required=False, nargs='+', type=int)
    parser.add_argument('--data-dir', help='directory containing the dataset', required=False, default='data')
    parser.add_argument('--all-present', help='if specified, all the proteins have to be in the graph', action='store_true')
    parser.add_argument('--ckpt', help='file containing the DGN model, if not specified uses the default model', required=False, default='dgn.ckpt')
    args = parser.parse_args()
    

    # load the protein list
    with open(args.p, 'r') as f:
        proteins = f.readlines()

    # load the input proteins
    with open(args.p_in, 'r') as f:
        u_in = f.readlines()

    # load the output proteins
    with open(args.p_out, 'r') as f:
        u_out = f.readlines()

    # remove the newline characters
    proteins = pd.Series([p.strip() for p in proteins])
    u_in = pd.Series([p.strip() for p in u_in])
    u_out = pd.Series([p.strip() for p in u_out])
    
    biogrid_graph = load_biogrid()
    embeddings_dict = pickle.load((get_data_path() / "prediction_data/uniprot_embeddings_pca_128.pkl").open('rb'))

    #check if all proteins are in the graph
    if args.all_present:
        if len(proteins[~proteins.isin(biogrid_graph.nodes())]) > 0:
            print("Some proteins are not in the graph")
            sys.exit(1)
    else:
        proteins = proteins[proteins.isin(biogrid_graph.nodes())]
        if len(proteins) == 0:
            print("No proteins found in the graph")
            sys.exit(1)
            
    if args.all:
        u_in = proteins
        u_out = proteins
    else:
        u_in = u_in[u_in.isin(proteins)]
        if len(u_in) == 0:
            print("No input proteins found in the graph")
            sys.exit(1)
        u_out = u_out[u_out.isin(proteins)]
        if len(u_out) == 0:
            print("No output proteins found in the graph")
            sys.exit(1)

    datalist, pairs = get_input_graphs(proteins, u_in, u_out, biogrid_graph, embeddings_dict)

    # load the DGN model
    ckpts={}
    for fold in range(4):
        dir = get_data_path() / f'prediction_data/ckpts/{fold}'
        config = pickle.load((dir/"params.pkl").open("rb"))
        ckpts[fold] = {
            'config': config,
            'ckpt': dir/"last.ckpt"
        }
        
    # perform the prediction with the four checkpoints and then average them
    predictions = []    
    for k, v in ckpts.items():
        pred = load_and_predict(v['config'], v['ckpt'], datalist)
        predictions.append(pred)

    # average the predictions
    predictions = torch.stack(predictions)        
    predictions = torch.mean(predictions, dim=0)
        
    # save the results printing (in, out, prediction) to the output file
    if args.outpath:
        with open(args.outpath, 'w') as f:
            for i, p in pairs.iterrows():
                f.write(f"{p.input}\t{p.output}\t{predictions[i].item()}\n")
    else:
        for i, p in pairs.iterrows():
            print(f"{p.input}\t{p.output}\t{predictions[i].item()}")


