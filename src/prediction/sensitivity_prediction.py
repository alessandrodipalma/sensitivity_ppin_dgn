import sys
import argparse
import torch

from src.prediction.build_graphs import *
from src.prediction.predict import load_and_predict


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
    parser.add_argument('--emb', help='whther to use protein embeddings or not', action='store_true')
    parser.add_argument('--batch-size', help='batch size for prediction', required=False, default=256, type=int)
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

    if args.emb:
        embeddings_dict = pickle.load((get_data_path() / "prediction_data/uniprot_embeddings_pca_128.pkl").open('rb'))
        datalist, pairs = get_input_graphs(proteins, u_in, u_out, biogrid_graph, embeddings_dict)
        base_dir = get_data_path() / f'prediction_data/ckpts/io+emb/'
    else:
        datalist, pairs = get_input_graphs(proteins, u_in, u_out, biogrid_graph)
        base_dir = get_data_path() / f'prediction_data/ckpts/io/'

    # load the DGN model
    ckpts={}
    for fold in range(4):
        dir = base_dir / fold
        config = pickle.load((dir/"params.pkl").open("rb"))
        config['batch_size'] = args.batch_size
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


