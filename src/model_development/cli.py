import argparse
import os

def get_args():
        
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, help="seed", default=42)

    # tune related
    parser.add_argument("--gpu", type=str, help="list of gpus to use")
    parser.add_argument("--workers", type=int, help="number of workers for dataloader", default=4)
    parser.add_argument("--gpus-per-trial", type=float, help="number of gpus per trial", default=1)

    # data related
    parser.add_argument("--fold", type=int, nargs="+", help="folds to use", default=None)
    parser.add_argument("--use-case", nargs="+", help="hold out by", default=["UC1"], choices=["UC1", "UC2", "UC3"])
    parser.add_argument("--disjoint-train-ratio", nargs="+", type=float, default=[0.3])  
    parser.add_argument("--embeddings-len", nargs="+", help="length of the embeddings", default=[0], choices=["0","128","575","onehot","hash"])
    parser.add_argument("--bs", nargs='+',help="batch size", default=[4096], type=int)
    parser.add_argument("--weighted-sampler", nargs="+", help="weighted sampler", default=[None], choices=['None', "class", "biomodel", "nodes"])
    parser.add_argument("--dataloader", nargs="+", type=str, help="dataloader", default=["normal"], choices=["neighbor", "full"])
    parser.add_argument("--num-neighbors", nargs="+", type=int, help="number of neighbors to use", default=[5])
    parser.add_argument("--production", action="store_true", help="whether to use production mode, which trains over all DyPPIN", default=False)
    
    # model related
    parser.add_argument("--model", type=str, help="model to use", default="gcn", choices=["gcn", "gesn","deepsets"])
    parser.add_argument("--conv", nargs="+", help="convolution type", default=["SAGEConv"], choices=["SAGEConv", "GCNConv", "GATConv", "GINConv", "GraphConv", "DirGNNConv", "Reservoir"])
    parser.add_argument("--aggr", nargs="+", help="SAGE aggregation type", default=["add"], choices=["add", "mean", "max"])
    parser.add_argument("--pooling", nargs="+", help="pooling type", default=["add"], choices=["add", "mean", "max"])
    parser.add_argument("--pool-from", nargs="+", help="pooling from", default=["last"], choices=["last","all"])
    parser.add_argument("--hidden-dim", nargs="+", type=int, help="hidden dimension", default=[512])
    parser.add_argument("--layers", nargs="+", help="number of layers", default=[4], type=int)
    parser.add_argument("--dropout", nargs="+", type=float, help="dropout", default=[0.5])
    parser.add_argument("--bn", nargs="+", type=bool, help="whether to use batch normalization", default=[False])
    parser.add_argument("--w-init", nargs="+", type=str, help="weight initializer", default=["kaiming_uniform"], choices=["glorot","uniform","kaiming_uniform"])
    parser.add_argument("--uniform-bound", nargs="+", type=float, help="uniform bound", default=[None])
    
    # DirGNN related
    parser.add_argument("--dirgnn-conv", nargs="+", type=str, help="DirGNN convolution type", default=["GraphConv"], choices=["GCNConv", "GINConv", "GraphConv"])
    parser.add_argument("--dirgnn-alpha", nargs="+", type=float, help="DirGNN alpha", default=[0.5])

    # optimizer related
    parser.add_argument("--max-epochs", type=int, help="maximum number of epochs", default=5000)
    parser.add_argument("--optimizer", nargs="+", type=str, help="optimizer", default=["adam"], choices=["adam", "rmsprop", "sgd"])
    parser.add_argument("--lr", nargs="+", help="learning rate", default=[5e-4], type=float)
    parser.add_argument("--scheduler", nargs="+", help="scheduler", default=[None], choices=["None", "plateau", "step", "cosine"])
    parser.add_argument("--weight-decay", nargs="+", help="weight decay", default=[0.0], type=float)
    parser.add_argument("--early-stop", action="store_true", help="whether to use early stopping", default=False)
    parser.add_argument("--patience", type=int, help="patience for early stopping", default=100)
    parser.add_argument("--es-eps", type=float, help="early stopping epsilon", default=1e-8)
    parser.add_argument("--ckpt", action="store_true", help="whether to save checkpoints", default=False)
    parser.add_argument("--warmup", nargs="+", type=int, help="number of warmup steps", default=[0])

    # logger related
    parser.add_argument("--log-every-n-steps", type=int, help="log every n steps", default=5)
    parser.add_argument("--wandb-project", type=str, help="wandb project name", default="gnn-ppi-sens")


    args = parser.parse_args()

    if args.embeddings_len is not None:
        embeddings_len = []
        for l in args.embeddings_len:
            try:
                embeddings_len.append(int(l))
            except:
                embeddings_len.append(l)
    args.embeddings_len = embeddings_len


    if args.gpu is not None:
        os.environ.CUDA_VISIBLE_DEVICES = args.gpu

    if args.weighted_sampler is None:
        args.weighted_sampler = [None]
    else:
        sampler = []
        for s in args.weighted_sampler:
            if s == "None":
                sampler.append(None)
            else:
                sampler.append(s)
        args.weighted_sampler = sampler

    return args
