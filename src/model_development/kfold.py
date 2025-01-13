import json
import os.path as osp
import pickle
import copy
import ray
from ray.tune import grid_search
from pathlib import Path

data_path = Path(osp.expanduser(json.load(open("config.json"))["data_path"]))/'training_data'

def create_fold_dict(config):
    
    filename = f"graph_classification_{config['embeddings_len']}.pkl"

    datalist = pickle.load((data_path/ "pyg_datalists" / filename).open("rb"))

    if type(config['use_case']) == str:
        config['use_case'] = grid_search([config['use_case']])

    folds = {}
    for use_case in config['use_case']['grid_search']:
        folds[use_case] = load_folds(datalist, use_case)        

    return folds, None

def load_folds(graphs, use_case):
    folds = pickle.load((data_path/f'folds/{use_case}.pkl').open("rb"))

    fold_pointers = {}

    for fold, outer_indices in folds.items():
        test = [g for g in graphs if g.df_index in outer_indices['test']]
        
        if type(outer_indices['train']) == dict:
            train = {}
            for inner_fold, indices in outer_indices['train'].items():
                train[inner_fold] = {"train": [g for g in graphs if g.df_index in indices['train']], 
                                     "val": [g for g in graphs if g.df_index in indices['test']]}
        else:
            train = [g for g in graphs if g.df_index in outer_indices['train']]
    
        fold_pointers[fold] = ray.put({'train':train,'test':test})

    return fold_pointers


def create_fold_pointers(config):
    data_dict = {}
    if type(config['embeddings_len']) == int:
        config['embeddings_len'] = grid_search([config['embeddings_len']])
    for embeddings_len in config['embeddings_len']['grid_search']:
        config_copy = copy.deepcopy(config)
        config_copy['embeddings_len'] = embeddings_len
        
        data_dict[embeddings_len] = create_fold_dict(config_copy)
                             
    return data_dict
        