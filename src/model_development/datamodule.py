from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader, NeighborLoader
from lightning import LightningDataModule
import torch
from torch.utils.data import WeightedRandomSampler
import pandas as pd



class GraphDataModule(LightningDataModule):
    def __init__(self, train_graphs, val_graphs, test_graphs, config, unique_graph=None, property_df=None):
        """
        Args:
            train_graphs (list): List of networkx graphs for training
            val_graphs (list): List of networkx graphs for validation
            test_graphs (list): List of networkx graphs for testing
            config (dict): Dictionary of configuration parameters
            property_df (pd.DataFrame): DataFrame containing the property values for each graph
        """
        
        super().__init__()


        self.train_graphs = train_graphs
        self.val_graphs = val_graphs
        self.test_graphs = test_graphs

        self.batch_size = config['batch_size']
        
        self.shuffle = config['shuffle'] if config['weighted_sampler'] is None else False
        self.workers = config['workers']

        self.weighted_sampler = config['weighted_sampler']

        self.dataloader_params = {'batch_size': self.batch_size, 'num_workers': self.workers}
        if config['dataloader']=='neighbor':
            self.dataloader = NeighborLoader
            self.num_neighbors= [config['num_neighbors']]*config['layers']
        else:
            self.dataloader = DataLoader


    def setup(self, stage=None):
        def convert_graphs(graphs):
            out_graphs = []
            sample_id = 0
            for graph in graphs:
                if graph['edge_index'].shape[0] == 0:
                    continue
                x = graph['x']
                edge_index = graph['edge_index']
                
                if 'y' in graph.keys():
                    y = graph['y']
                    biomodel = graph['biomodel']
                    
                    out_graphs.append(Data(x=x, edge_index=edge_index, y=y, sample_id=torch.full((x.shape[0], ), sample_id) if self.dataloader==NeighborLoader else sample_id,
                                            biomodel=biomodel, input_species=graph['input_species'], output_species=graph['output_species']))
                else:
                    out_graphs.append(Data(x=x, edge_index=edge_index, sample_id=torch.full((x.shape[0], ), sample_id) if self.dataloader==NeighborLoader else sample_id,
                                            input_species=graph['input_species'], output_species=graph['output_species']))
                sample_id += 1
            return out_graphs

       
        self.train_data = convert_graphs(self.train_graphs)
        self.val_data = convert_graphs(self.val_graphs)
        self.test_data = convert_graphs(self.test_graphs)

        if self.weighted_sampler is not None:
            if self.weighted_sampler == 'class':
                train_targets = torch.tensor([data.y for data in self.train_data], dtype=torch.int)
                sample_count = torch.tensor(
                    [(train_targets== c).sum() for c in torch.unique(train_targets, sorted=True)])
                
                weights = 1.0 / sample_count.float()
                sample_weights = weights[train_targets]
                
            elif self.weighted_sampler == 'biomodel':
                train_samples = pd.Series([data.biomodel for data in self.train_data])
                sample_count = train_samples.value_counts()

                weights = sample_count.apply(lambda x: 1/x)
                sample_weights = torch.tensor(weights[train_samples])
            elif self.weighted_sampler == 'nodes':
                train_samples = pd.Series([data.x.shape[0] for data in self.train_data])
                sample_count = train_samples.value_counts()

                weights = sample_count.apply(lambda x: 1/x)
                sample_weights = torch.tensor(weights[train_samples].values)
            else:
                raise ValueError(f"Unknown weighted sampler {self.weighted_sampler}")

            self.train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

            
    def train_dataloader(self):
        dl_args = self.dataloader_params.copy()
        dl_args['shuffle'] = self.shuffle
        if self.weighted_sampler is not None:
            dl_args['sampler'] = self.train_sampler

        if self.dataloader==NeighborLoader:
            dl_args['shuffle'] = False
            dl_args['num_neighbors'] = self.num_neighbors
            return self.dataloader(Batch.from_data_list(self.train_data), **dl_args)
        else:
            return self.dataloader(self.train_data, **dl_args)


    def val_dataloader(self):
        return DataLoader(self.val_data, **self.dataloader_params)


    def test_dataloader(self):
        dl_args = self.dataloader_params
        dl_args['drop_last'] = False
    
        return DataLoader(self.test_data, **dl_args)
