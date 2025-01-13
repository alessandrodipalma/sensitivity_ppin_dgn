from torch_geometric.nn import SAGEConv, DirGNNConv
import torch
from src.model_development.dgn.gcn_conv import GCNConv
from src.model_development.dgn.graph_conv import GraphConv


class DirGNNConvWrapper(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv, alpha, conv_params: dict = {}):
        super(DirGNNConvWrapper, self).__init__()
        
        # convert the string to the actual class
        if conv == "GCNConv":
            conv = GCNConv
        elif conv == "GraphConv":
            conv = GraphConv

        # assert conv_params['uniform_bound'] is not None, "uniform_bound must be set"

        self.conv = DirGNNConv(conv(in_channels, out_channels, **conv_params), alpha=alpha)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class ConvFactory():

    def __init__(self, config):
        super(ConvFactory, self).__init__()
        self.conv_params={}
        
        if config["conv"] == "GCNConv":
            self.conv = GCNConv
        elif config["conv"] == "SAGEConv":
            self.conv = SAGEConv
            self.conv_params["aggr"] = config["aggr"]
        elif config["conv"] == "DirGNNConv":
            self.conv_params['conv'] = config['dirgnn_conv']
            self.conv_params['alpha'] = config['dirgnn_alpha']
            self.conv = DirGNNConvWrapper
            
            if config['dirgnn_conv'] == "GraphConv" or config['dirgnn_conv'] == "SAGEConv":
                self.conv_params['conv_params'] = {"aggr": config["aggr"]}
            self.conv_params['conv_params'] = {"weight_initializer": config["weight_initializer"],
                                                "uniform_bound": config["uniform_bound"]}
        elif config["conv"] == "GraphConv":
            self.conv = GraphConv
            self.conv_params["aggr"] = config["aggr"]
        else:
            raise ValueError(f"Operator {config['conv']} not supported")

    def get(self, input_dim, output_dim):
        return self.conv(input_dim, output_dim, **self.conv_params)

    def get_conv_type(self):
        return self.conv