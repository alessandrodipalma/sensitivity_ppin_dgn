import torch
from torch.nn import functional as F
from torch import nn
from torch_geometric.nn.aggr import DeepSetsAggregation
from lightning import LightningModule
from src.model_development.log_metrics import *

class DeepSetsModule(LightningModule):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        hidden_dim = config["hidden_dim"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        # setup the linear layers to elaborate in inputs
        self.n_layers = config["layers"]
        self.layers = nn.Sequential()
        # self.batch_size = config["batch_size"]
        self.layers.add_module(f"linear_0", nn.Linear(input_dim, hidden_dim))
        for i in range(1, self.n_layers):
            self.layers.add_module(f"linear_{i}", nn.Linear(hidden_dim, hidden_dim))
        # create network for the node features from the created layers
        
        self.deepset = DeepSetsAggregation(local_nn=self.layers, global_nn=nn.Linear(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.deepset(x, index=batch)
        return x.view(-1)
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        
        out = torch.sigmoid(out)
        log_all(self, 'train', out, batch, loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        out = torch.sigmoid(out)
        log_all(self, 'val', out, batch, loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        out = torch.sigmoid(out)
        log_all(self, 'test', out, batch, loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

