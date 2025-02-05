import torch
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GraphNorm
from lightning import LightningModule
from src.model_development.log_metrics import *
from src.model_development.dgn.conv_factory import ConvFactory, DirGNNConvWrapper
from src.model_development.dgn.warmup_scheduler import WarmupScheduler



class DGN(LightningModule):

    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        
        hidden_dim = config["hidden_dim"]
        self.hidden_dim = hidden_dim
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.warmup_steps = config["warmup_steps"]
        self.max_epochs = config["max_epochs"]

        self.layers = config["layers"]
          
        if config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam
        elif config["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSprop
        elif config["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD
        else:
            raise ValueError(f"Optimizer {config['optimizer']} not supported")

        self.scheduler = config["scheduler"]
        self.dataloader = config["dataloader"]
        self.conv_factory = ConvFactory(config)
        
        if type(hidden_dim) == int:
            self.conv0 = self.conv_factory.get(input_dim, hidden_dim)
            for i in range(1, self.layers):
                setattr(self, f"conv{i}", self.conv_factory.get(hidden_dim, hidden_dim))
            self.classifier = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.conv0 = self.conv(input_dim, hidden_dim[0])
            for i in range(1, self.layers):
                setattr(self, f"conv{i}", self.conv_factory.get(hidden_dim[i-1], hidden_dim[i]))
            self.classifier = torch.nn.Linear(hidden_dim[-1], output_dim)

        if config["pool_from"] == "all":
            self.classifier = torch.nn.Linear(hidden_dim*self.layers, output_dim)
            
        if config["pooling"] == "add":
            self.pool = global_add_pool
        elif config["pooling"] == "mean":
            self.pool = global_mean_pool
        elif config["pooling"] == "max":
            self.pool = global_max_pool

        self.pool_from = config["pool_from"]

        if config["bn"]:
            self.bn = GraphNorm(hidden_dim)

        self.dropout = config["dropout"]

    def forward(self, data):
        x, edge_index, batch_id, sample_id = data.x, data.edge_index, data.batch, data.sample_id
            
        # Apply the GCN layers
        outputs = torch.zeros((x.shape[0], self.layers, self.hidden_dim)).to(x.device)
        for i in range(self.layers):
            conv = getattr(self, f"conv{i}")
            x = F.relu(conv(x, edge_index))
            if hasattr(self, "bn"):
                x = self.bn(x)
            if self.dropout>0:
                x = F.dropout(x, training=self.training, p=self.dropout)
            outputs[:, i, :] = x
        
        # Aggregate node features into a single graph feature vector
        if self.pool_from == "all":
            x = outputs.view(x.shape[0], -1)
        
        if self.dataloader == "neighbor" and self.training:
            x = self.pool(x, sample_id)
        else:
            x = self.pool(x, batch_id)
        out = self.classifier(x)

        return out.squeeze(1)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        
        if self.dataloader=='neighbor':
            batch.y = batch.y[:out.shape[0]]

        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        
        out = torch.sigmoid(out)

        log_all(self, "train", out, batch.y, loss)
        return loss

    def on_train_epoch_end(self):
        
        if self.conv_factory.conv == DirGNNConvWrapper:
            for i in range(self.layers):
                self.log_dict({f"W_in_{i}": torch.norm(list(self.__getattr__(f"conv{i}").conv.conv_in.parameters())[1]), f"W_out_{i}": torch.norm(list(self.__getattr__(f"conv{i}").conv.conv_out.parameters())[1])})
         

        return super().on_train_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        out = torch.sigmoid(out)

        log_all(self, "val", out, batch.y, loss)
        


    def test_step(self, batch, batch_idx):

        out = self(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        out = torch.sigmoid(out)

        log_all(self, "test", out, batch.y, loss)
        


    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        schedulers = []
        if self.scheduler == None:
            return {"optimizer": optimizer, "monitor": "val_loss"}
        
        if self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6, verbose=True)
        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=512, gamma=0.5)
        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)

        if self.warmup_steps>0:
            schedulers.append({"scheduler": WarmupScheduler(optimizer, self.warmup_steps, scheduler), "interval": "step"})

        schedulers.append({"scheduler": scheduler, "monitor":"val_loss"})
       
        return [optimizer], schedulers
    

