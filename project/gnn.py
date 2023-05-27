from logger import get_logger

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv, Linear
from focal_loss.focal_loss import FocalLoss
from sklearn.metrics import f1_score, accuracy_score, precision_score


logger = get_logger('log_gnn')

def get_conv(hidden_channels, aggr, version=1):
    try:

        if version == 1:
            conv = HeteroConv({
                ('source', 'edge', 'target'): SAGEConv((-1, -1), hidden_channels),
                ('target', 'rev_edge', 'source'): SAGEConv((-1, -1), hidden_channels)
            }, aggr=aggr)

        elif version == 2:
            conv = HeteroConv({
                ('source', 'edge', 'target'): GATConv((-1, -1), hidden_channels),
                ('target', 'rev_edge', 'source'): GATConv((-1, -1), hidden_channels)
            }, aggr=aggr)

        elif version == 3:
            conv = HeteroConv({
                ('source', 'edge', 'target'): GATv2Conv((-1, -1), hidden_channels),
                ('target', 'rev_edge', 'source'): GATv2Conv((-1, -1), hidden_channels)
            }, aggr=aggr)

        return conv


    except Exception as e:
        logger.info(f'Error creating Heteroconv layer: \n {e}')       

class HeteroGNN(torch.nn.Module):
       
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, p_dropout, aggr='sum', version=1):
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels,
        self.num_layers = num_layers
        self.p_dropout = p_dropout

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = get_conv(hidden_channels=hidden_channels, aggr=aggr, version=version)
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.p_dropout) for key, x in x_dict.items()}
        return self.lin(x_dict["source"])


def train_model(model, train_dataset, optimizer, loss_function='ce'):
    try: 


        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        train_dataset.to(device)
        model.train()
        optimizer.zero_grad()
        out = model(train_dataset.x_dict, train_dataset.edge_index_dict)
        prediction = out.argmax(1).cpu().numpy()

        if loss_function == 'ce':
            criterion = F.cross_entropy
            loss = criterion(out, train_dataset['source'].y) 
        elif loss_function == 'fl':
            criterion = FocalLoss(gamma=1)
            sm = torch.nn.Softmax(dim=-1)
            loss = criterion(sm(out), train_dataset['source'].y)           
    
        loss.backward()
        optimizer.step()
        true_values = train_dataset['source'].y.cpu().numpy()
        micro = f1_score(true_values, prediction, average='weighted', zero_division=0)
        accuracy = accuracy_score(true_values, prediction, normalize=True)
        return float(loss), micro, accuracy
    except Exception as e:
        logger.error(f'Error training model: \n {e}')
        raise

@torch.no_grad()
def test_model(model, val_dataset, loss_function='ce'):
    try:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        val_dataset.to(device)
        model.eval()
        out = model(val_dataset.x_dict, val_dataset.edge_index_dict)
        prediction = out.argmax(1).cpu().numpy()
        true_values = val_dataset['source'].y.cpu().numpy()
        micro = f1_score(true_values, prediction, average='weighted', zero_division=0)
        accuracy = accuracy_score(true_values, prediction, normalize=True)
        
        if loss_function == 'ce':
            criterion = F.cross_entropy
            loss = criterion(out, val_dataset['source'].y) 
        elif loss_function == 'fl':
            criterion = FocalLoss(gamma=1)
            sm = torch.nn.Softmax(dim=-1)
            loss = criterion(sm(out), val_dataset['source'].y)    

        loss = float(loss)
        precision = precision_score(true_values, prediction, average=None, zero_division=0)
        return loss, micro, accuracy

    except Exception as e:
        logger.error(f'Error testing model: \n {e}')
        raise



class HeteroGNN_bkp(torch.nn.Module):
       
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, p_dropout, aggr='sum'):
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels,
        self.num_layers = num_layers
        self.p_dropout = p_dropout

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in metadata[1]
            }, aggr=aggr)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.p_dropout) for key, x in x_dict.items()}
        return self.lin(x_dict["source"])
        


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    # model = GCN(hidden_channels=16)
    # print(model)