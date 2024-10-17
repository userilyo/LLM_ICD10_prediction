import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from icd_search import icd10_graph

class GNNModule(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNNModule, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def prepare_gnn_data(icd_codes, all_codes):
    # Create a mapping of ICD codes to indices
    code_to_index = {code: i for i, code in enumerate(all_codes)}
    
    # Prepare node features (one-hot encoding)
    x = torch.zeros(len(all_codes), len(all_codes))
    for code in icd_codes:
        if code in code_to_index:
            x[code_to_index[code], code_to_index[code]] = 1
    
    # Prepare edge index
    edge_index = []
    for edge in icd10_graph.edges():
        if edge[0] in code_to_index and edge[1] in code_to_index:
            edge_index.append([code_to_index[edge[0]], code_to_index[edge[1]]])
            edge_index.append([code_to_index[edge[1]], code_to_index[edge[0]]])  # Add reverse edge for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def gnn_prediction(icd_codes, all_codes, model):
    data = prepare_gnn_data(icd_codes, all_codes)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    probabilities = torch.exp(out)
    return probabilities.squeeze().tolist()

def load_gnn_model(num_features, hidden_channels, num_classes):
    model = GNNModule(num_features, hidden_channels, num_classes)
    # In a real scenario, we would load pre-trained weights here
    return model
