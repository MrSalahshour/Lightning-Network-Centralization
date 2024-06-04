import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn.functional as F


class CustomGATv2Extractor(BaseFeaturesExtractor):
    """
    Implements a custom feature extractor using the GATv2 (Graph Attention Network v2) architecture.
    
    The `CustomGATv2Extractor` class inherits from `BaseFeaturesExtractor` and is responsible for extracting features from graph-structured observations. It uses two GATv2 convolutional layers to process the node features and edge attributes, and outputs a feature vector for each node in the graph.
    
    The constructor takes the following parameters:
    - `observation_space`: The observation space of the environment, which contains information about the node features and edge attributes.
    - `features_dim`: The desired dimensionality of the output feature vectors.
    - `hidden_size`: The size of the hidden layers in the GATv2 convolutions.
    - `heads`: The number of attention heads in the GATv2 convolutions.
    - `dropout_rate`: The dropout rate applied to the GATv2 convolutions.
    
    The `forward` method takes an observation dictionary as input, which contains the node features, edge indices, and edge attributes. It applies the two GATv2 convolutions to the input, and returns a tensor of output feature vectors, where each row corresponds to a node in the graph.
    """
    def __init__(self, observation_space, features_dim=64, hidden_size = 64, heads = 4, dropout_rate = 0.2):
        super(CustomGATv2Extractor, self).__init__(observation_space, features_dim)
        num_features = observation_space['node_features'].shape[1]
        num_edge_features = observation_space['edge_attr'].shape[1]
        
        self.conv1 = GATv2Conv(num_features, hidden_size, heads=heads, edge_dim=num_edge_features, dropout = dropout_rate, concat = False)
        self.conv2 = GATv2Conv(hidden_size, hidden_size, heads=heads, edge_dim=num_edge_features, dropout = dropout_rate, concat = False)

    def forward(self, observations):
        x = [x.clone().detach().float() for x in observations['node_features']]
        edge_index = [x.clone().detach().long() for x in observations['edge_index']]
        edge_attr = [x.clone().detach().float() for x in observations['edge_attr']]
        outputs = []
        for i in range(len(x)):
          x_1 = self.conv1(x[i].squeeze(0), edge_index[i].squeeze(0), edge_attr[i].squeeze(0))
          x_1 = F.elu(x_1)
          x_1 = self.conv2(x_1, edge_index[i].squeeze(0), edge_attr[i].squeeze(0))
          outputs.append(x_1.mean(dim=0, keepdim=True))
        final_output = torch.cat(outputs, dim=0)
        return final_output
        # x = [torch.tensor(x, dtype=torch.float) for x in observations['node_features']]
        # edge_index = [torch.tensor(x, dtype=torch.int64) for x in observations['edge_index']]
        # edge_attr = [torch.tensor(x, dtype=torch.float) for x in observations['edge_attr']]
        # outputs = []
        # for i in range(len(x)):
        #   x_1 = self.conv1(x[i].squeeze(0), edge_index[i].squeeze(0), edge_attr[i].squeeze(0))
        #   x_1 = F.elu(x_1)
        #   x_1 = self.conv2(x_1, edge_index[i].squeeze(0), edge_attr[i].squeeze(0))
        #   outputs.append(x_1.mean(dim=0, keepdim=True))
        # final_output = torch.cat(outputs, dim = 0)
        # return final_output
    
