from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn



class GATv2(nn.Module):
    def __init__(self, observation_space, features_dim=32, hidden_size=32, heads=4, dropout_rate=0.2):
        super(GATv2, self).__init__()
        num_features = observation_space['node_features'].shape[1]
        num_edge_features = observation_space['edge_features'].shape[1]
        
        self.conv1 = GATv2Conv(num_features, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.conv2 = GATv2Conv(hidden_size, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.conv3 = GATv2Conv(hidden_size, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.conv4 = GATv2Conv(hidden_size, int(features_dim/50), heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        
        # Additional layers for reconstruction
        self.recon_conv1 = GATv2Conv(int(features_dim/50), hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.recon_conv2 = GATv2Conv(hidden_size, num_features, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)

    def forward(self, observations):
        data_list = []
        for i in range(len(observations['node_features'])):
            x = observations['node_features'][i].clone().detach().float()
            edge_index = observations['edge_index'][i].clone().detach().long()
            edge_attr = observations['edge_features'][i].clone().detach().float()
            data = Data(x=x.squeeze(0), edge_index=edge_index.squeeze(0), edge_attr=edge_attr.squeeze(0))
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        x_input = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        
        x = self.conv1(x_input, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)
        encoded_features = self.conv4(x, edge_index, edge_attr)
    
    
        # Determine the maximum number of nodes in the batch
        max_num_nodes = max(batch.batch.bincount()).item()
        node_feature_size = encoded_features.size(1)

        # Initialize a tensor to hold the padded outputs
        final_output = torch.zeros((batch.num_graphs, max_num_nodes, node_feature_size), device=encoded_features.device)

        # Fill the tensor with the node features
        for i in range(batch.num_graphs):
            graph_mask = batch.batch == i
            num_nodes = graph_mask.sum().item()
            final_output[i, :num_nodes, :] = encoded_features[graph_mask]
            #[batch, node, hidden_dim=32]
            
        return #(final_output, mask)



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
    def __init__(self, observation_space, features_dim=32, hidden_size=32, heads=4, dropout_rate=0.2):
        super(CustomGATv2Extractor, self).__init__(observation_space, features_dim)
        
        self.GATv2 = GATv2(observation_space, features_dim, hidden_size, heads, dropout_rate)
        self.GATv2.load_state_dict(torch.load("model/gnn.pth", map_location=torch.device('cpu')))
        self.flatten = torch.nn.Flatten(start_dim=1)
        
    def forward(self, observations):
        output = self.GATv2(observations)
        return output