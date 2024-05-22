from karateclub import FeatherGraph
from karateclub import GeoScattering
from karateclub import LDP
from karateclub import GL2Vec
from karateclub import Graph2Vec


import networkx as nx

#Note that the size of the graph embedding with default values would be 500.
def get_feather_embedding(model, G):
    """
    This function generates a graph embedding for the input graph 'G' using the FeatherGraph model from the KarateClub library.

    Parameters:
    G (networkx.classes.graph.Graph): The input graph for which the embedding is to be generated.

    Returns:
    numpy.ndarray: The embedding of the input graph 'G'. The embedding is a one-dimensional numpy array.

    The FeatherGraph model parameters are set as follows:
    - order: The order of the Chebyshev polynomials, set to 5.
    - eval_points: The number of evaluation points for the Chebyshev polynomials, set to 25.
    - theta_max: The maximum value for the random walk length, set to 2.5.
    - seed: The seed for the random number generator, set to 42.
    - pooling: The pooling method applied to the node features, set to 'mean'.
    """
    # print(is_attributed(G))
    G_temp = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    model.fit([G_temp])  
    graph_embedding = model.get_embedding()[0]
    return model, graph_embedding

def get_feather_embedder():
    model = FeatherGraph(order=3, eval_points=9, theta_max=2.5, seed=42, pooling='mean')
    return model


#Note that the size of the graph embedding with default values would be 111.
def get_geo_scattering_embedding(G):
    """
    This function generates a graph embedding for the input graph 'G' using the GeoScattering model from the KarateClub library.

    Parameters:
    G (networkx.classes.graph.Graph): The input graph for which the embedding is to be generated.

    Returns:
    numpy.ndarray: The embedding of the input graph 'G'. The embedding is a one-dimensional numpy array.

    The GeoScattering model parameters are set as follows:
    - order: The order of the Chebyshev polynomials, set to 4.
    - moments: The number of moments calculated, set to 4.
    - seed: The seed for the random number generator, set to 42.
    """
    model = GeoScattering(order=4, moments=4, seed=42)
    model.fit([G])  
    graph_embedding = model.get_embedding()[0]
    return graph_embedding


#Note that the size of the graph embedding with default values would be 160.
def get_LDP_embedding(G):
    """
    This function generates a graph embedding for the input graph 'G' using the LDP model from the KarateClub library.

    Parameters:
    G (networkx.classes.graph.Graph): The input graph for which the embedding is to be generated.

    Returns:
    numpy.ndarray: The embedding of the input graph 'G'. The embedding is a one-dimensional numpy array.

    The LDP model parameters are set as follows:
    - bins: The number of bins in the histogram, set to 32.
    """
    model = LDP(bins=32)
    model.fit([G])  
    graph_embedding = model.get_embedding()[0]
    return graph_embedding

#Note: This method works for attributed nodes too.
def get_GL2Vec_embedding(graph_list):
    """
    This function generates a graph embedding for the input graph 'G' using the GL2Vec model from the KarateClub library.

    Parameters:
    G (networkx.classes.graph.Graph): The input graph for which the embedding is to be generated.

    Returns:
    numpy.ndarray: The embedding of the input graph 'G'. The embedding is a one-dimensional numpy array.

    The GL2Vec model parameters are set as follows:
    - wl_iterations: The number of Weisfeiler-Lehman iterations, set to 2.
    - dimensions: The number of dimensions of the embedding, set to 128.
    - workers: The number of worker threads to train the model, set to 4.
    - down_sampling: The threshold for configuring which higher-frequency words are randomly downsampled, set to 0.0001.
    - epochs: The number of epochs to train the model, set to 10.
    - learning_rate: The initial learning rate, set to 0.025.
    - min_count: The minimum count of words to consider when training the model, set to 5.
    - seed: The seed for the random number generator, set to 42.
    - erase_base_features: Whether to erase the base features, set to False.
    """
    model = GL2Vec(wl_iterations=2, dimensions=128, workers=4, down_sampling=0.0001, epochs=10, learning_rate=0.025, min_count=5, seed=42, erase_base_features=False)
    model.fit(graph_list)  
    # graph_embedding =  model.infer([G])[0]

    return model


def get_Graph2Vec_embedding(G):
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    """
    This function generates a graph embedding for the input graph 'G' using the Graph2Vec model from the KarateClub library.

    Parameters:
    G (networkx.classes.graph.Graph): The input graph for which the embedding is to be generated.

    Returns:
    numpy.ndarray: The embedding of the input graph 'G'. The embedding is a one-dimensional numpy array.

    The Graph2Vec model parameters are set as follows:
    - wl_iterations: The number of Weisfeiler-Lehman iterations, set to 2.
    - attributed: Whether the graph is attributed or not, set to False.
    - dimensions: The number of dimensions of the embedding, set to 128.
    - workers: The number of worker threads to train the model, set to 4.
    - down_sampling: The threshold for configuring which higher-frequency words are randomly downsampled, set to 0.0001.
    - epochs: The number of epochs to train the model, set to 10.
    - learning_rate: The initial learning rate, set to 0.025.
    - min_count: The minimum count of words to consider when training the model, set to 5.
    - seed: The seed for the random number generator, set to 42.
    - erase_base_features: Whether to erase the base features, set to False.
    """
    model = Graph2Vec(wl_iterations=2, attributed=True, dimensions=128, workers=4, down_sampling=0.0001, epochs=10, learning_rate=0.025, min_count=5, seed=42, erase_base_features=False)
    model.fit([G])  
    graph_embedding = model.get_embedding()[0]
    return graph_embedding

def is_attributed(G):
    # Check if there is any edge attribute
    for _, _, edge_data in G.edges(data=True):
        if edge_data:
            print("Edge attributes:")
            for key, value in edge_data.items():
                print(f"Attribute: {key}, Type: {type(value)}, Value: {value}")
            return True

    # If no node or edge attributes, then the graph is not attributed
    return False

def is_weighted(G):
    return all('weight' in data for _, _, data in G.edges(data=True))




import random
import numpy as np
import networkx as nx
import networkit as nk
from typing import Union
from collections import deque
from littleballoffur.sampler import Sampler


NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class ForestFireSampler(Sampler):
    """An implementation of forest fire sampling. The procedure is a stochastic
    snowball sampling method where the expansion is proportional to the burning probability.
    `"For details about the algorithm see this paper." <https://cs.stanford.edu/people/jure/pubs/sampling-kdd06.pdf>`_

    NOTICE THAT THIS IS A MODIFICATION THAT YOU CAN SET INITIAL SEED FOR STARTING THE SAMPLING PROCESS.
    Args:
        number_of_nodes (int): Number of sampled nodes. Default is 100.
        p (float): Burning probability. Default is 0.4.
        seed (int): Random seed. Default is 42.
    """

    def __init__(
        self,
        number_of_nodes: int = 100,
        p: float = 0.4,
        seed: int = 42,
        max_visited_nodes_backlog: int = 100,
        restart_hop_size: int = 10,
    ):
        self.number_of_nodes = number_of_nodes
        self.p = p
        self.seed = seed
        self._set_seed()
        self.restart_hop_size = restart_hop_size
        self.max_visited_nodes_backlog = max_visited_nodes_backlog

    def _create_node_sets(self, graph, seed_nodes):
        """
        Create a starting set of nodes.
        """
        if seed_nodes is None:
            self._sampled_nodes = seed_nodes
        else:
            self._sampled_nodes = seed_nodes
        self._set_of_nodes = set(range(self.backend.get_number_of_nodes(graph)))
        self._visited_nodes = deque(maxlen=self.max_visited_nodes_backlog)

    def _start_a_fire(self, graph):
        """
        Starting a forest fire froseed_nodes
        """
        remaining_nodes = list(self._set_of_nodes.difference(self._sampled_nodes))
        seed_node = random.choice(remaining_nodes)
        self._sampled_nodes.add(seed_node)
        node_queue = deque([seed_node])
        while len(self._sampled_nodes) < self.number_of_nodes:
            if len(node_queue) == 0:
                node_queue = deque(
                    [
                        self._visited_nodes.popleft()
                        for k in range(
                            min(self.restart_hop_size, len(self._visited_nodes))
                        )
                    ]
                )
                if len(node_queue) == 0:
                    print(
                        "Warning: could not collect the required number of nodes. The fire could not find enough nodes to burn."
                    )
                    break
            top_node = node_queue.popleft()
            self._sampled_nodes.add(top_node)
            neighbors = set(self.backend.get_neighbors(graph, top_node))
            unvisited_neighbors = neighbors.difference(self._sampled_nodes)
            score = np.random.geometric(self.p)
            count = min(len(unvisited_neighbors), score)
            burned_neighbors = random.sample(unvisited_neighbors, count)
            self._visited_nodes.extendleft(
                unvisited_neighbors.difference(set(burned_neighbors))
            )
            for neighbor in burned_neighbors:
                if len(self._sampled_nodes) >= self.number_of_nodes:
                    break
                node_queue.extend([neighbor])

    def sample(self, graph, seed_nodes = None) :
        """
        Sampling nodes iteratively with a forest fire sampler.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

            * **graph** *(NetworkX or NetworKit graph)* - The seed nodes that the sampling start with (could be none).

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_node_sets(graph, seed_nodes)
        while len(self._sampled_nodes) < self.number_of_nodes:
            self._start_a_fire(graph)
        new_graph = self.backend.get_subgraph(graph, self._sampled_nodes)
        return new_graph
