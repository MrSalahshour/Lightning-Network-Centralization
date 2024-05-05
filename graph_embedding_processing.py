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
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    model.fit([G])  
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
