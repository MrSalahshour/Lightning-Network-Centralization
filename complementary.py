import networkx as nx
def get_node_with_highest_degree(graph):
    """
    Returns the node with the highest degree in a directed graph.

    """

    # Calculate the degrees of all nodes in the graph
    degrees = dict(graph.in_degree())

    # Find the node with the highest degree
    node_with_highest_degree = max(degrees, key=degrees.get)

    return node_with_highest_degree

def get_node_with_highest_betweenness_centrality(G):
    """
    Returns the node with the highest betweenness centrality in a directed graph.

    """
    # Calculate the betweenness centrality for each node in the graph
    betweenness_centralities = nx.betweenness_centrality(G)

    # Find the node with the highest betweenness centrality
    highest_betweenness_node = max(betweenness_centralities, key=betweenness_centralities.get)

    return highest_betweenness_node

def add_edge_to_digraph(G: nx.DiGraph, source: any, target: any, edge_attrs: dict) -> nx.DiGraph:
    """
    Adds an edge between two nodes in a NetworkX DiGraph with the specified edge attributes.

    """

    # Check if the source and target nodes exist in the graph
    if not G.has_node(source):
        raise ValueError(f"The source node {source} does not exist in the graph.")
    if not G.has_node(target):
        raise ValueError(f"The target node {target} does not exist in the graph.")

    # Add the edge with the specified attributes
    G.add_edge(source, target, **edge_attrs)

    return G


def create_edge_attributes_dict(channel_id, capacity, fee_base_msat, fee_rate_milli_msat, balance):
    """
    Create and return a dictionary with the edge attributes.

    Args:
        channel_id (str): The ID of the channel.
        capacity (int): The maximum capacity of the channel.
        fee_base_msat (int): The base fee in millisatoshis.
        fee_rate_milli_msat (int): The fee rate in milli-millisatoshis per millionth.
        balance (int): The current balance of the channel.

    Returns:
        dict: A dictionary containing the edge attributes.
    """
    edge_attributes = {
        'channel_id': channel_id,
        'capacity': capacity,
        'fee_base_msat': fee_base_msat,
        'fee_rate_milli_msat': fee_rate_milli_msat,
        'balance': balance
    }

    return edge_attributes

## defining channel fees
## NOTE: is it needed?

def initiate_fees(directed_edges, approach='half'):
    '''
    approach = 'random'
    approach = 'half'


    NOTE : This Function is written assuming that two side of channels are next to each other in directed_edges
    '''
    G = directed_edges[['src', 'trg', 'channel_id', 'capacity', 'fee_base_msat', 'fee_rate_milli_msat']]
    G = G.assign(balance=None)
    r = 0.5
    for index, row in G.iterrows():
        balance = 0
        cap = row['capacity']
        if index % 2 == 0:
            if approach == 'random':
                r = np.random.random()
            balance = r * cap
        else:
            balance = (1 - r) * cap
        G.at[index, "balance"] = balance

    return G

def set_channels_fees(edges, src, trgs, channel_ids, static_fee_base_msat, static_fee_rate_milli_msat):
    if (2* len(trgs) == len(static_fee_base_msat) & len(static_fee_base_msat) == len(static_fee_rate_milli_msat)):
        for i in range(len(trgs)):
            trg = trgs[i]
            fee_base_src = static_fee_base_msat[2*i]
            fee_base_trg = static_fee_base_msat[2*i+1]
            fee_rate_src = static_fee_rate_milli_msat[2*i]
            fee_rate_trg = static_fee_rate_milli_msat[2*i+1]
            
            index = edges.index[(edges['src'] == src) & (edges['trg'] == trg)]
            reverse_index = edges.index[(edges['src'] == trg) & (edges['trg'] == src)]

            edges.at[index[0], 'fee_base_msat'] = fee_base_src
            edges.at[index[0], 'fee_rate_milli_msat'] = fee_rate_src
            
            edges.at[reverse_index[0], 'fee_base_msat'] = fee_base_trg
            edges.at[reverse_index[0], 'fee_rate_milli_msat'] = fee_rate_trg

        return edges
    else:
        print("Error : Invalid Input Length")
## defining action space

## defining observation space

## initially adding the first channel, which is connected to a hub




# piple-line of functions : 

'''
scripts.ln_fee:
    main -> train ->

'''