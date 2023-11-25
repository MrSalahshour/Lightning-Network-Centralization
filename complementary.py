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
#TODO: #17 add this function to simulator, it will be called in STEP, in order to have the data when something external happens to  the process
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

# return [alpha_0,....,alpha_n,beta_0,....,beta_n] for fees.
def calculate_fees(src,trgs,graph):
  ##TODO
  return fees

## in simulator.py add a function to add our new channels to active channels.

def add_to_active_channels(self,src,action):
    midpoint = len(action)/2
    trg = action[:midpoint]
    for x,y in zip(action[:midpoint],action[midpoint:]):
        self.active_channels[(src,trg)][0] = self.active_channels[(src,trg)][0]
        self.active_channels[(trg,src)][0] = self.active_channels[(trg,src)][0]

## def to trnsform capacities indesx(selected in action space) to values of capacities







#


## initially adding the first channel, which is connected to a hub




# piple-line of functions : 

'''
scripts.ln_fee:
    main -> train ->

'''
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

# return [alpha_0,....,alpha_n,beta_0,....,beta_n] for fees.
def calculate_fees(src,trgs,graph):
  return fees

# in multchannel.py we have this func that we get action for fees but in our work we should generate these fees another way (for example mean of lcal fees).
# the fees should be like this [alpha_0,...alpha_n,beta_0,...beta_n]
def simulate_transactions(self, action, fees):
        self.simulator.set_channels_fees(fees)

        output_transactions_dict = self.simulator.run_simulation(action)
        balances, transaction_amounts, transaction_numbers = self.simulator.get_simulation_results(output_transactions_dict)

        return balances, transaction_amounts, transaction_numbers


# so in simulator.py the func get_simulation_results should be like this :
def get_simulation_results(self, output_transactions_dict):
        channels_balances = []
        channels_ks = []
        channels_txs = []
        src = self.src

        for i, trg in enumerate(self.trgs):
          k, tx = self.get_k_and_tx(src, trg, output_transactions_dict)
          balance = self.active_channels[(src,trg)][0]
          channels_ks.append(k)
          channels_txs.append(tx)
          channels_balances.append(balance)

        return channels_balances, channels_ks, channels_txs

# in simulator.py (th action space in  the previous was to choose alphas and betas and after setting these values you calculate
 # the reward but in our work we set channels an their capacity so ew should change these and then calculate the reward)
def preprocess_amount_graph(self,amount,fees):
      graph = self.graphs_dict[amount]
      src = self.src
      self.trgs = action[0:number_of_channels]
      number_of_channels = len(self.trgs)
      # ids = action[0:number_of_channels]
      # capactities = action[number_of_channels:]
      # # fee_base_msat = ???
      # # fee_rate_milli_msat = ????
      # # channel_id = ????
      # for i in range(number_of_channels):
      #   add_edge_to_digraph(graph, src, ids[i], create_edge_attributes_dict(channel_id,
      #                                                                       capactities[i], fee_base_msat,
      #                                                                       fee_rate_milli_msat, capactities[i]/2))
      alphas = fees[0:number_of_channels]
      betas = fees[number_of_channels:]

      # for fees we should use local channels mean fee values with regards to the channel.
      for i,trg in enumerate(self.trgs) :
        if graph.has_edge(src, trg) :
          graph[src][trg]['weight'] = alphas[i]*amount + betas[i]
      self.graphs_dict[amount] = graph
      return graph












## add src node to dataset. in previous work, the source node would be chosen randmoly from graph

## initially adding the first channel, which is connected to a hub

