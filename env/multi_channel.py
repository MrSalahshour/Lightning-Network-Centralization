import gym
from gym import spaces
from gym.spaces import *
from gym.utils import seeding
import numpy as np
import graph_embedding_processing
from simulator import preprocessing
from simulator.simulator import simulator
from simulator.preprocessing import generate_transaction_types
import time

import random
from collections import Counter
import networkx as nx


from scipy.special import softmax
import math


class FeeEnv(gym.Env):
    """
    ### Description

    This environment corresponds to the LIGHTNING NETWORK simulation. A source node is chosen and a local network
    around that node with radius 2 is created and at each time step, a certain number of transitions are being simulated.

    ### Scales

    We are using the following scales for simulating the real world Lightning Network:

    - Fee Rate: msat                                      - Base Fee: msat
    - Transaction amounts: sat                            - Reward(income): msat
    - Capacity: sat                                       - Balance: sat

    ### Action Space

    The action is a `ndarray` with shape `(2*n_channel,)` which can take values `[0,upper bound]`
    indicating the fee rate and base fee of each channel starting from source node.

    | dim       | action                 | dim        | action                |
    |-----------|------------------------|------------|-----------------------|
    | 0         | fee rate channel 0     | 0+n_channel| fee base channel 0    |
    | ...       |        ...             | ...        |         ...           |
    | n_channel | fee rate last channel  | 2*n_channel| fee base last channel |

    ### Observation Space

    The observation is a `ndarray` with shape `(2*n_channel,)` with the values corresponding to the balance of each
    channel and also accumulative transaction amounts in each time steps.

    | dim       | observation            | dim        | observation                 |
    |-----------|------------------------|------------|-----------------------------|
    | 0         | balance channel 0      | 0+n_channel| sum trn amount channel 0    |
    | ...       |          ...           | ...        |            ...              |
    | n_channel | balance last channel   | 2*n_channel| sum trn amount last channel |

    ### Rewards

    Since the goal is to maximize the return in the long term, the reward is the sum of incomes from fee payments of each channel.
    The reward scale is Sat to control the upper bound.

    ***Note:
    We are adding the income from each payment to the balance of the corresponding channel.
    """

    def __init__(self, mode, data, max_capacity, fee_base_upper_bound, max_episode_length,
                  number_of_transaction_types, counts, amounts, epsilons, capacity_upper_scale_bound,
                    seed, LN_graph):
        # Source node\
        self.embedder = None
        self.total_channel_changes = 0
        self.LN_graph = LN_graph
        self.data = data
        self.prev_action = []
        self.providers = data['providers']
        self.src = self.data['src']
        self.n_channel = data['n_channels']
        self.average_transaction_amounts =  amounts[1]
        self.transaction_types = generate_transaction_types(number_of_transaction_types, counts, amounts,
                                                       epsilons)
        self.mode = mode
        self.current_graph = self.set_new_graph_environment()
        
        self.n_nodes = len(self.data['nodes']) - 1 # nodes should be minus one to not include our node
       
        print("number of nodes: ",self.n_nodes)
        self.graph_nodes = list(self.data['nodes'])
        if self.src in self.graph_nodes:
            self.graph_nodes.remove(self.src)
        
        self.embedding_mode = "feather"
        self.embedding_size = 108
        print('action dim:', self.n_nodes)
        

        #NOTE: The following lines are for fee selection mode
        '''# Base fee and fee rate for each channel of src
        self.action_space = spaces.Box(low=-1, high=+1, shape=(2 * self.n_channel,), dtype=np.float32)
        self.fee_rate_upper_bound = 1000
        self.fee_base_upper_bound = fee_base_upper_bound

        # Balance and transaction amount of each channel
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.n_channel,), dtype=np.float32)

        # Defining action space & editing of step function in multi-channel
        # first n_channels are id's  of connected nodes and the seconds are corresponding  capacities
        # add self.capacities to the fields of env class'''

        self.maximum_capacity = max_capacity

        self.action_space = spaces.MultiDiscrete([self.n_nodes for _ in range(self.n_channel)] + [capacity_upper_scale_bound for _ in range(self.n_channel)])

        self.observation_space = Dict({
            'capacities': Box(low=0, high=capacity_upper_scale_bound, shape=(self.n_nodes,)),
            'transaction_amounts': Box(low=0, high=np.inf, shape=(self.n_nodes,)),
            'graph_embedding': Box(low=-np.inf, high=np.inf, shape=(self.embedding_size,))
        })

        #NOTE: Initial values of each channel for fee selection mode
        # self.initial_balances = data['initial_balances']
        # self.capacities = data['capacities']
        # self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))
        

        self.graph_embedding =self.get_new_graph_embedding(self.current_graph,self.embedding_mode)

        self.state = {
            'capacities': np.zeros(self.n_nodes),
            'transaction_amounts': np.zeros(self.n_nodes),
            'graph_embedding': self.graph_embedding
        }
            
        self.time_step = 0
        self.max_episode_length = max_episode_length
        
        self.transaction_amounts_list = np.zeros((self.n_nodes,))

        
        #NOTE: for fee selection
        # self.balance_ratio = 0.1

        # Simulator
        
        # self.simulator = simulator(mode=self.mode,
        #                            src=self.src,
        #                            trgs=self.data['trgs'],
        #                            channel_ids=self.data['channel_ids'],
        #                            active_channels=self.data['active_channels'],
        #                            network_dictionary=self.data['network_dictionary'],
        #                            merchants=self.providers,
        #                            transaction_types=self.transaction_types,
        #                            node_variables=self.data['node_variables'],
        #                            active_providers=self.data['active_providers'],
        #                            fee_policy = self.data["fee_policy"],
        #                            fixed_transactions=False)
        # self.seed(seed)
        
        



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #NOTE: might need to add fees attr to set_channel_fees in the following
    # def step(self, action, rescale=True):
    #     # Rescaling the action vector (fee selection mode)
    #     # if rescale:
    #     #     action[0:self.n_channel] = .5 * self.fee_rate_upper_bound * action[0:self.n_channel] + \
    #     #                                .5 * self.fee_rate_upper_bound
    #     #     action[self.n_channel:2 * self.n_channel] = .5 * self.fee_base_upper_bound * action[
    #     #                                                                                  self.n_channel:2 * self.n_channel] + \
    #     #                                                 .5 * self.fee_base_upper_bound

    #     # Running simulator for a certain time interval
    #     balances, transaction_amounts, transaction_numbers = self.simulate_transactions(action)
    #     self.time_step += 1

    #     reward = 1e-6 * np.sum(np.multiply(action[0:self.n_channel], transaction_amounts) + \
    #                     np.multiply(action[self.n_channel:2 * self.n_channel], transaction_numbers))

    #     info = {'TimeLimit.truncated': True if self.time_step >= self.max_episode_length else False}

    #     done = self.time_step >= self.max_episode_length

    #     self.state = np.append(balances, transaction_amounts)/1000

    #     return self.state, reward, done, info
    
    
    def step(self, action):
        
        action = self.aggregate_and_standardize_action(action)
                
        raw_action = action.copy()
        
        if self.time_step%100==0:
            print("action: ",action,"time step: ",self.time_step)
  
    
        action = self.map_action_to_capacity(action)
        midpoint = len(action) // 2

        self.simulator.trgs = action[:midpoint]
        self.n_channel = midpoint
         
        '''
        In the following couple of lines, new channels are being added to the network, along with active_
        _channels dict and also, channels not present anymore, will be deleted
        '''
        #attention: budget removed
        additive_channels, ommitive_channels = self.simulator.update_network_and_active_channels(action, self.prev_action)

        self.prev_action = action

        fees = self.simulator.get_additive_channel_fees(action)
        
        self.simulator.update_amount_graph(additive_channels, ommitive_channels,fees)

        fees_to_use_for_reward = fees[::2]
        if self.time_step + 1 == self.max_episode_length :
            reward = 0
            
            for i in range(self.max_episode_length):
                balances, transaction_amounts, transaction_numbers = self.simulate_transactions(fees,additive_channels)
                reward += 1e-6 *(np.sum(np.multiply(fees_to_use_for_reward[0:self.n_channel], transaction_amounts) + \
                        np.multiply(fees_to_use_for_reward[self.n_channel:], transaction_numbers)))

        else:
            balances, transaction_amounts, transaction_numbers = self.simulate_transactions(fees,additive_channels)
            reward = 1e-6 *(np.sum(np.multiply(fees_to_use_for_reward[0:self.n_channel], transaction_amounts ) + \
                    np.multiply(fees_to_use_for_reward[self.n_channel:], transaction_numbers)))

        self.time_step += 1
        
        info = {'TimeLimit.truncated': True if self.time_step >= self.max_episode_length else False}
        done = self.time_step >= self.max_episode_length

        capacities_list = np.zeros((self.n_nodes,))
        
        for idx, in range(midpoint):
            capacities_list[raw_action[idx]] = raw_action[idx+midpoint]
            self.transaction_amounts_list[raw_action[idx]] += transaction_amounts[idx]
  
        
        self.state = {
            'capacities': capacities_list,
            'transaction_amounts': self.transaction_amounts_list,
            'graph_embedding': self.graph_embedding
        } 

        return self.state, reward, done, info

    def simulate_transactions(self, action, additive_channels = None):
        #NOTE: fees set in the step, now will be added to network_dict and active_channels
        self.simulator.set_channels_fees(self.mode,action,additive_channels[:len(additive_channels)//2])

        output_transactions_dict = self.simulator.run_simulation(action)
        balances, transaction_amounts, transaction_numbers = self.simulator.get_simulation_results(action,
                                                                                                   output_transactions_dict)

        return balances, transaction_amounts, transaction_numbers
    

    def reset(self):
        print('episode ended!')
        self.time_step = 0
        if self.mode == 'fee_setting':
            self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))
            return np.array(self.state, dtype=np.float64)
        
        else:
            self.prev_action = []
            self.current_graph = self.set_new_graph_environment()
            self.graph_embedding = self.get_new_graph_embedding(self.current_graph,self.embedding_mode)
            self.state = {
                'capacities': np.zeros(self.n_nodes),
                'transaction_amounts': np.zeros(self.n_nodes),
                'graph_embedding': self.graph_embedding #sample new embedding
            }
            self.transaction_amounts_list = np.zeros((self.n_nodes,))
            return self.state 
        
    def is_weighted(self,G):
        return all('weight' in G[u][v] for u, v in G.edges())


    def action_fix_index_to_capacity(self,capacities,action):
        midpoint = len(action) // 2
        fixed_action = [self.graph_nodes[i] for i in action[:midpoint]]
        fixed_action.extend([capacities[i] for i in action[midpoint:]])
        return fixed_action
    
    def map_action_to_capacity(self, action):
        midpoint = len(action) // 2
        fixed_action = []
        #seeting up trgs from their ind
        fixed_trgs = [self.graph_nodes[i] for i in action[:midpoint]]
        
        if len(action) != 0:
            fixed_action = softmax(np.array(action[midpoint:])) * self.maximum_capacity      
      
        return fixed_trgs+fixed_action.tolist()
                
            
    
    def aggregate_and_standardize_action(self,action):
        midpoint = len(action) // 2
        unique_nodes = list(set(action[:midpoint]))
        nonzero_unique_nodes = []
        action_bal = []
          
        for node in unique_nodes:
            agg_bal = 0
            for i in range(midpoint):
                if action[i] == node:
                    agg_bal += action[i+midpoint]
            if agg_bal !=0:
                nonzero_unique_nodes.append(node)
                action_bal.append(agg_bal)
        
        #Standardizing the balances
        bal_gcd = math.gcd(*action_bal)
        action_bal = [balance/bal_gcd for balance in action_bal]
        
        return nonzero_unique_nodes + action_bal
    
    def action_fix(action):
        connected_node_ids = []
        connected_node_capacities = []
        for i, val in enumerate(action):
            if val != 0:
                connected_node_ids.append(i)
                connected_node_capacities.append(val)
        return connected_node_ids + connected_node_capacities
            


    def get_local_graph(self,scale):
        return self.current_graph
    
    # def count_identical_items(self,list_of_sets):
    #     n = len(list_of_sets)
    #     identical_counts = [[0]*n for _ in range(n)]
    #     for i in range(n):
    #         for j in range(i+1, n):
    #             identical_counts[i][j] = len(list_of_sets[i] & list_of_sets[j])
    #             print("identical_count","i:",i,"j:",j,identical_counts[i][j])
    #     return identical_counts

    def sample_graph_environment(self):
        
        random.seed()
        sampled_graph = preprocessing.get_fire_forest_sample(self.LN_graph,self.n_nodes)    
        return list(sampled_graph.nodes())
    
    def evolve_graph(self, G, number_of_new_channels):

        transformed_graph = self.add_edges(G, number_of_new_channels)

        return transformed_graph
    
    def add_edges(G, k): #Note we have to add edge feautres and update network dictionary
        #(and maybe other places that get influenced by adding new channels)

        fees = self.simulator.get_additive_channel_fees(action)
        
        self.simulator.update_amount_graph(additive_channels, ommitive_channels,fees)

        
        # Calculate degree of each node
        degrees = dict(G.degree())
        
        # Create a distribution based on logarithm of degree of nodes
        log_degrees = np.log(np.array(list(degrees.values())) + 1)
        distribution = log_degrees / np.sum(log_degrees)
        
        nodes = list(G.nodes())
        for _ in range(k):
            # Sample two nodes based on the distribution
            node1, node2 = np.random.choice(nodes, size=2, replace=False, p=distribution)

            
            # Add edge if not already exists
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2)
                G.add_edge(node2, node1)  # For directed graph, add both way edges

        return G
    

    def count_unique_graphs(self, graphs):
        """
        Counts the number of unique graphs in a list of graphs.

        Args:
            graphs (list): A list of nx.Graph objects.

        Returns:
            int: The number of unique graphs in the list.
        """

        # Convert the list of graphs to a set to remove duplicates.
        graph_set = set(graphs)

        # Return the size of the set, which represents the number of unique graphs.
        return len(graph_set)

    def get_new_graph_embedding(self, G, embedding_mode):

        if embedding_mode == 'feather':
            if self.embedder == None:
                self.embedder = graph_embedding_processing.get_feather_embedder()
                model , graph_embedding = graph_embedding_processing.get_feather_embedding(self.embedder, G)
                self.embedder = model
            else:
                model , graph_embedding = graph_embedding_processing.get_feather_embedding(self.embedder, G)
                self.embedder = model

            return graph_embedding
        
        elif embedding_mode == 'geo_scattering':
            return graph_embedding_processing.get_geo_scattering_embedding(G)

        elif embedding_mode == 'LDP':
            return graph_embedding_processing.get_LDP_embedding(G)
        
        elif embedding_mode == 'GL2Vec':
            return graph_embedding_processing.get_GL2Vec_embedding(G)

        elif embedding_mode == 'Graph2Vec':
            return graph_embedding_processing.get_Graph2Vec_embedding(G)

        else:
            print("Unknown embedding mode")
        
        return None
    
    def make_graph_weighted(self,graph, amount):
        #weights  based on satoshi
        for u, v, data in graph.edges(data=True):
            fee_rate = data.get('fee_rate_milli_msat', 0)
            fee_base = data.get('fee_base_msat', 0)
            weight = 1e-6 * (fee_rate * amount + fee_base *1000)
            graph[u][v]['weight'] = weight
            
        return graph
        
    def set_new_graph_environment(self):

        sub_nodes = self.sample_graph_environment()

        network_dictionary, sub_providers, sub_edges, sub_graph = preprocessing.get_sub_graph_properties(self.LN_graph,sub_nodes,self.providers)
        
        sub_graph = self.make_graph_weighted(sub_graph, amount = self.average_transaction_amounts)

        active_channels = preprocessing.create_active_channels(network_dictionary, [])

        try:
            node_variables, active_providers, _ = preprocessing.init_node_params(sub_edges, sub_providers, verbose=True)
        except:
            node_variables, active_providers = None
            


        self.data['active_channels'] = active_channels
        self.data['network_dictionary'] = network_dictionary
        self.data['node_variables'] = node_variables
        self.data['active_providers'] = active_providers
        self.data['nodes'] = sub_nodes

        self.graph_nodes = list(self.data['nodes'])
        if self.src in self.graph_nodes:
            self.graph_nodes.remove(self.src)



        self.simulator = simulator(mode=self.mode,
                                   src=self.src,
                                   trgs=self.data['trgs'],
                                   channel_ids=self.data['channel_ids'],
                                   active_channels=self.data['active_channels'],
                                   network_dictionary=self.data['network_dictionary'],
                                   merchants=self.providers,
                                   transaction_types=self.transaction_types,
                                   node_variables=self.data['node_variables'],
                                   active_providers=self.data['active_providers'],
                                   fee_policy = self.data["fee_policy"],
                                   fixed_transactions=False)
        
        return sub_graph
    
    # one channel distruction would cost 10^7 msat
    def calculate_penalty(self,prev_action,action,channel_distruction_penalty = 10000):
        midpoint = len(action)//2
        # max_episode_to_get_real_penalty = 100000

        # penalty = min(channel_distruction_penalty,self.time_step*channel_distruction_penalty/max_episode_to_get_real_penalty)
        penalty = channel_distruction_penalty
        # penalty = 0


        diff_items = self.min_changes(prev_action[:midpoint], action[:midpoint])
        self.total_channel_changes += diff_items

        return diff_items * penalty

    def min_changes(self, list1, list2):
        # Create multisets
        multiset1 = Counter(list1)
        multiset2 = Counter(list2)

        # Find the difference between the two multisets
        diff = multiset1 - multiset2

        # The total number of changes is the sum of the differences
        total_changes = sum(diff.values())

        return total_changes
    

