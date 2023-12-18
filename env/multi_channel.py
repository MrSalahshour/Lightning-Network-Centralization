import gym
from gym import spaces
from gym.spaces import *
from gym.utils import seeding
import numpy as np

from simulator.simulator import simulator
from simulator.preprocessing import generate_transaction_types

from scipy.special import softmax

#TODO: #12 self.mode
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

    Since the goal is to maximize the return in long term, reward is sum of incomes from fee payments of each channel.
    Reward scale is Sat in order to control the upperbound.

    ***Note:
    We are adding the income from each payment to balance of the corresponding channel.
    """

    def __init__(self, mode, data, max_capacity, fee_base_upper_bound, max_episode_length, number_of_transaction_types, counts, amounts, epsilons,capacity_upper_scale_bound, seed):
        # Source node
        self.src = data['src'] 
        self.prev_action = []
        # self.prev_violation = False
        #NOTE: added attribute
        self.n_nodes = len(data['nodes']) - 1 # nodes should be mines one to doesnt't include our node
        self.graph_nodes = list(data['nodes'])
        self.graph_nodes.remove(data['src'])
        print("Nodes: ",self.n_nodes)
        self.mode = mode
        self.trgs = data['trgs']
        self.n_channel = int(len(data["channel_ids"])/2)
        print('action dim:', self.n_nodes)
        

        #TODO: #18 remember that following lines are to be used in if structure after the structure of mode is being implemented
        #NOTE: following lines are for fee selection mode
        '''# Base fee and fee rate for each channel of src
        self.action_space = spaces.Box(low=-1, high=+1, shape=(2 * self.n_channel,), dtype=np.float32)
        self.fee_rate_upper_bound = 1000
        self.fee_base_upper_bound = fee_base_upper_bound

        # Balance and transaction amount of each channel
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.n_channel,), dtype=np.float32)

         # defining action space & edit of step function in multi channel
        # first n_channels are id's  of connected nodes and the seconds are corresponidg  capacities
        # add self.capacities to the fields of env class'''

        #NOTE: the following line is the total budget for the model to utilize in CHANNEL_OPENNING mode
        self.maximum_capacity = max_capacity
        #TODO #8:
        # self.capacities = [50000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
        #                         1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000] 

        # self.action_space = spaces.MultiDiscrete([self.n_nodes for _ in range(self.n_channel)] + [len(self.capacities) for _ in range(self.n_channel)])
        # self.action_space = Box(low = 0, high = max_capacity, shape=(self.n_nodes,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.n_nodes for _ in range(self.n_channel)] + [capacity_upper_scale_bound for _ in range(self.n_channel)])




        #NOTE: defining observation space
        # The observation is a ndarray with shape (n_nodes + 2*n_channels,). The first part of the observation space is 2*n_nodes with the values of 0 or 1, indicating whether we connect a channel with it or not.
        #The second part is 2*n_channels with the values corresponding to the balance of each channel and also accumulative transaction amounts in each time step.
        #Please note that the dimensions for balance and transaction amounts start from n_nodes and n_nodes + n_channels respectively. This allows us to separate the node connection information from the channel balance and transaction amounts.

        # self.max_balance = 100
        self.max_transaction_amount = 1000
        self.budget_scaling_constant = 1

        # self.observation_space = MultiDiscrete([2] * (self.n_nodes) + [self.max_balance + 1] * (self.n_nodes) + [self.max_transaction_amount + 1] * (self.n_nodes))
        self.observation_space = MultiDiscrete([2] * (self.n_nodes) + [self.maximum_capacity/self.budget_scaling_constant] * (self.n_nodes) + [self.max_transaction_amount + 1] * (self.n_nodes))

        # Initial values of each channel for fee selection mode
        # self.initial_balances = data['initial_balances']
        # self.capacities = data['capacities']
        # self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))
        
        #
        self.state = np.concatenate((np.zeros(shape=(self.n_nodes,)), np.zeros(shape=(self.n_nodes)),np.zeros(shape=(self.n_nodes))))
            
        self.time_step = 0
        self.max_episode_length = max_episode_length
        # for fee selection
        # self.balance_ratio = 0.1

        # Simulator
        transaction_types = generate_transaction_types(number_of_transaction_types, counts, amounts,
                                                       epsilons)
        self.simulator = simulator(mode=self.mode,
                                   src=data['src'],
                                   trgs=data['trgs'],
                                   channel_ids=data['channel_ids'],
                                   active_channels=data['active_channels'],
                                   network_dictionary=data['network_dictionary'],
                                   merchants=data['providers'],
                                   transaction_types=transaction_types,
                                   node_variables=data['node_variables'],
                                   active_providers=data['active_providers'],
                                   fee_policy = data["fee_policy"],
                                   fixed_transactions=False)

        self.seed(seed)


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
        # Execute one time step within the environment
        # The second part of the action is action[midpoint:]
        action_idx = action.copy()
        # print("action:",action)    
        # action = self.action_fix_index_to_capacity(self.capacities,action)
        # action = self.action_fix_to_id_format(action)
        action = self.aggregate_action(action)
        # print("action after aggregate:",action)  
        action = self.map_action_to_capacity(action)
        # print("action after map action to capacity:",action)  
        midpoint = len(action) // 2
        # updating trgs in simulator
        self.simulator.trgs = action[:midpoint]
        self.n_channel = midpoint
         
        '''
        In the following couple of lines, new channels are being added to the network, along with active_
        _channels dict and also, channels not present anymore, will be deleted
        '''
        violation = False
        #attention: budget removed
        additive_channels, ommitive_channels = self.simulator.update_network_and_active_channels(action, self.prev_action)

        
        if self.mode == "channel_openning":
            self.prev_action = action
    
            #TODO: #11 set reasonable fees
            fees = self.simulator.get_channel_fees(action)
            

            self.simulator.update_amount_graph(additive_channels, ommitive_channels,fees)
            # sum_second_part = np.sum(action[midpoint:]) 
            
            balances, transaction_amounts, transaction_numbers = self.simulate_transactions(fees,additive_channels,ommitive_channels)
            
            fees = fees[::2]
            # if self.maximum_capacity+additive_budget<0:
            #     reward = -0.01
            #     #NOTE: how to detemine this value of reward for violating(-inf cause infinite episode avg mean which is not okay and cause problems)
            #     violation = True

            # else:
                #updating the available budget
                # self.maximum_capacity += additive_budget
            reward = np.sum(np.multiply(fees[0:self.n_channel], transaction_amounts) + \
                    np.multiply(fees[self.n_channel:], transaction_numbers))
        else:
            balances, transaction_amounts, transaction_numbers = self.simulate_transactions(action)
            reward = 1e-6 * np.sum(np.multiply(action[0:self.n_channel], transaction_amounts) + \
                        np.multiply(action[self.n_channel:2 * self.n_channel], transaction_numbers))

        # Running simulator for a certain time interval
        # print("prev_action:", self.prev_action)
        # print("time_step: ", self.time_step)
            
            

        self.time_step += 1
        if self.time_step%100==0:
            print("action",action)
        info = {'TimeLimit.truncated': True if self.time_step >= self.max_episode_length else False}
        done = self.time_step >= self.max_episode_length
        # if self.prev_violation == True:
        #     done = True
        
        connected_nodes = np.zeros((self.n_nodes,))
        capacities_list = np.zeros((self.n_nodes,))
        # balances_list = np.zeros((self.n_nodes,))
        transaction_amounts_list = np.zeros((self.n_nodes,))
        # if violation == False: 
        for idx in range (len(action_idx[:midpoint])):
            connected_nodes[action_idx[idx]] = 1
            # balances_list[action_idx[idx]] = balances[idx]
            capacities_list[action_idx[idx]] = action[idx+midpoint]
            transaction_amounts_list[action_idx[idx]] = transaction_amounts[idx]

        # else:
        #     self.prev_violation = True
        #     print("....................VIOLAION IN BUDGET...........................")            
        #NOTE: what we should we do to the state if we vilaote (for now we set all connections and transactions and balance to zero)    
        if self.mode == "fee_setting":
            self.state = np.append(balances, transaction_amounts)/1000
        else:
            #changed from balanced-based to capacity-based
            self.state = np.concatenate((connected_nodes, (capacities_list)/1000,(transaction_amounts_list)/1000))

        return self.state, reward, done, info

    def simulate_transactions(self, action, additive_channels = None, ommitive_channels = None):
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
            self.state = np.concatenate((np.zeros(shape=(self.n_nodes,)), np.zeros(shape=(self.n_nodes)),np.zeros(shape=(self.n_nodes))))
            return self.state
            
        
    
    def action_fix_index_to_capacity(self,capacities,action):
        midpoint = len(action) // 2
        fixed_action = [self.graph_nodes[i] for i in action[:midpoint]]
        fixed_action.extend([capacities[i] for i in action[midpoint:]])
        return fixed_action
    
    def map_action_to_capacity(self, action):
        midpoint = len(action) // 2
        fixed_action = []
        fixed_indices = []
        #seeting up trgs from their ind
        fixed_trgs = [self.graph_nodes[i] for i in action[:midpoint]]
        #applying the softmax
        for i in range(midpoint):
            if action[i+midpoint] != 0:
                fixed_indices.append(i)
                fixed_action.append(action[i+midpoint])
        
        if len(fixed_action) != 0:
            fixed_action = softmax(np.array(fixed_action)) * self.maximum_capacity
        
        #putting into action
        output = [0] * midpoint
        for i in range(len(fixed_indices)):
            output[fixed_indices[i]] = fixed_action[i]
        
        return fixed_trgs+output
                
            
    
    def aggregate_action(self,action):
        midpoint = len(action) // 2
        unique_nodes = list(set(action[:midpoint]))
        action_bal = []
        
        for node in unique_nodes:
            agg_bal = 0
            for i in range(midpoint):
                if action[i] == node:
                    agg_bal += action[i+midpoint]
            action_bal.append(agg_bal)
        return unique_nodes + action_bal
    
    def action_fix(action):
        connected_node_ids = []
        connected_node_capacities = []
        for i, val in enumerate(action):
            if val != 0:
                connected_node_ids.append(i)
                connected_node_capacities.append(val)
        return connected_node_ids + connected_node_capacities
            





"""action space: normalise ->  [2,4,5] -> [4,16,32] -> sum = 52 - > [2,8,16] -> [10,40,80]
    softmax(x-1; x!=0) * max_budget;
    dimensionality:(10, 11)
    


observation space: omit balance
(connected_nodes, capacity, transaction amount)

functions to be updated:
ommiting, additive; updating budget


"""