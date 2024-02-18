import numpy as np
import networkx as nx
import pandas as pd
import math
import copy
import time
from . import generating_transactions


#environment has an object of simulator
class simulator():
  def __init__(self,
               mode,
               src,trgs,channel_ids,
               active_channels, network_dictionary,
               merchants,
               transaction_types, # = [(count, amount, epsilon),...]
               node_variables,
               active_providers,
               fee_policy,
               fixed_transactions = True,
               support_onchain_rebalancing = False):
    
    self.mode = mode
    self.src = src
    self.trgs = trgs
    self.channel_id = channel_ids
    self.transaction_types = transaction_types
    self.number_of_transaction_types = len(transaction_types)
    self.merchants = merchants #list of merchants
    self.node_variables = node_variables
    self.active_providers = active_providers
    self.active_channels = active_channels
    self.fee_policy = fee_policy
    self.network_dictionary = network_dictionary
    self.fixed_transactions = fixed_transactions
    self.support_onchain_rebalancing = support_onchain_rebalancing


    self.graphs_dict = self.generate_graphs_dict(transaction_types)

    if fixed_transactions : 
      self.transactions_dict = self.generate_transactions_dict(src, transaction_types, node_variables, active_providers)
    else :
      self.transactions_dict = None
 

 

  def calculate_weight(self,edge,amount): 
    return edge[2] + edge[1]*amount 
    


  def sync_network_dictionary(self):
    for (src,trg) in self.active_channels :
      self.network_dictionary[(src,trg)] = self.active_channels[(src,trg)]
      self.network_dictionary[(trg,src)] = self.active_channels[(trg,src)]


  def generate_transactions_dict(self, src, transaction_types, node_variables, active_providers):
    transactions_dict = dict()
    for (count, amount, epsilon) in transaction_types :
        trs = generating_transactions.generate_transactions(src, amount, count, node_variables, epsilon, active_providers, verbose=False, exclude_src=True)
        transactions_dict[amount] = trs
    return transactions_dict


  def generate_graph(self, amount):
    '''Description: for each transaction_type, outputs a subgraph
                    in which the relevant transactions can flow.

    '''
    self.sync_network_dictionary()
    graph = nx.DiGraph()
    for key in self.network_dictionary :
      val = self.network_dictionary[key]
      # val[0] represents balance key[0] src and key[1] target, val[1] is fee_base and val[2] is fee_rate
      if val[0] > amount :
          graph.add_edge(key[0],key[1],weight = val[1]*amount + val[2])
    
    return graph



  def generate_graphs_dict(self, transaction_types):
    '''
    Description: generates subgraph for each transaction_type.
    ]            This is done during each iteration.
                 
    '''
    graph_dicts = dict()
    for (count, amount, epsilon) in transaction_types :
        graph = self.generate_graph(amount)
        graph_dicts[amount] = graph
    return graph_dicts


  #NOTE: why are just active channels updated in the amount graphs
  def update_graphs(self, src, trg):
      for (count,amount,epsilon) in self.transaction_types:
          graph = self.graphs_dict[amount]  
          if self.is_active_channel(src,trg):
              src_trg = self.active_channels[(src,trg)]
              src_trg_balance = src_trg[0]
              trg_src = self.active_channels[(trg,src)]
              trg_src_balance = trg_src[0]
              
              if (src_trg_balance <= amount) and (graph.has_edge(src, trg)):
                graph.remove_edge(src, trg)
              elif (src_trg_balance > amount) and (not graph.has_edge(src,trg)): 
                graph.add_edge(src, trg, weight = self.calculate_weight(src_trg, amount))
              
              if (trg_src_balance <= amount) and (graph.has_edge(trg,src)):
                graph.remove_edge(trg, src)
              elif (trg_src_balance > amount) and (not graph.has_edge(trg,src)): 
                graph.add_edge(trg, src, weight = self.calculate_weight(trg_src, amount))

          self.graphs_dict[amount] = graph
            
    
  


  def update_active_channels(self, src, trg, transaction_amount):
      if self.is_active_channel(src,trg) :
        self.active_channels[(src,trg)][0] = self.active_channels[(src,trg)][0] - transaction_amount
        self.active_channels[(trg,src)][0] = self.active_channels[(trg,src)][0] + transaction_amount
        
# Complementary
  def update_network_and_active_channels(self, action, prev_action):
    additive_channels, omitting_channels = self.delete_previous_action_differences(action, prev_action)
    self.add_to_network_and_active_channels(additive_channels)
    #add budgets if needed, format can be achieved via looking into functions
    return additive_channels, omitting_channels
  
  # Complementary
  def delete_previous_action_differences(self, action, prev_action):
    '''
    In this function, channels of new action will be omited from the network iff they are old channels
    with new assigned capacities or 0 capacity.
    '''
    
    midpoint_prev_action = len(prev_action) // 2
    if midpoint_prev_action == 0:
      return action, []
    
    # budget = 0
    midpoint_action = len(action) // 2
    additive_idx = []
    additive_bal = []
    omitting_channels = []
    # trg, bals in action which are new
    for trg, bal in zip(action[:midpoint_action], action[midpoint_action:]):
      if (trg,bal) not in zip((prev_action[:midpoint_prev_action], prev_action[midpoint_prev_action:])):
        additive_idx.append(trg)
        additive_bal.append(bal)
        
        if trg in prev_action[:midpoint_prev_action]:
          # budget += self.network_dictionary[(self.src, trg)][0]
          omitting_channels.append(trg)
          del self.network_dictionary[(self.src, trg)]
          del self.network_dictionary[(trg, self.src)]
          
          del self.active_channels[(self.src, trg)]
          del self.active_channels[(trg, self.src)]
          
    # trgs in prev_action and not in action anymore      
    for trg in prev_action[:midpoint_prev_action]:
      if trg not in action[:midpoint_action]:
          # budget += self.network_dictionary[(self.src, trg)][0]
          omitting_channels.append(trg)
          del self.network_dictionary[(self.src, trg)]
          del self.network_dictionary[(trg, self.src)]
          
          del self.active_channels[(self.src, trg)]
          del self.active_channels[(trg, self.src)]
        
    #add budget in output if needed
    return additive_idx+additive_bal, omitting_channels
    
  # Complementary
  def add_to_network_and_active_channels(self, additive_channels):
    '''
    In this function, channels of new action will be added to the network iff they are new channels
    or have new capacities assigned.
    '''
    if not additive_channels:
      return 0
    
    midpoint = len(additive_channels) // 2
    cumulative_budget = 0
    for trg, bal in zip(additive_channels[:midpoint], additive_channels[midpoint:]):
      # [balance, fee_base, fee_rate, capacity]
      self.network_dictionary[(self.src, trg)] = [bal, None, None, 2* bal]
      self.network_dictionary[(trg, self.src)] = [bal, None, None, 2* bal]
      
      self.active_channels[(self.src, trg)] = self.network_dictionary[(self.src, trg)]
      self.active_channels[(trg, self.src)] = self.network_dictionary[(trg, self.src)]
      
      cumulative_budget -= bal
      
    return cumulative_budget


  def get_local_graph(self, amount):
    self.sync_network_dictionary()
    graph = nx.DiGraph()
    for key in self.network_dictionary :
      val = self.network_dictionary[key]
      # val[0] represents balance key[0] src and key[1] target, val[1] is fee_base and val[2] is fee_rate
      graph.add_edge(key[0],key[1],weight = val[1]*amount + val[2])
    
    return graph
    

  def update_network_data(self, path, transaction_amount):
      for i in range(len(path)-1) :
        src = path[i]
        trg = path[i+1]
        if (self.is_active_channel(src, trg)) :
          self.update_active_channels(src,trg,transaction_amount)
          self.update_graphs(src, trg)
          
          
            
      
  def is_active_channel(self, src, trg):
    return ((src,trg) in self.active_channels)
        

#TODO: #19 take a look at this for first lines of step function
  def onchain_rebalancing(self, onchain_rebalancing_amount, src, trg, channel_id):
    if self.is_active_channel(src,trg) :
      self.active_channels[(src,trg)][0] += onchain_rebalancing_amount  
      self.active_channels[(src,trg)][3] += onchain_rebalancing_amount   
      self.active_channels[(trg,src)][3] += onchain_rebalancing_amount   
      self.update_graph(src, trg)
                  




  def get_path_value(self,nxpath,graph) :
    val = 0 
    for i in range(len(nxpath)-1):
      u,v = nxpath[i],nxpath[i+1]
      weight = graph.get_edge_data(u, v)['weight']
      val += weight
    return val
    



  def set_node_fee(self,src,trg,channel_id,action):
      alpha = action[0]
      beta = action[1]
      self.network_dictionary[(src,trg)][1] = alpha
      self.network_dictionary[(src,trg)][2] = beta
      if self.is_active_channel(src,trg) :
        self.active_channels[(src,trg)][1] = alpha
        self.active_channels[(src,trg)][2] = beta

  #TODO: #15 remember to utilize best approach for peer-fee setting
  def set_channels_fees(self, mode, fees, trgs) : # fees = [alpha1, alpha2, ..., alphan, beta1, beta2, ..., betan] ~ action
    n = len(self.trgs)

    if mode == 'channel_openning':
          alphas = fees[:2*n]
          betas = fees[2*n:]
          src = self.src
          for i,trg in enumerate(trgs):
              self.network_dictionary[(src,trg)][1] = alphas[2*i]
              self.network_dictionary[(src,trg)][2] = betas[2*i]
              
              self.network_dictionary[(trg,src)][1] = alphas[2*i+1]
              self.network_dictionary[(trg,src)][2] = betas[2*i+1]
              
              if self.is_active_channel(src,trg) :
                self.active_channels[(src,trg)][1] = alphas[2*i]
                self.active_channels[(src,trg)][2] = betas[2*i]
                
                self.active_channels[(trg,src)][1] = alphas[2*i+1]
                self.active_channels[(trg,src)][2] = betas[2*i+1]
      
    else:
      alphas = fees[0:n]
      betas = fees[n:]
      src = self.src
      for i,trg in enumerate(self.trgs):
          self.network_dictionary[(src,trg)][1] = alphas[i]
          self.network_dictionary[(src,trg)][2] = betas[i]
          if self.is_active_channel(src,trg) :
            self.active_channels[(src,trg)][1] = alphas[i]
            self.active_channels[(src,trg)][2] = betas[i]


  def run_single_transaction(self,
                             transaction_id,
                             transaction_amount,
                             src,trg,
                             graph):
    
    result_bit = 0
    info = {''}
    try:
      path = nx.shortest_path(graph, source=src, target=trg, weight="weight", method='dijkstra')
      
    except nx.NetworkXNoPath:
      return None,-1,{'no path'}
    val = self.get_path_value(path,graph)
    result_bit = 1
    info = {'successful'}
    return path, result_bit, info 
  
  #TODO: #20 edges should be added to and deleted from digraph
  def update_amount_graph(self,additive_channels,omitting_channels,fees):
    midpoint = len(additive_channels) // 2
    additive_ind = additive_channels[:midpoint]
    additive_bal = additive_channels[midpoint:]
    
    #removing omitting channels from amount graphs
    for key in omitting_channels :
      for amount, graph in self.graphs_dict.items():
        # print("Removing : ")
        # print("ommitting channels:",omitting_channels)
        # print("key in ommitting channels:",key)
        if graph.has_edge(self.src,key): #NOTE: why we should check? does it even added?
          graph.remove_edge(self.src,key)
        if graph.has_edge(self.src,key):
          graph.remove_edge(key,self.src)
      
    
    midpoint = len(fees) // 2
    fee_rates = fees[:midpoint]
    base_fees = fees[midpoint:]
    #adding channels with weight to relevant amount graphs
    for i in range(len(additive_ind)):
      trg, bal = additive_ind[i], additive_bal[i]
      for amount, graph in self.graphs_dict.items():
        # print("amount:",amount)
        # print("bal:",bal)
        if bal >= amount:
          # print("Adding")
          graph.add_edge(trg,self.src,weight = base_fees[2*i]*amount + fee_rates[2*i])
          graph.add_edge(self.src,trg,weight = base_fees[2*i + 1]*amount + fee_rates[2*i + 1])



    # print("Nodes of the graph: ")
    # print(graph.nodes())

    # # Print edges
    # print("Edges of the graph: ")
    # print(graph.edges())

    
          
  
        
      
    

  def preprocess_amount_graph(self,amount,action):
      if self.mode == 'channel_openning':
        return self.graphs_dict[amount]
      
      graph = self.graphs_dict[amount]
      src = self.src
      number_of_channels = len(self.trgs)
      alphas = action[0:number_of_channels]
      betas = action[number_of_channels:]
      for i,trg in enumerate(self.trgs) :
        if graph.has_edge(src, trg):
          graph[src][trg]['weight'] = alphas[i]*amount + betas[i]
      self.graphs_dict[amount] = graph
      return graph


  def run_simulation(self, action):   # action = [alphas, betas] = [alpha1, ..., alphan, beta1, ..., betan]
    output_transactions_dict = dict()
    for (count,amount,epsilon) in self.transaction_types:
        trs = self.run_simulation_for_each_transaction_type(count, amount, epsilon ,action)
        output_transactions_dict[amount] = trs
    return output_transactions_dict
   


  def run_simulation_for_each_transaction_type(self, count, amount, epsilon, action):  
    
      graph = self.preprocess_amount_graph(amount, action)

      #Run Transactions
      if self.fixed_transactions : 
        transactions = self.transactions_dict[amount]
      else :
        transactions = generating_transactions.generate_transactions(self.src, amount, count, self.node_variables, epsilon, self.active_providers, verbose=False, exclude_src=True)
      transactions = transactions.assign(path=None)
      transactions['path'] = transactions['path'].astype('object')
      for index, transaction in transactions.iterrows(): 
        src,trg = transaction['src'], transaction['trg']
        if (not src in graph.nodes()) or (not trg in graph.nodes()):
          path, result_bit, info = [], -1, {'src and/or trg dont exist in the graph'}
        else : 
          path, result_bit, info = self.run_single_transaction(transaction['transaction_id'], amount, transaction['src'], transaction['trg'], graph) 
          
        if result_bit == 1 : #successful transaction
            self.update_network_data(path, amount)
            transactions.at[index,"result_bit"] = 1
            transactions.at[index,"path"] = path

        elif result_bit == -1 : #failed transaction
            transactions.at[index,"result_bit"] = -1   
            transactions.at[index,"path"] = []
      # print("random transactions ended succussfully!")
      return transactions    #contains final result bits  #contains paths


 
 


  #getting the statistics


  
  def get_balance(self,src,trg,channel_id):
      self.sync_network_dictionary()
      return self.network_dictionary[(src,trg)][0]


  def get_capacity(self,src,trg,channel_id):
      self.sync_network_dictionary()
      return self.network_dictionary[(src,trg)][3]



  def get_network_dictionary(self):
    return self.network_dictionary


  def get_k_and_tx(self, src, trg, transactions_dict):
    k = 0
    tx = 0
    for amount, transactions in transactions_dict.items():
      temp_k = 0
      temp_tx = 0
      for index, row in transactions.iterrows():
          path = row["path"]  
          for i in range(len(path)-1) :
            if (path[i]==src) & (path[i+1]==trg) :
                temp_k += 1
      temp_tx = temp_k*amount
      k += temp_k
      tx += temp_tx
      
    return k,tx




  def get_total_fee(self,path) :
    self.sync_network_dictionary()
    alpha_bar = 0
    beta_bar = 0
    for i in range(len(path)-1):
      src = path[i]
      trg = path[i+1]
      src_trg = self.network_dictionary[(src,trg)]
      alpha_bar += src_trg[1]
      beta_bar += src_trg[2]
    return alpha_bar,beta_bar




  def get_excluded_total_fee(self,path, excluded_src, excluded_trg) :
    self.sync_network_dictionary()
    alpha_bar = 0
    beta_bar = 0
    for i in range(len(path)-1):
      src = path[i]
      trg = path[i+1]
      if (src!=excluded_src) or (trg!=excluded_trg) :
        src_trg = self.network_dictionary[(src,trg)]
        alpha_bar += src_trg[1]
        beta_bar += src_trg[2]
    return alpha_bar,beta_bar



  def find_rebalancing_cycle(self, rebalancing_type, src, trg, channel_id, rebalancing_amount):
      rebalancing_graph = self.generate_graph(rebalancing_amount)  
      cheapest_rebalancing_path = []
      
      alpha_bar = 0
      beta_bar = 0
      reult_bit = -1

      if rebalancing_type == -1 : #clockwise
          if (not src in rebalancing_graph.nodes()) or (not trg in rebalancing_graph.nodes()) or (not rebalancing_graph.has_edge(trg, src)):
            return -4,None,0,0
          if  rebalancing_graph.has_edge(src,trg):
            rebalancing_graph.remove_edge(src,trg)  
          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-1,rebalancing_amount,src,trg,rebalancing_graph) 
          if result_bit == -1 :
            return -5,None,0,0
          elif result_bit == 1 :
            cheapest_rebalancing_path.append(src)
            #alpha_bar,beta_bar = self.get_total_fee(cheapest_rebalancing_path)
            alpha_bar,beta_bar = self.get_excluded_total_fee(cheapest_rebalancing_path,src,trg)
            

      elif rebalancing_type == -2 : #counter-clockwise
          if (not trg in rebalancing_graph.nodes()) or (not src in rebalancing_graph.nodes()) or (not rebalancing_graph.has_edge(src, trg)):
            return -6,None,0,0
          if  rebalancing_graph.has_edge(trg,src):
            rebalancing_graph.remove_edge(trg,src)  
          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-2,rebalancing_amount,trg,src,rebalancing_graph) 
          if result_bit == -1 :
            return -7,None,0,0
          elif result_bit == 1 :
            cheapest_rebalancing_path.insert(0,src)
            #alpha_bar,beta_bar = self.get_total_fee(cheapest_rebalancing_path)
            alpha_bar,beta_bar = self.get_excluded_total_fee(cheapest_rebalancing_path,src,trg)
            
   
      
      return result_bit,cheapest_rebalancing_path,alpha_bar,beta_bar
      



      

  def get_coeffiecients(self,action,transactions,src,trg,channel_id, simulation_amount, onchain_transaction_fee):
        k = self.get_k(src,trg,channel_id,transactions)
        tx = simulation_amount*k
        rebalancing_fee, rebalancing_type  = self.operate_rebalancing(action[2],src,trg,channel_id,onchain_transaction_fee)
        return k,tx, rebalancing_fee, rebalancing_type

  


  def get_simulation_results(self, action, output_transactions_dict):
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


  def operate_rebalancing(self,gamma,src,trg,channel_id,onchain_transaction_fee):
    if self.support_onchain_rebalancing==True :
        return self.operate_rebalancing_with_onchain(gamma,src,trg,channel_id,onchain_transaction_fee)
    else :
        return self.operate_rebalancing_without_onchain(gamma,src,trg,channel_id)



  def operate_rebalancing_with_onchain(self,gamma,src,trg,channel_id,onchain_transaction_fee):
    fee = 0
    if gamma == 0 :
      return 0,0  # no rebalancing
    elif gamma > 0 :
      rebalancing_type = -1 #clockwise
      result_bit, cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
        cost = alpha_bar*gamma + beta_bar
        if cost <= onchain_transaction_fee:
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
        else :
          self.onchain_rebalancing(gamma,src,trg,channel_id)
          fee = onchain_transaction_fee
          rebalancing_type = -3 #onchain
      
      else :
        self.onchain_rebalancing(gamma,src,trg,channel_id)
        fee = onchain_transaction_fee
        rebalancing_type = -3

      return fee, rebalancing_type

    else :
      rebalancing_type = -2 #counter-clockwise
      gamma = gamma*-1    
      result_bit,cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
        cost = alpha_bar*gamma + beta_bar
        if cost <= onchain_transaction_fee:
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
        else :
          self.onchain_rebalancing(gamma,trg,src,channel_id)
          fee = onchain_transaction_fee
          rebalancing_type = -3 #onchain
      
      elif result_bit == -1: 
        self.onchain_rebalancing(gamma,trg,src,channel_id)
        fee = onchain_transaction_fee
        rebalancing_type = -3

      return fee, rebalancing_type



  def operate_rebalancing_without_onchain(self,gamma,src,trg,channel_id):
    fee = 0
    if gamma == 0 :
      return 0,0  # no rebalancing


    elif gamma > 0 :
      rebalancing_type = -1 #clockwise
      result_bit, cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
      elif result_bit == -1 :
          fee, rebalancing_type = 0 , -10 # clockwise failed

    else :
      rebalancing_type = -2 #counter-clockwise
      gamma = gamma*-1    
      result_bit,cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
      elif result_bit == -1: 
          fee, rebalancing_type = 0 , -20 # counter-clockwise failed

    return fee, rebalancing_type
  
  
  def get_channel_fees(self, additive_channels):
    #NOTE: the approach taken is Match-Peer approach, and for the peer, we use median approach
    bases = []
    rates = []
    midpoint = len(additive_channels) // 2
    additive_channels = additive_channels[:midpoint]
    for trg in range (len(additive_channels)):
      # print("additive_channels:", additive_channels)
      # print("trg:", trg)
      base,rate = self.fee_policy[additive_channels[trg]]
      bases.extend([base, base])
      rates.extend([rate, rate])
      
    return rates + bases
  
    


