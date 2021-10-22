import numpy as np
import pandas as pd
import time
from Environment import Action, State, SupplyChainEnvironment
from datetime import date
from datetime import datetime
import os 
# We use (s, Q)-policy as a baseline
# => Order your Economic Order Quantity Q, every time your inventory
# position drops below s (Reorder Point or Safety Stock).


class SQPolicy(object):
    def __init__(self, factory_safety_stock, factory_reorder_amount, safety_stock, reorder_amount) -> None:
        self.factory_safety_stock = factory_safety_stock
        self.factory_reorder_amount = factory_reorder_amount
        self.safety_stock = safety_stock
        self.reorder_amount = reorder_amount

    def select_action(self, state: State) -> Action:
        action = Action(state.warehouse_num)

        for w in range(state.warehouse_num):
            if state.warehouse_stock[w] < self.safety_stock[w]:
                action.shippings_to_warehouses[w] = self.reorder_amount[w]

        if state.factory_stock - np.sum(action.shippings_to_warehouses) < self.factory_safety_stock:
            action.production_level = self.factory_reorder_amount
        else:
            action.production_level = 0

        return action


def simulate_episode(policy: SQPolicy, log=False, episode=1, log_file=None) -> list:
    env = SupplyChainEnvironment()
    state = env.initial_state()
    transitions = []
    expanded_data = []
    total_reward = 0
    done = False
    num=env.warehouse_num
    for t in range(env.T):
        action = policy.select_action(state)
        state, reward, done = env.step(state, action, log)
        total_reward += reward
        #print('Here--------')
        #print('warehouses:',num)
        transitions.append([state, action, reward])
        
        #values=state[0:num+1]
        values=[]
        action_values=[]
        for i in range(num+1): #0,1,2,3
            if i== 0:
                values.append(state.factory_stock)
                action_values.append(action.production_level)
            else:
                values.append(state.warehouse_stock[i-1])
                action_values.append(action.shippings_to_warehouses[i-1])
        
        expanded_data.append([episode, t, *values,
                             *action_values,
                             reward, total_reward])
        whstock=['warehouse_stock_'+str(i) for i in range(0,num)]
        shwarehouse=['shipping_to_warehouse_'+str(i) for i in range(0,num)]
        columns_names=['episode', 't', 'factory_stock',*whstock,'production_level',*shwarehouse,'timestep_reward', 'total_reward']

        #expanded_data.append([episode, t, state.factory_stock,
        #                     state.warehouse_stock[0], state.warehouse_stock[1],
        #                     action.production_level,
        #                     action.shippings_to_warehouses[0], action.shippings_to_warehouses[1],
        #                     reward, total_reward])

    if log:
        
        df = pd.DataFrame(expanded_data, columns=columns_names)
        #df = pd.DataFrame(expanded_data, columns=['episode', 't', 'factory_stock',
        #              'warehouse_stock_0', 'warehouse_stock_1', 'production_level', 'shippings_to_warehouses_0', 'shippings_to_warehouses_1', 'timestep_reward', 'total_reward'])
        if log_file is None:
            dirname = os.path.dirname(os.path.abspath(__file__))
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%Y-%m-%d-%H-%M")
            log_file = './results/output/transitions/sQ/transitions-'+str(dt_string)+'.csv'
            csvfilename = os.path.join(dirname, 'results','transitions','sQ',log_file)
        df.to_csv(log_file, index=False)

    return transitions, expanded_data,columns_names


def simulate(policy: SQPolicy, num_episodes: int, log=False, log_file=None) -> list:
    returns_trace = []
    expanded_data = []
    for episode in range(num_episodes):
        return_trace, exp_data,columns_names = simulate_episode(policy, log=False, episode=episode, log_file=log_file)

        returns_trace.append(sum(np.array(return_trace).T[2]))
        expanded_data += (exp_data)

    if log:
        #columns_names=exp_data.columns
        #df = pd.DataFrame(expanded_data, columns=['episode', 't', 'factory_stock',
        #              'warehouse_stock_0', 'warehouse_stock_1', 'production_level', 'shippings_to_warehouse_0', 'shippings_to_warehouse_1', 'timestep_reward', 'total_reward'])
        df= pd.DataFrame(expanded_data, columns=columns_names)
        
        if log_file is None:
            dirname = os.path.dirname(os.path.abspath(__file__))
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%Y-%m-%d-%H-%M")
            log_file = './results/output/transitions/sQ/transitions-'+str(dt_string)+'.csv'
            csvfilename = os.path.join(dirname, 'results','transitions','sQ',log_file)
        df.to_csv(log_file, index=False)

    return returns_trace,expanded_data
