import numpy as np
import pandas as pd

def q_sim(index_agent ,parameters, num_of_trials, expected_reward,
          probability_to_switch_parameters, max_change):
    
    data = { 
    'agent':[],
    'block':[],
    'trial':[],
    'card_0':[],
    'card_1':[],
    'action':[],
    'reward':[],
    'state':[],
    'state_onehot':[],
    'p_0':[],
    'drift_0':[],
    'drift_1':[],
    'drift_2':[],
    'drift_3':[],
    'Q_0':[],
    'Q_1':[],
    'Q_2':[],
    'Q_3':[],
    'alpha':[],
    'beta':[]
    }

   # set up parameters
    alpha = parameters['alpha']
    beta = parameters['beta']
    
    c_a , c_b = 0 , 0 
    cc_a , cc_b = 0 , 0

    deck_size = 4

    # q_values 
    q = np.zeros(deck_size) 
    
    block = -1
    
    for t in range(num_of_trials):
      
        if t%25 == 0:
            q = np.zeros(deck_size)
            block+=1
    
        cc_a +=1
        cc_b +=1
        
        # switch alpha 
        if cc_a > 100 and c_a < max_change and np.random.random() < probability_to_switch_parameters:
            alpha = np.random.uniform(0, 1)
            c_a += 1
            cc_a = 0

        # switch beta 
        if cc_b > 100 and c_b < max_change and np.random.random() < probability_to_switch_parameters:
            beta = np.random.uniform(0, 10)
            c_b += 1
            cc_b = 0
            
        # sample two arms out of four
        cards = np.random.choice(deck_size, 2, replace=False)

        state = 0
        
        if(cards[0] == 0 and cards[1] == 1):
            state = 0
        elif(cards[0] == 0 and cards[1] == 2):
            state = 1
        elif(cards[0] == 0 and cards[1] == 3):
            state = 2
        elif(cards[0] == 1 and cards[1] == 0):
            state = 3
        elif(cards[0] == 1 and cards[1] == 2):
            state = 4
        elif(cards[0] == 1 and cards[1] == 3):
            state = 5
        elif(cards[0] == 2 and cards[1] == 0):
            state = 6
        elif(cards[0] == 2 and cards[1] == 1):
            state = 7
        elif(cards[0] == 2 and cards[1] == 3):
            state = 8
        elif(cards[0] == 3 and cards[1] == 0):
            state = 9
        elif(cards[0] == 3 and cards[1] == 1):
            state = 10
        elif(cards[0] == 3 and cards[1] == 2):
            state = 11

        state_onhot = np.zeros(12)
        state_onhot[state] = 1

        q_cards = np.array([q[cards[0]], q[cards[1]]])

        # calc prob with softmax 
        p = np.exp(beta*q_cards) / np.sum(np.exp(beta*q_cards))

        # choose action according to prob 
        action = np.random.choice([0,1] , p=p)

        probability_reward = [(1 - expected_reward[cards[action],t]), expected_reward[cards[action],t]]
       
        # check if the trial is rewarded
        reward = np.random.choice([0,1] , p=probability_reward)     
        
        # prediction error
        prediction_error = reward - q[cards[action]]
        
        # update q values 
        q[cards[action]] = q[cards[action]] + alpha*prediction_error 
        
        # stroe data of the trial
        data['agent'].append(index_agent)
        data['block'].append(block)
        data['trial'].append(t)
        data['card_0'].append(cards[0])
        data['card_1'].append(cards[1])
        data['action'].append(action)
        data['reward'].append(reward)
        data['state'].append(state)
        data['state_onehot'].append(state_onhot)
        data['p_0'].append(p[0])
        data['drift_0'].append(expected_reward[0,t])
        data['drift_1'].append(expected_reward[1,t])
        data['drift_2'].append(expected_reward[2,t])
        data['drift_3'].append(expected_reward[3,t])
        data['Q_0'].append(q[0])
        data['Q_1'].append(q[1])
        data['Q_2'].append(q[2])
        data['Q_3'].append(q[3])
        data['alpha'].append(alpha)
        data['beta'].append(beta)

    df = pd.DataFrame(data)
    return df
