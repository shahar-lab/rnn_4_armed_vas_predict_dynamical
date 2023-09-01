import numpy as np
import pandas as pd

def q_pred(df,parameters):

    # counter of the number of action classified correctly (accuracy)
    accuracy = 0 
    num_of_trials = len(df)
    choices_probs_0 = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_list = df['action'].astype(int)
    reward_list = df['reward'].astype(np.float32)
    card_0_list = df['card_0'].astype(int)
    card_1_list = df['card_1'].astype(int)

    # set up paramters of the agent     
    alpha = parameters[0] 
    beta = parameters[1]

    deck_size = 4

    # initialize q-values and preservation
    q = np.zeros(deck_size)

    for t in range(num_of_trials):
            
        if t%25 == 0:
            q = np.zeros(deck_size)

        # get presented cards
        card_0 = card_0_list[t]
        card_1 = card_1_list[t]

        q_cards = np.array([q[card_0], q[card_1]])

        p = np.exp( beta * q_cards ) / np.sum( np.exp( beta*q_cards ) ) 
        
        # predict action according max probs 
        action_predict = np.argmax(p)
        choices_probs_0[t] = p[0]

        # get true action and reward 
        action = action_list[t]
        reward = reward_list[t]

        chosen_card = card_0 if action == 0 else card_1

        # prediction error
        prediction_error = reward - q[chosen_card] 

        # update q_learning formula
        q[chosen_card] = q[chosen_card] + alpha*prediction_error 

        # cheek if prediction match the true action
        if action_predict == action_list[t]:
            accuracy+=1
            
    return (accuracy/num_of_trials), choices_probs_0 
