import numpy as np
import pandas as pd
from scipy.optimize import minimize

def q_fit(df,num_of_parameters_to_recover=2):

    # sample initial guess of the parameters to recover
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    initial_guess[1] = np.random.uniform(0,10)
    
    # set bounds to the recover parameters 
    bounds = [(0,1), (0,10)] 
    res = minimize(
                    fun=parameters_recovary,
                    x0=initial_guess,
                    args=df,
                    bounds=bounds,
                    method='L-BFGS-B'
    )
    return res

def parameters_recovary(parameters, df):

    # objective to minimize
    log_loss = 0 
    
    num_of_trials = len(df)
    choices_probs = np.zeros(num_of_trials)
    
    # upload data of the subject/agent
    action_list = df['action'].astype(int)
    reward_list = df['reward'].astype(np.float32)
    card_0_list = df['card_0'].astype(int)
    card_1_list = df['card_1'].astype(int)

    # set up paramters for recovary    
    alpha = parameters[0] 
    beta = parameters[1]

    deck_size = 4

    # initialize q-values
    q = np.zeros(deck_size)

    for t in range(num_of_trials):
        
        if t%25 == 0:
            q = np.zeros(deck_size)

        # get presented cards
        card_0 = card_0_list[t]
        card_1 = card_1_list[t]

        q_cards = np.array([q[card_0], q[card_1]])

        # get true first action
        action = action_list[t]
        choices_probs[t] = np.exp( beta * q_cards[action] ) / np.sum( np.exp( beta*q_cards ) ) 

        # get true reward
        reward = reward_list[t]

        chosen_card = card_0 if action == 0 else card_1
        # prediction error
        prediction_error = reward - q[chosen_card] 

        # update q_learning formula
        q[chosen_card] = q[chosen_card] + alpha*prediction_error 

        
    eps = 1e-10
    log_loss = -(np.sum(np.log(choices_probs + eps)))
    return log_loss



