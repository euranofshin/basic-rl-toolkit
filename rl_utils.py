import numpy as np


def state_to_index(s, dim_sizes): 
    '''
    Given a discrete state (e.g. [1, 1, 0]) returns single index (e.g. the index of the flattened state space)

    Inputs: 
    - s: a tuple or array representing the state. 
    - dim_sizes: the size of each (discrete) dimension

    Output: an int representing the (flattened) state index
    '''
    return np.ravel_multi_index(s, dim_sizes)
def index_to_state(ind, dim_sizes): 
    '''
    Given a single index (e.g. the index of the flattened state space), returns a discrete state tuple
    
    Inputs: 
    - s: an int representing the (flattened) state index 
    - dim_sizes: the size of each (discrete) dimension

    Output: a tuple representing the state
    '''
    return np.unravel_index(ind, dim_sizes)  

def check_T(T_true):
    '''
    Print the sum of each s x a row in the transition matrix. The sum of probabilities over next state should be 1. 
    '''
    for a in range(T_true.shape[1]): 
        print(T_true[:, a, :].sum(axis =1))

def V_to_Q(V, T, R, gamma): 
    '''
    Turns a value function V into an action-value function Q
    
    '''
    if len(R.shape) > 2: 
        future_mat = np.array([[T[s, a, :].dot(gamma * V + R[s, a, :]) for a in range(T.shape[1])] for s in range(T.shape[0])])
    else:
        future_mat = R + gamma * np.hstack([T[:, a, :].dot(V) for a in range(T.shape[1])])

    return future_mat 


def value_iteration(T, R, gamma, delta = 0.1, verbose = False): 
    '''
    Value iteration to solve for the optimal value function. 

    Inputs: 
    - T: numpy array of dimensionality [S, A, S]
    - R: numpy array of dimensionality [S, A, S]
    - gamma: float discount factor from (0, 1)
    
    Outputs:
    - pi: numpy array of dimensionality [S]. Each element is the optimal action at state s.  
    - Q: numpy array of dimensionality [S, A]. Each element is the action-value at a given state. 
    - V: numpy array of dimensionality [S]. Each element is the value at a given state. 
    '''
    n_actions = T.shape[1]
    n_states = T.shape[0]
    V = np.zeros((n_states, ))
    Q = np.zeros((n_states, n_actions))
    max_change = delta
    i = 0
    while max_change >= delta: 
        max_change = 0.
        for s in range(Q.shape[0]):
            for a in range(n_actions): 
                '''
                indices, s_next_probs = T.get_transition_probabilities(s, a)
                r_next = np.array([r(s, a, s_next) for s_next in indices])
                Q[s, a] = np.dot(gamma * V[indices] + r_next, s_next_probs)
                '''
                Q[s, a] = np.dot(T[s, a, :], gamma * V + R[s, a, :])
                
            v = V[s]
            V[s] = np.amax(Q[s, :])
            max_change = max(max_change, np.abs(v - V[s]))
           

        if verbose: 
            print("iteration {}: max change {}".format(i, max_change))
        i+=1  

    pi = np.argmax(Q, axis = 1)

    return pi, Q, V



def policy_iteration(T, R, gamma, max_iters = 10):
    '''
    T is S x A x S
    R is S x A
    '''
    S = T.shape[0]
    A = T.shape[1]
    V = np.zeros((S, 1))
    Q = np.zeros((S, A))
    pi = np.zeros((S, A)) # S x A
    pi[:, 0] = 1

    policy_stable = False
    i = 0
    while not policy_stable and i < max_iters:     
        #print(pi, V)
        #print(V, "\n")
        
        if len(R.shape) > 2: 
            # Policy evaluation
            T_pi = np.vstack([pi[s, :].dot(T[s, :, :]) for s in range(S)]) # S x S
            R_T = np.array([[T[s, a, :].dot(R[s, a, :]) for a in range(A)] for s in range(S)])
            R_pi = np.hstack([R_T[s, :].dot(pi[s, :].T) for s in range(S)]) # S x 1
            V = np.linalg.inv(np.eye(S) - (gamma * T_pi)).dot(R_pi)

            # Policy improvement 
            policy_stable = True
            for s in range(S):
                Q[s, :] = [T[s, a, :].dot(R[s, a, :] + (gamma * V).flatten()) for a in range(A)]
                old_action = np.argmax(pi[s, :])
                new_action = np.argmax(Q[s, :])
                pi[s, :] = 0
                pi[s, new_action] = 1
                if old_action != new_action: 
                    policy_stable = False   
        else: 
            # Policy evaluation
            T_pi = np.vstack([pi[s, :].dot(T[s, :, :]) for s in range(S)]) # S x S
            R_pi = np.hstack([R[s, :].dot(pi[s, :].T) for s in range(S)]) # S x 1
            V = np.linalg.inv(np.eye(S) - (gamma * T_pi)).dot(R_pi)

            # Policy improvement 
            policy_stable = True
            for s in range(S): 
                Q[s, :] = [T[s, a, :].dot(R[s, a] + (gamma * V)) for a in range(A)]
                old_action = np.argmax(pi[s, :])
                new_action = np.argmax(Q[s, :])
                pi[s, :] = 0
                pi[s, new_action] = 1
                if old_action != new_action: 
                    policy_stable = False   
        i+=1
    return pi.argmax(axis = 1), Q, V.reshape(-1, 1)


