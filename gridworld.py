import numpy as np
from rl_utils import index_to_state, state_to_index, check_T

A = 4
def make_transitions(X = 6, Y = 4, p = 0.9, x_goal = 0, y_goal = 0): 
    S = X * Y
    dims = (X, Y)

    T_north = np.zeros((S, S)) 
    for x in range(X): 
        for y in range(Y):
            s = state_to_index((x, y), dims)
            
            if y < Y - 1: 
                s_prime = state_to_index((x, y + 1), dims)
                T_north[s, s_prime] = p
                T_north[s, s] = 1 - p
            else: 
                T_north[s, s] = 1
    
    T_south = np.zeros((S, S)) 
    for x in range(X): 
        for y in range(Y):
            s = state_to_index((x, y), dims)
            if y > 0: 
                s_prime = state_to_index((x, y-1), dims)
                T_south[s, s_prime] = p
                T_south[s, s] = 1 - p
            else:
                T_south[s, s] = 1
    
    T_east = np.zeros((S, S))
    for x in range(X): 
        for y in range(Y):
            s = state_to_index((x, y), dims)

            if x < X - 1: 
                s_prime = state_to_index((x + 1, y), dims)
                T_east[s, s_prime] = p
                T_east[s, s] = 1 - p
            else: 
                T_east[s, s] = 1 
    
    T_west = np.zeros((S, S))
    for x in range(X): 
        for y in range(Y):
            s = state_to_index((x, y), dims)
            if x > 0: 
                s_prime = state_to_index((x - 1, y), dims)
                T_west[s, s_prime] = p
                T_west[s, s] = 1 - p
            else: 
                T_west[s, s] = 1

    T = np.array([T_north, T_south, T_east, T_west])
    T = np.transpose(T, axes = [1, 0, 2])


    # Deal with absorbing states
    s_goal = state_to_index((x_goal, y_goal), (X, Y)) 
    T[s_goal, :, :] =  0
    T[s_goal, :, s_goal] = 1
    
    return T

def make_rewards(X = 5, Y = 4, x_goal = 4, y_goal = 3, r_goal = 10): 
    S = X * Y
    dims = (X, Y)
    
    R = np.ones((S,A,S)) * -1 # -1 reward for each timestep
    
    # Goal reward 
    s_goal = state_to_index((x_goal, y_goal), dims)
    R[:, :, s_goal] = r_goal
    R[s_goal, :, :] = 0

    return R


def print_user_policy(pi, X = 10, Y = 4):
    dims = (X, Y)
    for y in reversed(range(Y)): 
        policy_string = []
        for x in range(X): 
            s = state_to_index([x, y], dims)
            action = pi[s]
            move = "-"
            if action == 0:
                move = "^"
            elif action == 1:
                move = "v"
            elif action == 2:
                move = ">"
            elif action == 3:
                move = "<"
            policy_string.append(move)
        print(policy_string)
