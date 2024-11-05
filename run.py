from rl_utils import value_iteration, state_to_index, index_to_state
from gridworld import make_transitions, make_rewards, print_user_policy

# Gridworld parameters
X = 3
Y = 2
x_goal = 2 
y_goal = 1
r_goal = 10
p = 1.

print("Gridworld parameters:")
print("Gridworld dimensions are {} x {}".format(X, Y))
print("Goal is at ({}, {}) with value {}".format(x_goal, y_goal, r_goal))
print("Probability of moving is {}\n".format(p))


# Initialize world
T = make_transitions(X = X, Y = Y, p = p, x_goal = x_goal, y_goal = y_goal)
R = make_rewards(X = X, Y = Y, x_goal = x_goal, y_goal = y_goal, r_goal = r_goal)


# Solve for optimal policy 
gamma = 0.9
pi, Q, V = value_iteration(T, R, gamma)

print("Optimal policy found!")
print_user_policy(pi, X = X, Y = Y)
