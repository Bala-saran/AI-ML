import numpy as np

# Grid size and states
grid_size = 6
start_state = (0, 0)
goal_state = (5, 5)

# Define actions
actions = ["U", "D", "L", "R"]

# Define obstacles (cells agent cannot enter)
obstacles = {(1, 1), (2, 2), (3, 1), (4, 3), (2, 4)}

# Reward function
def get_reward(state):
    if state == goal_state:
        return 20
    elif state in obstacles:
        return -10
    else:
        return -1  # small penalty to encourage shortest path

# Step function (transition dynamics)
def step(state, action):
    row, col = state
    if action == "U" and row > 0:
        next_state = (row - 1, col)
    elif action == "D" and row < grid_size - 1:
        next_state = (row + 1, col)
    elif action == "L" and col > 0:
        next_state = (row, col - 1)
    elif action == "R" and col < grid_size - 1:
        next_state = (row, col + 1)
    else:
        next_state = state

    # If next_state is obstacle → stay in same state
    if next_state in obstacles:
        next_state = state

    return next_state
