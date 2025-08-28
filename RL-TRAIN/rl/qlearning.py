import random
from rl.environment import grid_size, actions, start_state, goal_state, step, get_reward, obstacles

# Q-table initialization
Q = {}
for row in range(grid_size):
    for col in range(grid_size):
        if (row, col) not in obstacles:
            Q[(row, col)] = {a: 0.0 for a in actions}

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

# Training using Q-learning
def train_qlearning():
    for ep in range(episodes):
        state = start_state
        while state != goal_state:
            # epsilon-greedy policy
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get)

            next_state = step(state, action)
            reward = get_reward(next_state)

            Q[state][action] += alpha * (
                reward + gamma * max(Q[next_state].values()) - Q[state][action]
            )
            state = next_state

# Get the learned path
def get_path():
    state = start_state
    path = [state]
    visited = set()
    while state != goal_state and len(path) < 100:  # prevent infinite loop
        action = max(Q[state], key=Q[state].get)
        state = step(state, action)
        if state in visited:  # if looping, break
            break
        path.append(state)
        visited.add(state)
    return path
