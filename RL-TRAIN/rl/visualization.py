import matplotlib.pyplot as plt
import numpy as np
from rl.environment import grid_size, obstacles, start_state, goal_state

# Draw the grid with agent position and path
def draw_grid(state, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(True)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.invert_yaxis()

    # Draw obstacles
    for (r, c) in obstacles:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="black"))

    # Draw start and goal
    ax.text(start_state[1] + 0.5, start_state[0] + 0.5, "S", ha="center", va="center", fontsize=16)
    ax.text(goal_state[1] + 0.5, goal_state[0] + 0.5, "   ", ha="center", va="center", fontsize=16)

    # Draw visited path
    for p in path:
        ax.plot(p[1] + 0.5, p[0] + 0.5, "bo", markersize=10, alpha=0.3)

    # Draw agent
    ax.plot(state[1] + 0.5, state[0] + 0.5, "ro", markersize=20)

    return fig
 