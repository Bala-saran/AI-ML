import streamlit as st
import time
from rl.qlearning import train_qlearning, get_path
from rl.visualization import draw_grid
from rl.environment import start_state, goal_state

st.title("Reinforcement Learning - Grid World with Obstacles")
st.write("Agent learns to move from Start (S) to Goal ( ) while avoiding obstacles.")

if st.button("Train & Animate"):
    train_qlearning()
    path = get_path()

    placeholder = st.empty()
    for i, state in enumerate(path):
        fig = draw_grid(state, path[:i + 1])
        placeholder.pyplot(fig)
        time.sleep(0.5)

    if path[-1] == goal_state:
        st.success("Agent reached the goal!")
    else:
        st.error("Agent got stuck or failed to reach the goal.")
