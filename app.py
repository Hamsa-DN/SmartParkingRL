# app.py

import streamlit as st
from PIL import Image

st.title("Smart Parking Allocation System")

st.subheader("SDG 11 – Sustainable Cities and Communities")

st.write("""
This project uses Deep Reinforcement Learning (DQN)
to optimize smart parking allocation and reduce
vehicle waiting time and congestion.
""")

# Reward Graph
st.header("Reward vs Episodes")

reward_img = Image.open("plots/reward_plot.png")

st.image(reward_img)

# Waiting Time Graph
st.header("Waiting Time Analysis")

wait_img = Image.open("plots/waiting_time_plot.png")

st.image(wait_img)

# Queue Graph
st.header("Queue Length Analysis")

queue_img = Image.open("plots/queue_plot.png")

st.image(queue_img)

st.success("DQN Smart Parking System Running Successfully!")