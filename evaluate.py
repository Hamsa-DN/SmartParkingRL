# evaluate.py

import random
import torch
import numpy as np

from sim.parking_env import SmartParkingEnv
from rl.dqn import DQN


# =========================
# Environment Setup
# =========================

env = SmartParkingEnv()

state_size = 4

action_size = 3


# =========================
# Load Trained DQN Model
# =========================

model = DQN(state_size, action_size)

model.load_state_dict(
    torch.load("models/dqn_policy_v1.pth")
)

model.eval()


# =========================
# Evaluate DQN Policy
# =========================

dqn_total_reward = 0

dqn_wait_times = []

dqn_queue_lengths = []

state = env.reset()

state = torch.FloatTensor(state)

done = False


# Prevent infinite loop
max_steps = 50

step_count = 0


while not done and step_count < max_steps:

    with torch.no_grad():

        q_values = model(state)

        action = torch.argmax(q_values).item()

    # Take action
    next_state, reward, done = env.step(action)

    # Update state
    state = torch.FloatTensor(next_state)

    # Store metrics
    dqn_total_reward += reward

    dqn_wait_times.append(env.wait_time)

    dqn_queue_lengths.append(env.queue)

    # Increment step counter
    step_count += 1


# =========================
# Fixed Baseline Policy
# =========================

fixed_total_reward = 0

fixed_wait_times = []

fixed_queue_lengths = []

slots = [10, 10, 10]


for step in range(30):

    # Fixed parking allocation strategy

    if slots[0] > 0:

        slots[0] -= 1

        reward = 5

    elif slots[1] > 0:

        slots[1] -= 1

        reward = 5

    elif slots[2] > 0:

        slots[2] -= 1

        reward = 5

    else:

        reward = -10

    # Simulated waiting time
    wait_time = random.randint(5, 15)

    # Simulated queue length
    queue_length = random.randint(3, 10)

    fixed_wait_times.append(wait_time)

    fixed_queue_lengths.append(queue_length)

    fixed_total_reward += reward


# =========================
# Final Comparison Results
# =========================

print("\n========== FINAL COMPARISON ==========")

print("\nDQN SMART PARKING POLICY")

print(f"Total Reward: {dqn_total_reward:.2f}")

print(
    f"Average Waiting Time: "
    f"{np.mean(dqn_wait_times):.2f}"
)

print(
    f"Average Queue Length: "
    f"{np.mean(dqn_queue_lengths):.2f}"
)


print("\nFIXED BASELINE POLICY")

print(f"Total Reward: {fixed_total_reward:.2f}")

print(
    f"Average Waiting Time: "
    f"{np.mean(fixed_wait_times):.2f}"
)

print(
    f"Average Queue Length: "
    f"{np.mean(fixed_queue_lengths):.2f}"
)

print("\n======================================")