# train.py

import random
import os
import csv
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sim.parking_env import SmartParkingEnv
from rl.dqn import DQN

# =========================
# Create Required Directories
# =========================

os.makedirs("models", exist_ok=True)

os.makedirs("plots", exist_ok=True)

os.makedirs("experiments", exist_ok=True)
# =========================
# Environment Initialization
# =========================

env = SmartParkingEnv()

state_size = 4
action_size = 3


# =========================
# DQN Model
# =========================

model = DQN(state_size, action_size)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.MSELoss()


# =========================
# Hyperparameters
# =========================

episodes = 500

gamma = 0.99

epsilon = 1.0

epsilon_decay = 0.995

epsilon_min = 0.01


# =========================
# MLflow Tracking
# =========================

mlflow.set_experiment("SmartParkingDQN")

mlflow.start_run()

# Log Hyperparameters
mlflow.log_param("episodes", episodes)

mlflow.log_param("gamma", gamma)

mlflow.log_param("epsilon_start", epsilon)

mlflow.log_param("epsilon_decay", epsilon_decay)

mlflow.log_param("learning_rate", 0.001)


# =========================
# Tracking Variables
# =========================

episode_rewards = []

episode_queue_avg = []

episode_wait_avg = []


# =========================
# Training Loop
# =========================

for episode in range(episodes):

    # Reset environment
    state = env.reset()

    state = torch.FloatTensor(state)

    done = False

    total_reward = 0

    # Per-episode metrics
    episode_queue = []

    episode_wait = []

    while not done:

        # =========================
        # ε-Greedy Action Selection
        # =========================

        if random.random() < epsilon:

            action = random.randint(0, 2)

        else:

            with torch.no_grad():

                q_values = model(state)

                action = torch.argmax(q_values).item()

        # =========================
        # Environment Step
        # =========================

        next_state, reward, done = env.step(action)

        next_state_tensor = torch.FloatTensor(next_state)

        # =========================
        # Current Q Values
        # =========================

        q_values = model(state)

        target = q_values.clone().detach()

        # =========================
        # Bellman Update
        # =========================

        with torch.no_grad():

            next_q_values = model(next_state_tensor)

            max_next_q = torch.max(next_q_values).item()

        target[action] = reward + gamma * max_next_q

        # =========================
        # Compute Loss
        # =========================

        loss = loss_fn(q_values, target)

        # =========================
        # Backpropagation
        # =========================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # =========================
        # Update State
        # =========================

        state = next_state_tensor

        # =========================
        # Store Metrics
        # =========================

        total_reward += reward

        episode_queue.append(env.queue)

        episode_wait.append(env.wait_time)

    # =========================
    # Decay Exploration
    # =========================

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # =========================
    # Save Episode Metrics
    # =========================

    avg_queue = np.mean(episode_queue)

    avg_wait = np.mean(episode_wait)

    episode_rewards.append(total_reward)

    episode_queue_avg.append(avg_queue)

    episode_wait_avg.append(avg_wait)

    # =========================
    # MLflow Metrics Logging
    # =========================

    mlflow.log_metric(
        "reward",
        total_reward,
        step=episode
    )

    mlflow.log_metric(
        "avg_queue_length",
        avg_queue,
        step=episode
    )

    mlflow.log_metric(
        "avg_wait_time",
        avg_wait,
        step=episode
    )

    mlflow.log_metric(
        "epsilon",
        epsilon,
        step=episode
    )

    print(
        f"Episode {episode+1} | "
        f"Reward: {total_reward:.2f} | "
        f"Avg Queue: {avg_queue:.2f} | "
        f"Avg Wait: {avg_wait:.2f} | "
        f"Epsilon: {epsilon:.3f}"
    )


# =========================
# Save Trained Model
# =========================

torch.save(
    model.state_dict(),
    "models/dqn_policy_v1.pth"
)

print("\nModel saved successfully!")

# Log model to MLflow
mlflow.pytorch.log_model(
    model,
    "dqn_model"
)


# =========================
# Save Experiment Results
# =========================

with open(
    "experiments/results.csv",
    "a",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        episodes,
        np.mean(episode_rewards),
        np.mean(episode_queue_avg),
        np.mean(episode_wait_avg),
        gamma,
        epsilon
    ])

print("Experiment results saved!")


# =========================
# Generate Graphs
# =========================

# -------------------------
# Reward Plot
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(episode_rewards)

plt.xlabel("Episodes")

plt.ylabel("Reward")

plt.title("Reward vs Episodes")

plt.grid(True)

plt.savefig("plots/reward_plot.png")


# -------------------------
# Waiting Time Plot
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(episode_wait_avg)

plt.xlabel("Episodes")

plt.ylabel("Average Waiting Time")

plt.title("Average Waiting Time vs Episodes")

plt.grid(True)

plt.savefig("plots/waiting_time_plot.png")


# -------------------------
# Queue Length Plot
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(episode_queue_avg)

plt.xlabel("Episodes")

plt.ylabel("Average Queue Length")

plt.title("Average Queue Length vs Episodes")

plt.grid(True)

plt.savefig("plots/queue_plot.png")


# =========================
# Log Plot Artifacts
# =========================

mlflow.log_artifact(
    "plots/reward_plot.png"
)

mlflow.log_artifact(
    "plots/waiting_time_plot.png"
)

mlflow.log_artifact(
    "plots/queue_plot.png"
)


# =========================
# Show All Graphs
# =========================

plt.show()


# =========================
# End MLflow Run
# =========================

mlflow.end_run()