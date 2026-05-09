# sim/parking_env.py

import random
import numpy as np


class SmartParkingEnv:

    def __init__(self):

        # Maximum parking slots per area
        self.max_slots = 10

        # Total parking areas
        self.num_areas = 3

        self.reset()

    def reset(self):

        # Parking slots in 3 parking areas
        self.slots = [
            self.max_slots,
            self.max_slots,
            self.max_slots
        ]

        # Number of incoming vehicles
        self.queue = random.randint(1, 5)

        # Simulated waiting time
        self.wait_time = 0

        # Environment state
        self.state = self._get_state()

        return self.state

    def _get_state(self):

        return np.array([
            self.slots[0],
            self.slots[1],
            self.slots[2],
            self.queue
        ], dtype=np.float32)

    def step(self, action):

        done = False

        # Random waiting time simulation
        self.wait_time = random.randint(1, 10)

        # Congestion penalty based on queue
        congestion_penalty = self.queue * 0.5

        # Check parking availability
        if self.slots[action] > 0:

            # Allocate parking slot
            self.slots[action] -= 1

            # Reward calculation
            reward = 20 - self.wait_time - congestion_penalty

        else:

            # Penalty for selecting full parking area
            reward = -20

        # Simulate new incoming vehicles
        self.queue = random.randint(1, 5)

        # Next environment state
        next_state = self._get_state()

        # End episode when all parking becomes full
        if sum(self.slots) == 0:

            done = True

        return next_state, reward, done

    def render(self):

        print("\n===== SMART PARKING STATUS =====")

        print(f"Parking Area A Slots: {self.slots[0]}")
        print(f"Parking Area B Slots: {self.slots[1]}")
        print(f"Parking Area C Slots: {self.slots[2]}")

        print(f"Vehicle Queue: {self.queue}")

        print(f"Waiting Time: {self.wait_time}")

        print("================================\n")