import gymnasium as gym
from gymnasium import spaces
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env import PricingEnvironment
from baseline import RandomAgent   # we'll make this configurable later


class PricingGymEnv(gym.Env):

    def __init__(self, opponent):
        super().__init__()

        self.env = PricingEnvironment()
        self.opponent = opponent


        # 5 possible prices
        self.price_map = [50, 55, 60, 65, 70]

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=np.array([50, 50], dtype=np.float32),
            high=np.array([70, 70], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state = self.env.reset()
        self.last_actions = (70,70)
        self.history = []
        self.last_agent_action = 70
        return np.array(state, dtype=np.float32), {}
    
    def step(self, action):

        my_price = self.price_map[action]
        self.last_agent_action = my_price
        opponent_price = self.opponent.choose_action(
            self
        )

        next_state, rewards, done = self.env.step(
            (my_price, opponent_price)
        )

        reward = rewards[0]
        self.last_actions = (
            my_price,
            opponent_price
        )

        self.history.append(
            (
                my_price,
                opponent_price,
                reward
            )
        )

        return (
            np.array(next_state, dtype=np.float32),
            reward,
            done,
            False,
            {}
        )