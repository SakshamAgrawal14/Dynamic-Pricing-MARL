import numpy as np
from stable_baselines3 import DQN, PPO


class FrozenRLAgent:

    def __init__(self, model_path, algorithm, price_map, flip_state=False):

        if algorithm.lower() == "dqn":
            self.model = DQN.load(model_path)

        elif algorithm.lower() == "ppo":
            self.model = PPO.load(model_path)

        else:
            raise ValueError("Algorithm must be 'dqn' or 'ppo'")

        self.price_map = price_map
        self.flip_state = flip_state

    def choose_action(self, env):

        if hasattr(env, "env"):
            state = env.env.current_state
        else:
            state = env.current_state

        if self.flip_state:
            state = (state[1], state[0])

        obs = np.array(state, dtype=np.float32)

        action, _ = self.model.predict(
            obs,
            deterministic=True
        )

        return self.price_map[action]