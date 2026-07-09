from stable_baselines3 import DQN

from pricing_env import PricingGymEnv

from frozen_agent import FrozenRLAgent

PRICE_MAP = [50,55,60,65,70]

#First Iteration
# from baseline import RandomAgent

# env = PricingGymEnv(
#     RandomAgent([50,55,60,65,70])
# )

# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1
# )

# model.learn(
#     total_timesteps=50000
# )

# model.save("agent_A")

#Second Iteration
opponent = FrozenRLAgent(
    "agent_A",
    "dqn",
    PRICE_MAP,
    flip_state=False
)

env = PricingGymEnv(
    opponent
)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=50000
)

model.save("agent_B")

