from stable_baselines3 import PPO

from pricing_env import PricingGymEnv

from frozen_agent import FrozenRLAgent

PRICE_MAP = [50,55,60,65,70]

#First Iteration
# from baseline import RandomAgent

# env = PricingGymEnv(
#     RandomAgent([50,55,60,65,70])
# )

# model = PPO(
#     "MlpPolicy",
#     env,
#     verbose=1
# )

# model.learn(
#     total_timesteps=50000
# )

# model.save("ppo_agent_A")

#Second Iteration
opponent = FrozenRLAgent(
    "ppo_agent_A",
    "ppo",
    PRICE_MAP,
    flip_state=False
)

env = PricingGymEnv(
    opponent
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=50000
)

model.save("ppo_agent_B")

