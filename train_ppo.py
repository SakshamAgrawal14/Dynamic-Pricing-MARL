from stable_baselines3 import PPO
from pricing_env import PricingGymEnv
from baseline import Always70Agent
from baseline import Always50Agent
from baseline import RandomAgent
from baseline import TitForTatAgent

# env = PricingGymEnv(Always70Agent())
# env = PricingGymEnv(Always50Agent())
# env = PricingGymEnv(RandomAgent([50,55,60,65,70]))
env = PricingGymEnv(TitForTatAgent())

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.95,
    verbose=1
)

model.learn(
    total_timesteps=50000
)

model.save("pricing_ppo")