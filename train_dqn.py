from stable_baselines3 import DQN
from baseline import Always70Agent
from baseline import Always50Agent
from baseline import RandomAgent
from baseline import TitForTatAgent
from pricing_env import PricingGymEnv

# env = PricingGymEnv(Always70Agent())
# env = PricingGymEnv(Always50Agent())
# env = PricingGymEnv(RandomAgent([50,55,60,65,70]))
env = PricingGymEnv(TitForTatAgent())

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=500,
    batch_size=64,
    gamma=0.95,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1
)

model.learn(total_timesteps=50000)

model.save("pricing_dqn")

