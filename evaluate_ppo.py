from stable_baselines3 import PPO
from baseline import Always70Agent
from baseline import Always50Agent
from baseline import RandomAgent
from baseline import TitForTatAgent
from pricing_env import PricingGymEnv

# env = PricingGymEnv(Always70Agent())
# env = PricingGymEnv(Always50Agent())
# env = PricingGymEnv(RandomAgent([50,55,60,65,70]))
env = PricingGymEnv(TitForTatAgent())

model = PPO.load("pricing_ppo")

#Evaluate without exploration
obs, _ = env.reset()

done = False

while not done:

    action, _ = model.predict(obs, deterministic=True)

    price = env.price_map[action]

    obs, reward, done, _, _ = env.step(action)

    print(
        f"State: {obs}, "
        f"My Price: {price}, "
        f"Reward: {reward}"
    )

# Instead of printing every round, compute averages.

episodes = 20

total_reward = 0

for ep in range(episodes):

    obs, _ = env.reset()

    done = False

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, _, _ = env.step(action)

        total_reward += reward

print("Average Reward:", total_reward/(episodes*100))