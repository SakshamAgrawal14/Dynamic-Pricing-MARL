from stable_baselines3 import DQN
from pricing_env import PricingGymEnv

env = PricingGymEnv()

model = DQN.load("pricing_dqn")

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