from frozen_agent import FrozenRLAgent
from Env import PricingEnvironment

PRICE_MAP = [50,55,60,65,70]

env = PricingEnvironment()

A = FrozenRLAgent(
    "ppo_agent_A",
    "ppo",
    PRICE_MAP,
    flip_state=False
)

B = FrozenRLAgent(
    "ppo_agent_B",
    "ppo",
    PRICE_MAP,
    flip_state=True
)

state = env.reset()

done = False

while not done:

    price_A = A.choose_action(env)
    price_B = B.choose_action(env)

    state, rewards, done = env.step(
        (price_A, price_B)
    )

    print(
        f"State:{state}  "
        f"A:{price_A}  "
        f"B:{price_B}  "
        f"Rewards:{rewards}"
    )