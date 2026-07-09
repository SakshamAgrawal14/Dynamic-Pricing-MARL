from Env import PricingEnvironment
from Agent import QLearningAgent
import matplotlib.pyplot as plt
from baseline import RandomAgent, Always50, Always70, tit_for_tat

env = PricingEnvironment()

# -------------------------
# Choose agents
# -------------------------

agent_A = QLearningAgent(env.actions)
agent_B = QLearningAgent(env.actions)

# Examples

# agent_A = RandomAgent(env.actions)
# agent_A = Always50Agent()
# agent_A = Always70Agent()
# agent_A = TitForTatAgent()

num_episodes = 1000

prices_A = []
prices_B = []

action_rewards_A = {
    50: [],
    55: [],
    60: [],
    65: [],
    70: []
}

action_rewards_B = {
    50: [],
    55: [],
    60: [],
    65: [],
    70: []
}

for episode in range(num_episodes):
    state = env.reset()
    done = False
    # B = 60  --> S

    while not done:
        if isinstance(agent_A, QLearningAgent):
            action_A = agent_A.choose_action(state)

        else:
            action_A = agent_A.choose_action(env)

        if isinstance(agent_B, QLearningAgent):
            action_B = agent_B.choose_action(state)

        else:
            action_B = agent_B.choose_action(env)
        
        # B = action_B

        next_state, rewards, done = env.step(
            (action_A, action_B)
        )

        prices_A.append(action_A)
        prices_B.append(action_B)

        reward_A, reward_B = rewards
        env.last_actions = (
            action_A,
            action_B
        )

        action_rewards_A[action_A].append(reward_A)
        action_rewards_B[action_B].append(reward_B)

        if isinstance(agent_A, QLearningAgent):

            agent_A.update(
                state,
                action_A,
                reward_A,
                next_state
            )

        if isinstance(agent_B, QLearningAgent):

            agent_B.update(
                state,
                action_B,
                reward_B,
                next_state
            )

        if isinstance(agent_A, QLearningAgent):
            agent_A.epsilon *= 0.9999

        if isinstance(agent_B, QLearningAgent):
            agent_B.epsilon *= 0.9999

        print(
            f"State: {state}, "
            f"A: {action_A}, "
            f"B: {action_B}, "
            f"Rewards: {rewards}"
        )


        state = next_state


# print("\nAverage Rewards By Action:\n")


# for action, rewards_list in action_rewards_A.items():

#     avg = (
#         sum(rewards_list)
#         / len(rewards_list)
#     )

#     print(
#         "Agent A",
#         action,
#         "->",
#         round(avg, 2)
#     )


# for action, rewards_list in action_rewards_B.items():

#     avg = (
#         sum(rewards_list)
#         / len(rewards_list)
#     )

#     print(
#         "Agent B",
#         action,
#         "->",
#         round(avg, 2)
#     )

# print("\nLearned Q-Values:\n")


# for state, actions in agent_A.q_table.items():

#     print("State:", state)

#     for action, value in actions.items():

#         print(
#             "  Action:",
#             action,
#             "Q:",
#             round(value, 2)
#         )

#     print()

plt.plot(
    prices_A,
    label="Firm A",
    linewidth=0.5,
    alpha=0.7
)

plt.plot(
    prices_B,
    label="Firm B",
    linewidth=0.5,
    alpha=0.7
)


plt.xlabel("Round")

plt.ylabel("Price")

plt.title("Price Dynamics Over Time")


plt.legend()


# plt.show()



tsum = 0
tnum = 0

for action, rewards_list in action_rewards_A.items():

    avg = 0
    if len(rewards_list):
        avg = (
            sum(rewards_list)
            / len(rewards_list)
        )

    tsum += sum(rewards_list)
    tnum += len(rewards_list)

    print(
        "Agent A",
        action,
        "->",
        round(avg, 2)
    )

print('A:', tsum/tnum)


tsum = 0
tnum = 0

for action, rewards_list in action_rewards_B.items():
    avg = 0
    if len(rewards_list):
        avg = (
            sum(rewards_list)
            / len(rewards_list)
        )

    tsum += sum(rewards_list)
    tnum += len(rewards_list)

    print(
        "Agent B",
        action,
        "->",
        round(avg, 2)
    )

print('B:', tsum/tnum)