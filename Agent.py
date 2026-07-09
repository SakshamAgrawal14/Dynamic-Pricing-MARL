import random


class QLearningAgent:

    def __init__(self, actions):

        # Available pricing actions
        self.actions = actions


        # Learning rate
        self.alpha = 0.1


        # Discount factor
        self.gamma = 0.95


        # Exploration probability
        self.epsilon = 1.0


        # Q-table
        # Stores:
        # Q(state, action)
        self.q_table = {}


    def choose_action(self, state):

        # -------------------------
        # Initialize unseen states
        # -------------------------

        if state not in self.q_table:

            self.q_table[state] = {
                action: 0
                for action in self.actions
            }


        # -------------------------
        # Exploration
        # -------------------------

        if random.random() < self.epsilon:

            return random.choice(self.actions)


        # -------------------------
        # Exploitation
        # -------------------------

        return max(
            self.q_table[state],
            key=self.q_table[state].get
        )


    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        # -------------------------
        # Initialize unseen next state
        # -------------------------

        if next_state not in self.q_table:

            self.q_table[next_state] = {
                action: 0
                for action in self.actions
            }


        # -------------------------
        # Current estimate
        # -------------------------

        old_q = self.q_table[state][action]


        # -------------------------
        # Best future estimate
        # -------------------------

        future_q = max(
                self.q_table[next_state].values()
            )


        # -------------------------
        # Bellman target
        # -------------------------

        target = reward + self.gamma * future_q


        # -------------------------
        # Q-learning update
        # -------------------------

        new_q = (
            old_q
            + self.alpha * (target - old_q)
        )


        # -------------------------
        # Store updated value
        # -------------------------

        self.q_table[state][action] = new_q