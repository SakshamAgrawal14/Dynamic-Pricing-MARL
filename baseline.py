import random


class RandomAgent:

    def __init__(self, actions):
        self.actions = actions

    def choose_action(self, env):
        return random.choice(self.actions)


class Always50Agent:

    def choose_action(self, env):
        return 50


class Always70Agent:

    def choose_action(self, env):
        return 70


class TitForTatAgent:

    def choose_action(self, env):

        return env.last_actions[0]