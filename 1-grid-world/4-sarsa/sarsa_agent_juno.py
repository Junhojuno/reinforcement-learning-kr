import numpy as np
import random
from collections import defaultdict
from environment import Env


class SARSAagent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda : [0.0,0.0,0.0,0.0])

    # q update!!
    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        new_q = current_q + self.learning_rate*(reward + next_q - current_q)

        self.q_table[state][action] = new_q

    # epsilon greedy
    def get_action(self, state):
        if np.random.randn() < self.epsilon:
            action = np.random.choice(self.actions)

        else:
            state_action = self.q_table[state]
            action = np.argmax(state_action)

        return action

if __name__== "__main__":
    env = Env()
    agent = SARSAagent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        action = agent.get_action(str(state))

        while True:
            env.render()

            next_state, reward, done = env.step(action)
            next_action = agent.get_action(str(next_state))

            agent.learn(str(state), action, reward, str(next_state), next_action)

            state = next_state
            action = next_action

            env.print_value_all(agent.q_table)

            if done:
                break



