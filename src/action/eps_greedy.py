import numpy as np
from utils.logging import get_stats
import logging


class EpsGreedy:

    def __init__(self, args):

        self.epsilon_start = args.epsilon_start
        self.epsilon_finish = args.epsilon_finish
        self.epsilon_time_length = args.epsilon_time_length

        self.num_actions = args.num_actions
        self.stats = get_stats()
        self.logger = logging.getLogger("EpsGreedy")

    def select_actions(self, q_values, t, info={}):

        epsilon = max(
            self.epsilon_finish,
            self.epsilon_start
            - (t / self.epsilon_time_length)
            * (self.epsilon_start - self.epsilon_finish),
        )
        self.logger.debug("Epsilon: {:.2f}".format(epsilon))
        if t % 1000 == 0:
            self.logger.info("Epsilon: {:.2f}".format(epsilon))
        self.log_stats(epsilon)

        if np.random.random() < epsilon:
            # Random action
            action = np.random.randint(self.num_actions)
            self.logger.debug("Random action selected")
        else:
            # Argmax over Q-Values
            action = q_values.argmax().cpu().item()
            self.logger.debug("Argmax action over Q-Values selected")

        return action, {}

    def log_stats(self, epsilon):
        self.stats.update_stats("Epsilon", epsilon)
