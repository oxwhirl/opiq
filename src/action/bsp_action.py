import numpy as np
from utils.logging import get_stats
import logging
import torch


class BSPAction:

    def __init__(self, args):

        self.epsilon_start = args.epsilon_start
        self.epsilon_finish = args.epsilon_finish
        self.epsilon_time_length = args.epsilon_time_length

        self.num_actions = args.num_actions
        self.stats = get_stats()
        self.logger = logging.getLogger("BSPAction")

        self.config = args

        self.current_k = 0

    def update_k(self, k):
        self.current_k = k

    def select_actions(self, q_values, t, info={}, testing=False):

        state = info["state"]

        epsilon = max(
            self.epsilon_finish,
            self.epsilon_start
            - (t / self.epsilon_time_length)
            * (self.epsilon_start - self.epsilon_finish),
            )
        if not testing:
            self.logger.debug("Epsilon: {:.2f}".format(epsilon))
            if t % 1000 == 0:
                self.logger.info("Epsilon: {:.2f}".format(epsilon))

        if not testing and np.random.random() < epsilon:
            # Random action
            action = np.random.randint(self.num_actions)
            self.logger.debug("Random action selected")
        elif testing:
            # Majority vote
            action_votes = np.array([0 for _ in range(self.num_actions)])
            for k in range(self.config.bsp_k):
                argmax_action = np.argmax(q_values.detach()[0,:,k].cpu().numpy())
                action_votes[argmax_action] += 1
            # Random tie breaking
            action = np.random.choice(np.flatnonzero(action_votes == action_votes.max()))
        else:
            q_vals = q_values[:,:,self.current_k].detach()[0].cpu().numpy()
            action = np.argmax(q_vals)
            if not testing:
                self.logger.debug("Argmax action over Q-Values selected")
                if self.num_actions < 20:
                    for a in range(self.num_actions):
                        self.stats.update_stats("BSP_Q_Value_{}".format(a), q_vals[a])
                else:
                    self.stats.update_stats("BSP_Mean_Q_Value", np.mean(q_vals))

        self.stats.update_stats("Epsilon", epsilon)

        return action, {}
