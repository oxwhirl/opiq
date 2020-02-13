import numpy as np
from utils.logging import get_stats
import logging
import torch


class OptimisticAction:

    def __init__(self, count_model, args):

        self.epsilon_start = args.epsilon_start
        self.epsilon_finish = args.epsilon_finish
        self.epsilon_time_length = args.epsilon_time_length

        self.num_actions = args.num_actions
        self.stats = get_stats()
        self.logger = logging.getLogger("OptimisticAction")

        self.count_model = count_model
        self.m = args.optim_m
        self.tau = args.optim_action_tau

        self.config = args

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
        else:
            q_vals_copy = (q_values + 0).detach()[0].cpu().numpy()
            state_action_counts = self.count_model.get_all_action_counts(state)[:, 0]
            if getattr(self.count_model, "reward_directly", False):
                if self.config.dora_count:
                    optims = state_action_counts
                    optims = optims * self.config.dora_action / self.config.dora_beta
                    optims = optims.detach().cpu().numpy()
                elif self.config.rnd_net_count:
                    optims = state_action_counts
                    optims = optims * self.tau / self.config.rnd_net_scaler
            else:
                optims = self.tau / ((state_action_counts + 1.0) ** self.m)
            # optims = np.concatenate([self.tau / np.power(self.count_model.get_count(state, a) + 1, self.m) for a in range(self.num_actions)])
            # np_optims = np.array(optims)

            optim_q_vals = q_vals_copy + optims
            if self.config.optim_interpolation:
                w = 1 / ((state_action_counts + 1.0) ** self.m)
                optim_q_vals = q_vals_copy * (1 - w) + self.tau * (w)
            action = np.argmax(optim_q_vals)
            if not testing:
                self.logger.debug("Argmax action over Q-Values selected")
                if self.num_actions < 20:
                    for a in range(self.num_actions):
                        self.stats.update_stats("Optim_Q_Value_{}".format(a), optim_q_vals[a])
                else:
                    self.stats.update_stats("Optim_Mean_Q_Value", np.mean(optim_q_vals))

        self.stats.update_stats("Epsilon", epsilon)

        return action, {}
