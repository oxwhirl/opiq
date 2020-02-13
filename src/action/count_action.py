import numpy as np
from utils.logging import get_stats
import logging


class CountBonus:

    def __init__(self, count_model, args):

        self.epsilon_start = args.epsilon_start
        self.epsilon_finish = args.epsilon_finish
        self.epsilon_time_length = args.epsilon_time_length

        self.num_actions = args.num_actions
        self.stats = get_stats()
        self.logger = logging.getLogger("CountBonus")

        self.min_q = 1000
        self.max_q = -1000

        self.count_model = count_model
        self.args = args

    def select_actions(self, q_values, t, info):

        state = info["state"]

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

        info = {}
        info["Q_Values"] = q_values[0].detach().numpy()
        self.max_q = max(self.max_q * 0.99999999, info["Q_Values"].max())
        self.min_q = min(self.min_q * 0.99999999, info["Q_Values"].min())
        info["Max_Q_Value"] = self.max_q
        info["Min_Q_Value"] = self.min_q

        if np.random.random() < epsilon:
            # Random action
            action = np.random.randint(self.num_actions)
            self.logger.debug("Random action selected")
        else:
            action_bonuses = []
            # Argmax over Q-Values with the count bonus added
            for a in range(self.args.num_actions):
                _, action_info = self.count_model.bonus(state, a, dont_remember=True)
                pseudo_count = action_info["Pseudo_Count"]

                action_bonus = self.args.action_bonus_scaler / np.sqrt(
                    pseudo_count + 0.01
                )
                q_values[0, a] += action_bonus
                action_bonuses.append(action_bonus)

            action = q_values.argmax().cpu().item()
            self.logger.debug("Argmax action over Q-Values selected")
            info["Action_Bonus"] = action_bonuses

        info["Action"] = action
        info["Epsilon"] = epsilon

        return action, info

    def log_stats(self, epsilon):
        self.stats.update_stats("Epsilon", epsilon)
