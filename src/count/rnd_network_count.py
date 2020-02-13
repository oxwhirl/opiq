from collections import defaultdict
import numpy as np
import torch
from utils.logging import get_stats
from agent.rnd_net import specifier as rnd_specifier

class RndNetworkDistill:

    def __init__(self, config, device):
        self.config = config

        # assert self.config.count_state_action # For now
        num_actions = config.num_actions
        if not self.config.count_state_action:
            num_actions = 1

        self.num_actions = num_actions

        # Maybe the predictors should all start the same and have the same target
        self.predictors = [rnd_specifier.get_pred(config.rnd_net_name)(config).to(device) for _ in range(num_actions)]
        self.targets = [rnd_specifier.get_target(config.rnd_net_name)(config).to(device) for _ in range(num_actions)]

        self.states = [None for _ in range(num_actions)]
        self.states_idx = [0 for _ in range(num_actions)]

        if self.config.rnd_same_starts:
            p_dict = self.predictors[0].state_dict()
            t_dict = self.targets[0].state_dict()
            for p in self.predictors:
                p.load_state_dict(p_dict)
            for t in self.targets:
                t.load_state_dict(t_dict)

        if self.config.count_state_only_rewards:
            self.state_predictor = rnd_specifier.get_pred(config.rnd_net_name)(config).to(device)
            self.state_target = rnd_specifier.get_target(config.rnd_net_name)(config).to(device)
            if self.config.rnd_same_starts:
                self.state_predictor.load_state_dict(p_dict)
                self.state_target.load_state_dict(t_dict)
            self.state_states = None
            self.state_states_idx = 0

        self.stats = get_stats()
        self.reward_directly = True

        # Training stuff
        self.net_parameters = []
        for n in self.predictors:
            self.net_parameters += n.parameters()
        if self.config.count_state_only_rewards:
            self.net_parameters += self.state_predictor.parameters()
        self.optimiser = torch.optim.RMSprop(params=self.net_parameters, lr=self.config.lr)

        self.train_times = self.config.rnd_train_times if self.config.rnd_train_times > 0 else self.config.rnd_batch_size

    def get_count(self, state, action=-1, visit=False):
        if self.config.count_state_only_rewards and action == -1:
            # Lots of code duplication :(
            if len(state.shape) >= 4 and state.shape[1] > 1:
                state = state[:, -1:]

            with torch.no_grad():
                target_x = self.state_target(state)
                pred_x = self.state_predictor(state)

                reward = (target_x - pred_x).pow(2).sum(dim=1) * self.config.rnd_net_scaler

            if visit:
                if np.random.random() < self.config.rnd_train_p:
                    if len(state.shape) >= 4:
                        # Remove the batch dimension
                        state = state[0]
                    if self.state_states is None:
                        self.state_states = torch.zeros((self.config.rnd_batch_size,) + state.shape, dtype=torch.float32, device=state.device)
                    self.state_states[self.state_states_idx] = state
                    self.state_states_idx += 1
                    if self.state_states_idx % self.train_times == 0:
                        self.train(action=-1)
                        if self.state_states_idx >= self.config.rnd_batch_size:
                            self.state_states_idx = 0

            return reward.to("cpu").numpy()

        if not self.config.count_state_action:
            action = 0

        if len(state.shape) >= 4 and state.shape[1] > 1:
            state = state[:, -1:]

        with torch.no_grad():
            target_x = self.targets[action](state)
            pred_x = self.predictors[action](state)

            reward = (target_x - pred_x).pow(2).sum(dim=1) * self.config.rnd_net_scaler

        if visit:
            if np.random.random() < self.config.rnd_train_p:
                if len(state.shape) >= 4:
                    # Remove the batch dimension
                    state = state[0]
                if self.states[action] is None:
                    self.states[action] = torch.zeros((self.config.rnd_batch_size,) + state.shape, dtype=torch.float32, device=state.device)
                self.states[action][self.states_idx[action]] = state
                self.states_idx[action] += 1
                if self.states_idx[action] % self.train_times == 0:
                    self.train(action)
                    if self.states_idx[action] >= self.config.rnd_batch_size:
                        self.states_idx[action] = 0

        return reward.to("cpu").numpy()

    def train(self, action):
        if action == -1:
            states = self.state_states
            prediction = self.state_predictor(states)
            targets = self.state_target(states)
        else:
            states = self.states[action]
            prediction = self.predictors[action](states)
            targets = self.targets[action](states)

        loss = (prediction - targets.detach()).pow(2).mean()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net_parameters, self.config.clip_grad_norm)
        self.optimiser.step()

        self.stats.update_stats("rnd_loss_{}".format(action), loss.to("cpu").item())

    def get_all_action_counts(self, states):
        # Could be sped up
        counts = np.zeros(shape=(self.num_actions, states.shape[0]))
        for a in range(self.num_actions):
            sa_counts = self.get_count(states, a, visit=False)
            counts[a] = sa_counts
        return counts

    def visit(self, state, action=0):
        if self.config.count_state_only_rewards and self.config.count_state_action:
            # Need to update the count for this action
            self.get_count(state, action, visit=True)
            action = -1
        return self.get_count(state, action, visit=True)

