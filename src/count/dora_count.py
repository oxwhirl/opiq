from collections import defaultdict
import numpy as np
import torch
from utils.logging import get_stats
from agent.dora.specifier import get_net as dora_specifier
from math import sqrt

class DoraCount:

    def __init__(self, config, device):
        self.config = config
        self.args = config # V. Lazy coding

        num_actions = config.num_actions
        if not self.config.count_state_action:
            raise Exception

        self.num_actions = num_actions

        self.net = dora_specifier(config.dora_name)(config).to(device)
        self.target_net = dora_specifier(config.dora_name)(config).to(device)
        self.target_net.load_state_dict(self.net.state_dict())

        self.stats = get_stats()
        self.reward_directly = True

        self.train_times = config.batch_size
        self.states = None
        self.states_idx = 0

        self.device = device
        self.agent_parameters = self.net.parameters()
        self.optimiser = torch.optim.RMSprop(params=self.agent_parameters, lr=self.config.lr)

    def update_target_agent(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def get_count(self, state, action=-1, visit=False):
        with torch.no_grad():
            e_values = self.net(state)
            e_value = e_values[:,action]
            reward = self.config.dora_beta / torch.sqrt(-torch.log(e_value)).to("cpu")
        return reward.numpy()

    def train(self, batch):
        states, actions, rewards, intrinsic_rewards, next_states, terminations, extra_info = batch

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        n_last_states = torch.from_numpy(extra_info["last_states"])
        terminations = torch.from_numpy(extra_info["dones"])
        steps = torch.from_numpy(extra_info["steps"])
        next_actions = torch.from_numpy(extra_info["sarsa_actions"])

        device = self.device

        # Move them over
        states = states.to(device)
        actions = actions.to(device)
        terminations = terminations.to(device)
        steps = steps.to(device)
        n_last_states = n_last_states.to(device)
        next_actions = next_actions.to(device)

        # Change dtypes
        states = states.type(torch.float32)
        n_last_states = n_last_states.type(torch.float32)
        actions = actions.type(torch.long)
        next_actions = next_actions.type(torch.long)

        terminations = terminations[:, 0]
        steps = steps[:, 0]

        states.requires_grad = True
        q_values = self.net(states)
        target_agent_q_values = self.target_net(n_last_states)

        taken_q_value = q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        taken_target_q_value = target_agent_q_values.gather(dim=1, index=next_actions.unsqueeze(1)).squeeze(1)

        # n-step Q-Learning target
        targets = (self.args.gamma ** steps.float()) * (1 - terminations) * taken_target_q_value

        td_error = taken_q_value - targets.detach()

        loss = td_error.pow(2).mean()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent_parameters, self.args.clip_grad_norm)
        self.optimiser.step()

        # Move stuff back to cpu for logging
        loss = loss.to("cpu")

        self.stats.update_stats("Dora_Loss", loss.item())

    def get_all_action_counts(self, states):
        e_values = self.net(states)
        final_values = self.config.dora_beta / torch.sqrt(torch.log(e_values).clamp_(0.00001, 0.99999))
        return final_values.transpose(0,1)

    def visit(self, state, action=0):
        return self.get_count(state, action, visit=True)

