import torch
import logging
from utils.logging import get_stats
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")


class DQNTrainer:

    def __init__(self, agent, target_agent, args, count_model=None, buffer=None):
        self.args = args
        self.agent = agent
        self.target_agent = target_agent

        # self.parameters = self.agent.parameters()
        self.agent_parameters = list(self.agent.parameters())
        if args.atari_rms:
            self.optimiser = torch.optim.RMSprop(params=self.agent_parameters, lr=self.args.lr, alpha=0.95, eps=0.00001, centered=True)
        else:
            self.optimiser = torch.optim.RMSprop(params=self.agent_parameters, lr=self.args.lr)

        self.logger = logging.getLogger("DQNTrainer")
        self.stats = get_stats()

        self.count_model = count_model

        self.nstep = args.n_step > 1
        self.goal_samples = 0

        self.buffer = buffer

    def train(self, batch):
        if self.nstep:
            # Bad coding with a lot of duplicated code.
            return self.train_nstep(batch)

        states, actions, rewards, intrinsic_rewards, next_states, terminations, extra_info = batch

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        intrinsic_rewards = torch.from_numpy(intrinsic_rewards)
        next_states = torch.from_numpy(next_states)
        terminations = torch.from_numpy(terminations)

        # Clip Rewards to [-1, 1]
        if self.args.reward_clipping:
            rewards = rewards.clamp(min=-1, max=+1)

        # Move them over
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        intrinsic_rewards = intrinsic_rewards.to(device)
        next_states = next_states.to(device)
        terminations = terminations.to(device)

        # Change dtypes
        states = states.type(torch.float32)
        next_states = next_states.type(torch.float32)
        actions = actions.type(torch.long)

        states.requires_grad = True
        if self.args.double_q:
            state_next_state_q_vals = self.agent(torch.cat([states, next_states], dim=0))
            num_states = states.shape[0]
            q_values = state_next_state_q_vals[:num_states]
            q_values_next_states = state_next_state_q_vals[num_states:]
        else:
            q_values = self.agent(states)
        target_agent_q_values = self.target_agent(next_states)

        if self.args.optim_bootstrap:
            assert self.args.double_q is False
            # Get the counts for the next states
            state_action_counts = self.count_model.get_all_action_counts(next_states.detach())
            counts_tensor = torch.tensor(state_action_counts.transpose(1,0), device=next_states.device, dtype=torch.float32)
            optims = self.args.optim_bootstrap_tau / (counts_tensor + 1.0).pow(self.args.optim_m)
            target_agent_q_values = target_agent_q_values + optims
            if self.args.double_q:
                q_values_next_states = q_values_next_states + optims

        taken_q_value = q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        if self.args.double_q:
            max_next_state_actions = q_values_next_states.max(dim=1)[1]
            max_target_q_values = target_agent_q_values.gather(dim=1, index=max_next_state_actions.unsqueeze(1)).squeeze(1)
        else:
            max_target_q_values = target_agent_q_values.max(dim=1)[0]

        if self.args.recompute_count_rewards and self.args.count_rewards:
            if self.args.count_state_only_rewards:
                # Counts for just the state
                taken_action_counts = torch.tensor(self.count_model.get_count(states.detach()), device=next_states.device, dtype=torch.float32)
            else:
                # Get the counts for the current states for all actions
                state_action_counts = self.count_model.get_all_action_counts(states.detach())
                counts_tensor = torch.tensor(state_action_counts.transpose(1,0), device=next_states.device, dtype=torch.float32)
                taken_action_counts = counts_tensor.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
                zero_counts = (taken_action_counts < 1).sum().item()
                self.stats.update_stats("0_Counts", zero_counts)
            if getattr(self.count_model, "reward_directly", False):
                taken_action_intrinsic_rewards = taken_action_counts
            else:
                taken_action_counts = taken_action_counts.clamp_(min=1, max=10000000)
                taken_action_intrinsic_rewards = self.args.count_beta / taken_action_counts.pow(0.5)
            self.stats.update_stats("intrinsic rewards", intrinsic_rewards.mean().to(cpu_device).item())
            self.stats.update_stats("intrinsic rewards recalc", taken_action_intrinsic_rewards.mean().to(cpu_device).item())
            self.stats.update_stats("intrinsic rewards diff", (intrinsic_rewards - taken_action_intrinsic_rewards).mean().to(cpu_device).item())
            intrinsic_rewards = torch.min(intrinsic_rewards, taken_action_intrinsic_rewards) # To ensure

        if self.args.reward_clipping:
            intrinsic_rewards = intrinsic_rewards.clamp(min=-1, max=+1)

        # 1-step Q-Learning target
        targets = rewards + intrinsic_rewards + self.args.gamma * max_target_q_values * (1 - terminations)

        td_error = taken_q_value - targets.detach()

        # Loss is td-error^2 for each sample. Take the mean over the batch
        loss = td_error.pow(2)

        loss = loss.mean()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent_parameters, self.args.clip_grad_norm)
        self.optimiser.step()

        # Move stuff back to cpu for logging
        loss = loss.to(cpu_device)

        self.logger.debug("Loss: {:.4f}, GradNorm: {:.4f}".format(loss, grad_norm))
        # self.logger.debug("Q-Values: {}".format(q_values))

        self.stats.update_stats("td_error", td_error.mean().to(cpu_device).item())
        self.stats.update_stats("Loss", loss.item())
        self.stats.update_stats("GradNorm", grad_norm)
        self.stats.update_stats("targets", targets.mean().to(cpu_device).item())

    def train_nstep(self, batch):
        states, actions, rewards, intrinsic_rewards, next_states, terminations, extra_info = batch

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        n_rewards = torch.from_numpy(extra_info["rewards"])
        n_intrinsic_rewards = torch.from_numpy(extra_info["intrin_rewards"])
        n_next_states = torch.from_numpy(extra_info["next_states"])
        n_last_states = torch.from_numpy(extra_info["last_states"])
        terminations = torch.from_numpy(extra_info["dones"])
        steps = torch.from_numpy(extra_info["steps"])
        next_actions = torch.from_numpy(extra_info["next_actions"])

        # Clip Rewards to [-1, 1]
        # n_rewards = n_rewards.clamp(min=-1, max=+1)

        # Move them over
        states = states.to(device)
        actions = actions.to(device)
        n_rewards = n_rewards.to(device)
        n_intrinsic_rewards = n_intrinsic_rewards.to(device)
        n_next_states = n_next_states.to(device)
        terminations = terminations.to(device)
        steps = steps.to(device)
        n_last_states = n_last_states.to(device)
        next_actions = next_actions.to(device)

        # Change dtypes
        states = states.type(torch.float32)
        n_next_states = n_next_states.type(torch.float32)
        n_last_states = n_last_states.type(torch.float32)
        actions = actions.type(torch.long)
        next_actions = next_actions.type(torch.long)

        terminations = terminations[:, 0]
        steps = steps[:, 0]

        if self.args.bsp:
            bsp_w = torch.from_numpy(extra_info["bsp_w"]).to(device).type(torch.long)

        if self.args.mmc:
            mmc_v = torch.from_numpy(extra_info["mmc"]).to(device).type(torch.float)

        states.requires_grad = True
        # assert not self.args.double_q
        if self.args.double_q:
            state_next_state_q_vals = self.agent(torch.cat([states, n_last_states], dim=0))
            num_states = states.shape[0]
            q_values = state_next_state_q_vals[:num_states]
            q_values_next_states = state_next_state_q_vals[num_states:]
        else:
            q_values = self.agent(states)
        # next_states_to_use = n_next_states.index_select(dim=1, index=(steps - 1).long())
        target_agent_q_values = self.target_agent(n_last_states)

        if self.args.optim_bootstrap:
            # assert self.args.double_q is False
            # Get the counts for the next states
            state_action_counts = self.count_model.get_all_action_counts(n_last_states.detach())
            counts_tensor = torch.tensor(state_action_counts.transpose(1,0), device=n_next_states.device, dtype=torch.float32)
            if getattr(self.count_model, "reward_directly", False):
                optims = counts_tensor
                # Hyperparameter scaling
                optims = optims / self.args.rnd_net_scaler
                optims = optims ** self.args.rnd_optim_m
                optims = optims * self.args.optim_bootstrap_tau
            else:
                optims = self.args.optim_bootstrap_tau / (counts_tensor + 1.0).pow(self.args.optim_m)
            target_agent_q_values = target_agent_q_values + optims
            if self.args.optim_interpolation:
                w = 1 / (counts_tensor + 1.0).pow(self.args.optim_m)
                target_agent_q_values = target_agent_q_values * (1 - w) + self.args.optim_bootstrap_tau * (w)
            if self.args.double_q:
                q_values_next_states = q_values_next_states + optims

        if self.args.bsp:
            actions = actions
            # Pytorch weirdness?
            bs = states.shape[0]
            taken_q_value = q_values[torch.arange(bs), actions, :]
        else:
            taken_q_value = q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        if self.args.double_q:
            max_next_state_actions = q_values_next_states.max(dim=1)[1]
            max_target_q_values = target_agent_q_values.gather(dim=1, index=max_next_state_actions.unsqueeze(1)).squeeze(1)
        else:
            max_target_q_values = target_agent_q_values.max(dim=1)[0]

        gamma_tensor = torch.tensor([self.args.gamma ** i for i in range(self.args.n_step)], device=n_next_states.device)
        batch_size = states.shape[0]
        gamma_tensor = gamma_tensor.repeat(batch_size, 1)
        # Slow
        for idx, step in enumerate(steps):
            gamma_tensor[idx, step:] = 0
        if self.args.recompute_count_rewards and self.args.count_rewards:
            # assert self.args.count_state_only_rewards
            batch_size = states.shape[0]
            shape_to_use = (batch_size * (self.args.n_step),) + n_next_states.shape[2:]
            states_to_count = torch.cat([states.unsqueeze(1), n_next_states[:, :-1]], dim=1)
            states_to_count_flattened = states_to_count.reshape(shape_to_use)
            # Counts for just the state
            if self.args.count_state_only_rewards:
                taken_action_counts = torch.tensor(self.count_model.get_count(states_to_count_flattened.detach(), action=-1), device=n_next_states.device, dtype=torch.float32)
            else:
                # Get the counts for the current states for all actions
                state_action_counts = self.count_model.get_all_action_counts(states_to_count_flattened.detach())
                counts_tensor = torch.tensor(state_action_counts.transpose(1, 0), device=n_next_states.device, dtype=torch.float32)
                nn = next_actions.reshape(counts_tensor.shape[0], 1)
                taken_action_counts = counts_tensor.gather(dim=1, index=nn)
                zero_counts = (taken_action_counts < 1).sum().item()
                self.stats.update_stats("0_Counts", zero_counts)
            #
            # # Slow count to ensure it works correctly on the gpu
            # scs = torch.ones(size=(batch_size, self.args.n_step), device=n_next_states.device, dtype=torch.float32)
            # for b_idx in range(batch_size):
            #     rc = self.count_model.get_count(states[b_idx].detach())[0]
            #     scs[b_idx, 0] = rc
            #     for ss in range(steps[b_idx] - 1):
            #         rc_n = self.count_model.get_count(n_next_states[b_idx, ss].detach())[0]
            #         scs[b_idx, ss + 1] = rc_n
            # taken_action_counts = scs

            if getattr(self.count_model, "reward_directly", False):
                taken_action_intrinsic_rewards = taken_action_counts
            else:
                taken_action_counts = taken_action_counts.clamp_(min=1, max=10000000)
                taken_action_intrinsic_rewards = self.args.count_beta / taken_action_counts.pow(0.5)
            self.stats.update_stats("intrinsic rewards", n_intrinsic_rewards.mean().to(cpu_device).item())
            intrinsic_rewards = taken_action_intrinsic_rewards
            self.stats.update_stats("intrin rewards recalc", intrinsic_rewards.mean().to(cpu_device).item())
            reshaped_i_rewards = intrinsic_rewards.view(batch_size, self.args.n_step)
            # reshaped_i_rewards = taken_action_intrinsic_rewards
            discounted_diff_rewards = (n_intrinsic_rewards - reshaped_i_rewards) * gamma_tensor
            self.stats.update_stats("intrin rewards diff", discounted_diff_rewards.mean().to(cpu_device).item())
            reshaped_i_rewards = torch.min(reshaped_i_rewards, n_intrinsic_rewards) # To ensure our recomputed rewards aren't > originals. Due to errors introduced into the keys when batching
            if self.args.reward_clipping:
                reshaped_i_rewards = reshaped_i_rewards.clamp(min=-1, max=+1)
            discounted_i_rewards = reshaped_i_rewards * gamma_tensor
            discounted_sum_i_rewards = discounted_i_rewards.sum(dim=1)
        else:
            if self.args.reward_clipping:
                n_intrinsic_rewards = n_intrinsic_rewards.clamp(min=-1, max=+1)
            discounted_sum_i_rewards = n_intrinsic_rewards * gamma_tensor
            discounted_sum_i_rewards = discounted_sum_i_rewards.sum(dim=1)

        if self.args.reward_clipping:
            n_rewards = n_rewards.clamp(min=-1, max=+1)
        discounted_rewards = n_rewards * gamma_tensor
        discounted_sum_rewards = discounted_rewards.sum(dim=1)

        # n-step Q-Learning target
        # targets = rewards + intrinsic_rewards + self.args.gamma * max_target_q_values * (1 - terminations)
        if self.args.bsp:
            targets = (discounted_sum_rewards + discounted_sum_i_rewards).unsqueeze(1) + ((self.args.gamma ** steps.float()) * (1 - terminations)).unsqueeze(1) * max_target_q_values
            targets = targets * bsp_w.float()
            if self.args.mmc:
                targets = (1 - self.args.mmc_beta) * targets + (self.args.mmc_beta) * mmc_v.unsqueeze(1).expand_as(targets) * bsp_w.float()
        else:
            targets = discounted_sum_rewards + discounted_sum_i_rewards + (self.args.gamma ** steps.float()) * (1 - terminations) * max_target_q_values

            if self.args.mmc:
                targets = (1 - self.args.mmc_beta) * targets + (self.args.mmc_beta) * mmc_v

        if any(discounted_sum_rewards > 0.9):
            self.goal_samples += 1
        self.stats.update_stats("goal_samples", self.goal_samples)

        td_error = taken_q_value - targets.detach()

        # Loss is td-error^2 for each sample. Take the mean over the batch
        loss = td_error.pow(2)

        loss = loss.mean()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent_parameters, self.args.clip_grad_norm)
        if self.args.bsp and self.args.bsp_grad_norm:
            self.agent.scale_gradients()
        self.optimiser.step()

        # Move stuff back to cpu for logging
        loss = loss.to(cpu_device)

        self.logger.debug("Loss: {:.4f}, GradNorm: {:.4f}".format(loss, grad_norm))
        # self.logger.debug("Q-Values: {}".format(q_values))

        self.stats.update_stats("td_error", td_error.mean().to(cpu_device).item())
        self.stats.update_stats("targets", targets.mean().to(cpu_device).item())
        self.stats.update_stats("Loss", loss.item())
        self.stats.update_stats("GradNorm", grad_norm)

    def update_target_agent(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
