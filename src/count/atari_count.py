from collections import defaultdict
import numpy as np
import torch
from utils.logging import get_stats

class AtariCount:

    def __init__(self, config):
        self.config = config

        self.num_actions = self.config.num_actions

        self.state_action_counters = [defaultdict(lambda: 0) for _ in range(self.num_actions)]

        self.target_shape = (config.atari_target_x, config.atari_target_y)
        self.max_pix_value = 255

        self.stats = get_stats()

        self.hash_vector = None

    def get_count(self, state, action=0, visit=False):
        if not self.config.count_state_action:
            action = 0

        if len(state.shape) >= 4 and state.shape[1] > 1:
            state = state[:, -1]

        abs_state = self.convert_state(state)[0]
        features_key = self.tensor_hash(abs_state).item()
        pseudo_count = self.state_action_counters[action][features_key]

        if visit:
            self.state_action_counters[action][features_key] = pseudo_count + 1
            pseudo_count += 1

        return pseudo_count

    def get_all_action_counts(self, states):
        counts = np.zeros(shape=(self.num_actions, states.shape[0]))
        if len(states.shape) >= 4 and states.shape[1] > 1:
            states = states[:, -1]
        # Get features for all states
        abs_states = self.convert_state(states)
        hashed_states = self.tensor_hash(abs_states).tolist()
        for si, s in enumerate(hashed_states):
            for a in range(self.num_actions):
                # features_key = s.item()
                features_key = s
                pseudo_count = self.state_action_counters[a][features_key]
                counts[a, si] = pseudo_count
        return counts

    def visit(self, state, action=0):
        visit_count = self.get_count(state, action, visit=True)
        self.stats.update_stats("Pseudo Counts", visit_count)
        return visit_count

    def convert_state(self, state):
        # State is already grayscale
        # Do this on the cpu if things get weird
        resized = torch.nn.functional.interpolate(state.detach().unsqueeze(1), size=self.target_shape, mode="area")[:, 0]
        intensity = (resized * 255 / 32).int()
        return intensity

    def tensor_hash(self, x):
        # Hash function to use for content-based hashing of tensors
        if self.hash_vector is None:
            self.hash_vector = (torch.randint_like(x, 57) * 317).int()

        hashes = None
        if len(x.size()) < 3:
            hashes = (self.hash_vector * x).sum()
        else:
            hashes = (self.hash_vector * x).sum(dim=1).sum(dim=1)

        return hashes.cpu()

        # Too slow
        # tupled = tuple(x.flatten().numpy())
        # return hash(tupled)

