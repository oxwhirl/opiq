from collections import defaultdict
import numpy as np
from .HashCount import HashingBonusEvaluator
from utils.logging import get_stats

class RndCount:

    def __init__(self, config):
        self.config = config

        self.num_actions = self.config.num_actions

        self.state_action_counters = [defaultdict(lambda: 0) for _ in range(self.num_actions)]

        self.count_embedding = config.count_size
        matrix_size = (np.prod(self.config.state_shape), self.count_embedding)
        self.rnd_matrix = np.random.normal(0, 1, size=matrix_size)

    def get_count(self, state, action=0, visit=False):
        if not self.config.count_state_action:
            action = 0

        features = np.sign(state.flatten().dot(self.rnd_matrix)).astype(np.int16)
        features_key = tuple(features)
        pseudo_count = self.state_action_counters[action][features_key]

        if visit:
            self.state_action_counters[action][features_key] = pseudo_count + 1
            pseudo_count += 1

        return pseudo_count

    def visit(self, state, action=0):
        return self.get_count(state, action, visit=True)


class ApproxRndCount:

    def __init__(self, config):
        # self.hash_counts = [HashingBonusEvaluator(dim_key=config.count_size, obs_processed_flat_dim=np.prod(config.state_shape)) for _ in range(config.num_actions)]
        self.hash_counts = HashingBonusEvaluator(dim_key=config.count_size, obs_processed_flat_dim=np.prod(config.state_shape), actions=config.num_actions)
        self.hash_state_count = HashingBonusEvaluator(dim_key=config.count_size, obs_processed_flat_dim=np.prod(config.state_shape), actions=1)
        self.config = config

        self.flatten = True if len(config.state_shape) > 1 else False

        self.stats = get_stats()

    def get_count(self, state, action=0, visit=False):
        if not self.config.count_state_action:
            action = 0

        if len(state.shape) >= 3 and state.shape[1] > 1:
            state = state[:, -1]

        # Must do hashing and key computation on the cpu in order to ensure we don't incur errors leading to wildly wrong counts
        # state = state.to("cpu")
        # if len(state.shape) == 1:
        #     state = [state]
        if self.flatten:
            # state = state.flatten()
            # state = np.asarray(state)
            state = state.reshape(state.shape[0], -1)
        if visit:
            self.hash_counts.inc_hash(state, action)
            if self.config.count_state_only_rewards:
                self.hash_state_count.inc_hash(state, 0)

        if self.config.count_state_only_rewards:
            return self.hash_state_count.query_hash(state, 0)
        else:
            return self.hash_counts.query_hash(state, action)

    def get_all_action_counts(self, state):
        if len(state.shape) >= 3 and state.shape[1] > 1:
            state = state[:, -1]
        if self.flatten:
            # state = np.asarray(state)
            state = state.reshape(state.shape[0], -1)
        return self.hash_counts.query_all_actions(state)

    def visit(self, state, action=0):
        visit_counts = self.get_count(state, action, visit=True)
        self.stats.update_stats("Pseudo Counts", visit_counts[0])
        return visit_counts


