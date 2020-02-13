# Taken from https://github.com/openai/EPG/blob/master/epg/exploration.py
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HashingBonusEvaluator(object):
    """Hash-based count bonus for exploration.

    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, X., Duan, Y., Schulman, J., De Turck, F., and Abbeel, P. (2017).
    #Exploration: A study of count-based exploration for deep reinforcement learning.
    In Advances in Neural Information Processing Systems (NIPS)
    """

    def __init__(self, dim_key=128, obs_processed_flat_dim=None, bucket_sizes=None, actions=1):
        # Hashing function: SimHash
        if bucket_sizes is None:
            # Large prime numbers
            # bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
            # Smaller prime numbers to (hopefully) have less precision errors when batching
            bucket_sizes = [911, 919, 929, 937, 941, 947]
        mods_list = []
        for bucket_size in bucket_sizes:
            mod = 1
            mods = []
            for _ in range(dim_key):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(bucket_sizes)
        # self.mods_list = np.asarray(mods_list).T
        self.mods_list = torch.tensor(mods_list).transpose(1,0).to(device).float()
        self.tables = np.zeros((actions, len(bucket_sizes), np.max(bucket_sizes)))

        # self.projection_matrix = np.random.normal(size=(obs_processed_flat_dim, dim_key))
        self.projection_matrix = torch.normal(mean=torch.zeros(size=(obs_processed_flat_dim, dim_key)),
                                              std=torch.ones(size=(1, obs_processed_flat_dim, dim_key))).to(device)

    def project(self, obss):
        return torch.sign(obss @ self.projection_matrix).float()

    def compute_keys(self, obss):
        binaries = torch.sign(obss @ self.projection_matrix).float()
        # binaries = np.sign(np.asarray(obss).dot(self.projection_matrix))
        keys = np.cast['int']((binaries @ self.mods_list).cpu()) % self.bucket_sizes
        # keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_hash(self, obss, action):
        keys = self.compute_keys(obss)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[action, idx], keys[:, idx], 1)

    def query_hash(self, obss, action):
        keys = self.compute_keys(obss)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[action, idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    def query_all_actions(self, obss):
        keys = self.compute_keys(obss)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[:, idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    # def fit_before_process_samples(self, obs):
    #     if len(obs.shape) == 1:
    #         obss = [obs]
    #     else:
    #         obss = obs
    #     before_counts = self.query_hash(obss)
    #     self.inc_hash(obss)
    #
    # def predict(self, obs):
    #     counts = self.query_hash(obs)
    #     return 1. / np.maximum(1., np.sqrt(counts))