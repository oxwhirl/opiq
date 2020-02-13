"""
Taken from:
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/dqn_utils.py
"""

import numpy as np
import random
from functools import partial
import multiprocessing.dummy as mp


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    times_tried = 0
    while len(res) < n and times_tried < 100:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
        else:
            times_tried += 1
    return res


def get_good_idx(dont_sample, ceiling):
    idx = random.randint(0, ceiling)
    while dont_sample[idx]:
        idx = random.randint(0, ceiling)
    return idx


class ReplayBuffer(object):

    def __init__(self, size, frame_history_len, obs_dtype=np.float32, obs_scaling=1.0, args=None):
        """This is a memory efficient implementation of the replay buffer.
        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.
        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes
        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len
        self.args = args

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.intrinsic_reward = None
        self.done = None
        self.dont_sample = None

        self.obs_dtype = obs_dtype
        self.obs_scaling = obs_scaling

        self.bsp = args.bsp
        if self.bsp:
            self.bsp_w = None

        self.mmc = args.mmc
        if self.mmc:
            self.mmc_v = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes, nstep=1):
        obs_batch = np.concatenate(
            [self._encode_observation(idx)[None] for idx in idxes], 0
        )
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        int_rew_batch = self.intrinsic_reward[idxes]

        if nstep == 1:
            next_obs_batch = np.concatenate(
                [self._encode_observation(idx + 1)[None] for idx in idxes], 0
            )
            done_mask = np.array(
                [1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32
            )
        else:
            next_obs_batch = None
            done_mask = None

        bs = len(idxes)
        extra_info = {}
        if nstep > 1:
            steps = np.zeros(shape=(bs, 1), dtype=np.uint8)
            done_nstep = np.ones(shape=(bs, 1), dtype=np.float32)
            reward_nstep = np.zeros(shape=(bs, nstep), dtype=np.float32)
            intrin_reward_nstep = np.zeros(shape=(bs, nstep), dtype=np.float32)

            next_state_shape = (bs,nstep,) + self.state_shape
            next_state_nstep = np.zeros(shape=next_state_shape, dtype=np.float32)
            last_states = np.zeros(shape=(bs,) + self.state_shape, dtype=np.float32)
            next_actions_nstep = np.zeros(shape=(bs, nstep), dtype=np.uint8)
            sarsa_actions = np.zeros(shape=(bs,), dtype=np.uint8)
            if self.bsp:
                bsp_ws = np.zeros(shape=(bs,self.args.bsp_k))
            if self.mmc:
                mmc_vs = np.zeros(shape=(bs,))

            for e, idx in enumerate(idxes):

                steps_to_use = 1

                while idx + steps_to_use <= self.num_in_buffer - 2 and not self.done[idx + steps_to_use - 1] and not self.dont_sample[(idx + steps_to_use) % self.size] and steps_to_use < nstep:
                    steps_to_use += 1

                reward_nstep[e, :steps_to_use] = self.reward[idx:idx + steps_to_use]
                intrin_reward_nstep[e, :steps_to_use] = self.intrinsic_reward[idx:idx + steps_to_use]
                next_actions_nstep[e, :steps_to_use] = self.action[idx: idx+steps_to_use]
                sarsa_actions[e] = self.action[idx + steps_to_use]
                encoded_states = self._encode_observation((idx + 1) % self.size, nstep=steps_to_use)[None]
                for j in range(steps_to_use):
                    next_state_nstep[e, j] = encoded_states[0, j:j+self.args.past_frames_input]
                    # next_state_nstep[e, j - idx] = self._encode_observation((j + 1) % self.size)[None]

                steps[e] = steps_to_use
                done_nstep[e] = self.done[idx + steps_to_use - 1]
                last_states[e] = next_state_nstep[e, steps_to_use - 1]

                if self.bsp:
                    bsp_ws[e] = self.bsp_w[idx]
                if self.mmc:
                    mmc_vs[e] = self.mmc_v[idx]

            extra_info["steps"] = steps
            extra_info["dones"] = done_nstep
            extra_info["rewards"] = reward_nstep
            extra_info["intrin_rewards"] = intrin_reward_nstep
            extra_info["next_states"] = next_state_nstep
            extra_info["last_states"] = last_states
            extra_info["next_actions"] = next_actions_nstep
            extra_info["sarsa_actions"] = sarsa_actions

            if self.bsp:
                extra_info["bsp_w"] = bsp_ws
            if self.mmc:
                extra_info["mmc"] = mmc_vs

        return obs_batch, act_batch, rew_batch, int_rew_batch, next_obs_batch, done_mask, extra_info


    def sample(self, batch_size, nstep=1):
        """Sample `batch_size` different transitions.
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(
            # lambda: random.randint(0, self.num_in_buffer - 2), batch_size
            partial(get_good_idx, self.dont_sample, self.num_in_buffer - 2), batch_size
        )
            # assert all(self.dont_sample[idxes] == False)
        return self._encode_sample(idxes, nstep=nstep)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.
        Returns
        -------
        observation: np.array
            Array of shape (img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8, where observation[i*img_c:(i+1)*img_c, :, :]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx, nstep=1):
        end_idx = idx + nstep  # make noninclusive
        start_idx = end_idx - (nstep - 1) - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2 and nstep==1 and self.frame_history_len==1:
            return self.obs[end_idx - 1].astype(np.float32) * self.obs_scaling
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len + (nstep - 1) - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0).astype(np.float32) * self.obs_scaling  # c, h, w instead of h, w c
        else:
            # this optimization has potential to saves about 30% compute time \o/
            # c, h, w instead of h, w c
            if len(self.obs.shape) == 3:
                return np.concatenate(self.obs[start_idx:end_idx], 0).astype(np.float32) * self.obs_scaling
            else:
                img_h, img_w = self.obs.shape[2], self.obs.shape[3]
                return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w).astype(np.float32) * self.obs_scaling

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        # if observation is an image...
        if len(frame.shape) > 2:
            # transpose image frame into c, h, w instead of h, w, c
            frame = frame.transpose(2, 0, 1)
        elif len(frame.shape) > 1:
            frame = frame.transpose(0, 1)

        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=self.obs_dtype)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.intrinsic_reward = np.empty([self.size], dtype=np.float32)
            self.pseudo_count = np.empty([self.size], dtype=np.int32)
            self.done = np.empty([self.size], dtype=np.bool)
            self.dont_sample = np.empty([self.size], dtype=np.bool)
            if self.bsp:
                self.bsp_w = np.empty([self.size, self.args.bsp_k], dtype=np.int32)
            if self.mmc:
                self.mmc_v = np.empty([self.size], dtype=np.float32)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        # State shape including stacked frames
        self.state_shape = frame.shape
        # Kind of hacky but whatever
        if self.frame_history_len > 1:
            self.state_shape = (self.frame_history_len, *self.state_shape[1:])

        return ret

    def store_effect(self, idx, action, reward, intrinsic_reward, done, pseudo_count, dont_sample=False):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.
        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.intrinsic_reward[idx] = intrinsic_reward
        self.done[idx] = done
        self.dont_sample[idx] = dont_sample
        self.pseudo_count[idx] = pseudo_count

        if self.bsp:
            self.bsp_w[idx] = np.random.binomial(1, self.args.bsp_p, size=self.args.bsp_k)

        if self.mmc:
            self.mmc_v[idx] = 0
            if done:
                # Go back and fill in the MMC
                eps_reward = np.clip(reward, -1, +1) if self.args.reward_clipping else reward
                self.mmc_v[idx] = eps_reward
                end_idx = (idx - 1) % self.num_in_buffer
                while not self.done[end_idx]:
                    reward_to_use = np.clip(self.reward[end_idx], -1, 1) if self.args.reward_clipping else self.reward[end_idx]
                    eps_reward = self.args.gamma * eps_reward + reward_to_use
                    self.mmc_v[end_idx] = eps_reward
                    end_idx -= 1
                    end_idx %= self.num_in_buffer
