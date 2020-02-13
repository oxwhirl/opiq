import gym
from .OpenAI_AtariWrapper import make_atari
from utils.logging import get_stats
import numpy as np


class AtariEnv(gym.Env):

    # Our wrapper handles the drawing
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, game):
        self.mont_env = gym.make("{}NoFrameskip-v4".format(game))
        # self.mario_env = modewrapper(self.mario_env)
        self.max_timesteps = 4500
        self.mont_env = make_atari(self.mont_env, max_episode_steps=self.max_timesteps)

        self.steps = 0

        self._seed = 56
        self.observation_space = self.mont_env.observation_space
        self.action_space = self.mont_env.action_space

        self.stats = get_stats()

        self.obs_dtype = np.uint8
        self.obs_scaling = 1/255.0

    def _step(self, a):
        self.steps += 1
        s, r, finished, info = self.mont_env.step(a)
        if self.steps >= self.max_timesteps and not finished:
            finished = True
            info["Steps_Termination"] = True
        return s, r, finished, info

    def _reset(self):
        self.steps = 0
        return self.mont_env.reset()

    def _render(self, mode="rgb_array", close=False):
        return self.mont_env.render(mode=mode)

    def _close(self):
        self.mont_env.close()

    def log_player_pos(self):
        return 0

    def trained_on_states(self, player_visits, args):
        pass

    def xp_and_frontier_states(self):
        pass

    def bonus_xp_and_frontier_states(self):
        pass

    def visits_and_frontier_states(self):
        pass

    def xp_replay_states(self, player_visits, args, bonus_replay=False):
        pass

    def player_visits(self, player_visits, args):
        pass

    def bonus_landscape(self, player_visits, exploration_bonuses, max_bonus, args):
        pass

    def frontier(self, exp_model, args, max_bonus=None):
        pass