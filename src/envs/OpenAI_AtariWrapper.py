import numpy as np
from collections import deque
from PIL import Image
import gym
from gym import spaces
import sys
import time
import cv2
from copy import copy

cv2.ocl.setUseOpenCL(False)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4, max_over=2):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=max_over)
        self._skip = skip

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class ClipNegativeRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        if reward >= 0:
            return reward
        else:
            return 0


class NoRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        return 0


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, res=84):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = res
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
                                                       resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1)) / 255.0


class ResizeFrame(gym.ObservationWrapper):
    def __init__(self, env, res=40):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = res
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = obs.astype("float32") * 255
        frame = frame[:, :, 0]
        # frame = np.concatenate([frame, frame, frame], axis=2)
        # frame = np.dot(frame, np.array([0.299, 0.587, 0.114], 'float32'))

        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
                                                       resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1)) / 255.0


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class GreyscaleRender(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.latest_obs = None

    def _step(self, action):
        stuff = self.env.step(action)
        self.latest_obs = stuff[0]
        return stuff

    def render(self, mode="human", close=False):
        if mode == "human":
            self.unwrapped._render(mode=mode, close=close)
        else:
            # print(self.unwrapped)
            # grid = self.env._observation()
            grid = self.latest_obs
            grid = grid[:, :, -1]
            grid = np.stack([grid for _ in range(3)], axis=2)
            return grid * 255


def wrap_maze(env):
    # Change the size of the maze to be (40, 40)
    env = ResizeFrame(env, res=40)
    env = FrameStack(env, 1)
    env = GreyscaleRender(env)
    print("Wrapping maze to be (40, 40)")
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, stack=4):
    """Configure environment for DeepMind-style Atari.

    Note: this does not include frame stacking!"""
    assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    if episode_life:
        env = EpisodicLifeEnv(env)
    # env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, res=42)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if stack > 1:
        env = FrameStack(env, 4)
    print("Wrapping environment with Deepmind-style setttings but 42 x 42.")
    env = GreyscaleRender(env)
    return env


resolutions = ['160x120', '200x125', '200x150', '256x144', '256x160', '256x192', '320x180', '320x200',
               '320x240', '320x256', '400x225', '400x250', '400x300', '512x288', '512x320', '512x384',
               '640x360', '640x400', '640x480', '800x450', '800x500', '800x600', '1024x576', '1024x640',
               '1024x768', '1280x720', '1280x800', '1280x960', '1280x1024', '1400x787', '1400x875',
               '1400x1050', '1600x900', '1600x1000', '1600x1200', '1920x1080']

__all__ = ['SetResolution']


def SetResolution(target_resolution):
    class SetResolutionWrapper(gym.Wrapper):
        """
            Doom wrapper to change screen resolution
        """

        def __init__(self, env):
            super(SetResolutionWrapper, self).__init__(env)
            if target_resolution not in resolutions:
                raise gym.error.Error(
                    'Error - The specified resolution "{}" is not supported by Vizdoom.'.format(target_resolution))
            parts = target_resolution.lower().split('x')
            width = int(parts[0])
            height = int(parts[1])
            screen_res = target_resolution
            self.screen_width, self.screen_height, self.unwrapped.screen_resolution = width, height, screen_res
            self.unwrapped.observation_space = gym.spaces.Box(low=0, high=255,
                                                              shape=(self.screen_height, self.screen_width, 3))
            self.observation_space = self.unwrapped.observation_space

    return SetResolutionWrapper


# Adapters
from gym.spaces import Discrete, MultiDiscrete


class DiscreteToMultiDiscrete(Discrete):
    """
    Adapter that adapts the MultiDiscrete action space to a Discrete action space of any size
    The converted action can be retrieved by calling the adapter with the discrete action
        discrete_to_multi_discrete = DiscreteToMultiDiscrete(multi_discrete)
        discrete_action = discrete_to_multi_discrete.sample()
        multi_discrete_action = discrete_to_multi_discrete(discrete_action)
    It can be initialized using 3 configurations:
    Configuration 1) - DiscreteToMultiDiscrete(multi_discrete)                   [2nd param is empty]
        Would adapt to a Discrete action space of size (1 + nb of discrete in MultiDiscrete)
        where
            0   returns NOOP                                [  0,   0,   0, ...]
            1   returns max for the first discrete space    [max,   0,   0, ...]
            2   returns max for the second discrete space   [  0, max,   0, ...]
            etc.
    Configuration 2) - DiscreteToMultiDiscrete(multi_discrete, list_of_discrete) [2nd param is a list]
        Would adapt to a Discrete action space of size (1 + nb of items in list_of_discrete)
        e.g.
        if list_of_discrete = [0, 2]
            0   returns NOOP                                [  0,   0,   0, ...]
            1   returns max for first discrete in list      [max,   0,   0, ...]
            2   returns max for second discrete in list     [  0,   0,  max, ...]
            etc.
    Configuration 3) - DiscreteToMultiDiscrete(multi_discrete, discrete_mapping) [2nd param is a dict]
        Would adapt to a Discrete action space of size (nb_keys in discrete_mapping)
        where discrete_mapping is a dictionnary in the format { discrete_key: multi_discrete_mapping }
        e.g. for the Nintendo Game Controller [ [0,4], [0,1], [0,1] ] a possible mapping might be;
        mapping = {
            0:  [0, 0, 0],  # NOOP
            1:  [1, 0, 0],  # Up
            2:  [3, 0, 0],  # Down
            3:  [2, 0, 0],  # Right
            4:  [2, 1, 0],  # Right + A
            5:  [2, 0, 1],  # Right + B
            6:  [2, 1, 1],  # Right + A + B
            7:  [4, 0, 0],  # Left
            8:  [4, 1, 0],  # Left + A
            9:  [4, 0, 1],  # Left + B
            10: [4, 1, 1],  # Left + A + B
            11: [0, 1, 0],  # A only
            12: [0, 0, 1],  # B only,
            13: [0, 1, 1],  # A + B
        }
    """

    def __init__(self, multi_discrete, options=None):
        # assert isinstance(multi_discrete, MultiDiscrete)
        self.multi_discrete = multi_discrete
        self.num_discrete_space = self.multi_discrete.n

        # Config 1
        if options is None:
            self.n = self.num_discrete_space + 1  # +1 for NOOP at beginning
            self.mapping = {i: [0] * self.num_discrete_space for i in range(self.n)}
            for i in range(self.num_discrete_space):
                self.mapping[i + 1][i] = self.multi_discrete.high[i]

        # Config 2
        elif isinstance(options, list):
            assert len(options) <= self.num_discrete_space
            self.n = len(options) + 1  # +1 for NOOP at beginning
            self.mapping = {i: [0] * self.num_discrete_space for i in range(self.n)}
            for i, disc_num in enumerate(options):
                assert disc_num < self.num_discrete_space
                self.mapping[i + 1][disc_num] = self.multi_discrete.high[disc_num]

        # Config 3
        elif isinstance(options, dict):
            self.n = len(options.keys())
            self.mapping = options
            # for i, key in enumerate(options.keys()):
            #     if i != key:
            #         raise Error('DiscreteToMultiDiscrete must contain ordered keys. ' \
            #                     'Item {0} should have a key of "{0}", but key "{1}" found instead.'.format(i, key))
            #     if not self.multi_discrete.contains(options[key]):
            #         raise Error('DiscreteToMultiDiscrete mapping for key {0} is ' \
            #                     'not contained in the underlying MultiDiscrete action space. ' \
            #                     'Invalid mapping: {1}'.format(key, options[key]))
        # Unknown parameter provided
        else:
            raise Error('DiscreteToMultiDiscrete - Invalid parameter provided.')

    def __call__(self, discrete_action):
        return self.mapping[discrete_action]


# Discrete Action Wrapper
# Constants
NUM_ACTIONS = 43
ALLOWED_ACTIONS = [
    [0, 10, 11],  # 0 - Basic
    [0, 10, 11, 13, 14, 15],  # 1 - Corridor
    [0, 14, 15],  # 2 - DefendCenter
    [0, 14, 15],  # 3 - DefendLine
    [13, 14, 15],  # 4 - HealthGathering
    [13, 14, 15],  # 5 - MyWayHome
    [0, 14, 15],  # 6 - PredictPosition
    [10, 11],  # 7 - TakeCover
    [x for x in range(NUM_ACTIONS) if x != 33],  # 8 - Deathmatch
    [13, 14, 15],  # 9 - MyWayHomeFixed
    [13, 14, 15],  # 10 - MyWayHomeFixed15
    [13, 14, 15],  # 10 - MyWayHomeFixed15
    [13, 14, 15],  # 10 - MyWayHomeFixed15
    [13, 14, 15],  # 10 - MyWayHomeFixed15
    [13, 14, 15],  # 10 - MyWayHomeFixed15
    [13, 14, 15],  # 10 - MyWayHomeFixed15
    [13, 14, 15],  # 10 - MyWayHomeFixed15
    [13, 14, 15]
]


def ToDiscrete(config):
    # Config can be 'minimal', 'constant-7', 'constant-17', 'full'

    class ToDiscreteWrapper(gym.Wrapper):
        """
            Doom wrapper to convert MultiDiscrete action space to Discrete
            config:
                - minimal - Will only use the levels' allowed actions (+ NOOP)
                - constant-7 - Will use the 7 minimum actions (+NOOP) to complete all levels
                - constant-17 - Will use the 17 most common actions (+NOOP) to complete all levels
                - full - Will use all available actions (+ NOOP)
            list of commands:
                - minimal:
                    Basic:              NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT
                    Corridor:           NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    DefendCenter        NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    DefendLine:         NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    HealthGathering:    NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    MyWayHome:          NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    PredictPosition:    NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    TakeCover:          NOOP, MOVE_RIGHT, MOVE_LEFT
                    Deathmatch:         NOOP, ALL COMMANDS (Deltas are limited to [0,1] range and will not work properly)
                - constant-7: NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, SELECT_NEXT_WEAPON
                - constant-17: NOOP, ATTACK, JUMP, CROUCH, TURN180, RELOAD, SPEED, STRAFE, MOVE_RIGHT, MOVE_LEFT, MOVE_BACKWARD
                                MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, LOOK_UP, LOOK_DOWN, SELECT_NEXT_WEAPON, SELECT_PREV_WEAPON
        """

        def __init__(self, env):
            super(ToDiscreteWrapper, self).__init__(env)
            if config == 'minimal':
                allowed_actions = ALLOWED_ACTIONS[self.unwrapped.level]
            elif config == 'constant-7':
                allowed_actions = [0, 10, 11, 13, 14, 15, 31]
            elif config == 'constant-17':
                allowed_actions = [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32]
            elif config == 'full':
                allowed_actions = None
            else:
                raise gym.error.Error(
                    'Invalid configuration. Valid options are "minimal", "constant-7", "constant-17", "full"')
            self.action_space = DiscreteToMultiDiscrete(self.action_space, allowed_actions)

        def _step(self, action):
            return self.env._step(self.action_space(action))

    return ToDiscreteWrapper


def wrap_vizdoom(env, stack=4, action_repeat=4):
    # resolution_wrapper = SetResolution("160x120")
    # env = resolution_wrapper(env)
    if action_repeat > 1:
        env = MaxAndSkipEnv(env, skip=action_repeat, max_over=1)
    env = WarpFrame2(env, res=42)
    env = ClipNegativeRewardEnv(env)
    # env = FrameStack(env, 4)
    # env = GreyscaleRender(env)
    return env


def ToDiscreteMario():
    class ToDiscreteWrapper(gym.Wrapper):
        """
            Wrapper to convert MultiDiscrete action space to Discrete
            Only supports one config, which maps to the most logical discrete space possible
        """

        def __init__(self, env):
            super(ToDiscreteWrapper, self).__init__(env)
            mapping = {
                0: [0, 0, 0, 0, 0, 0],  # NOOP
                1: [1, 0, 0, 0, 0, 0],  # Up
                2: [0, 0, 1, 0, 0, 0],  # Down
                3: [0, 1, 0, 0, 0, 0],  # Left
                4: [0, 1, 0, 0, 1, 0],  # Left + A
                5: [0, 1, 0, 0, 0, 1],  # Left + B
                6: [0, 1, 0, 0, 1, 1],  # Left + A + B
                7: [0, 0, 0, 1, 0, 0],  # Right
                8: [0, 0, 0, 1, 1, 0],  # Right + A
                9: [0, 0, 0, 1, 0, 1],  # Right + B
                10: [0, 0, 0, 1, 1, 1],  # Right + A + B
                11: [0, 0, 0, 0, 1, 0],  # A
                12: [0, 0, 0, 0, 0, 1],  # B
                13: [0, 0, 0, 0, 1, 1],  # A + B
            }
            self.action_space = DiscreteToMultiDiscrete(self.action_space, mapping)

        def _step(self, action):
            return self.env.step(self.action_space(action))

    return ToDiscreteWrapper

# FROm https://github.com/openai/large-scale-curiosity/blob/master/wrappers.py
import itertools

class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    '''
    Reproduces the action space from curiosity paper.
    '''

    def __init__(self, env, all_buttons, whitelist=KNOWN_BUTTONS | KNOWN_SHOULDERS):
        gym.ActionWrapper.__init__(self, env)

        self._num_buttons = len(all_buttons)
        button_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_BUTTONS}
        buttons = [(), *zip(button_keys), *itertools.combinations(button_keys, 2)]
        shoulder_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_SHOULDERS}
        shoulders = [(), *zip(shoulder_keys), *itertools.permutations(shoulder_keys, 2)]
        arrows = [(), (4,), (5,), (6,), (7,)]  # (), up, down, left, right
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask

# From https://github.com/pathak22/noreward-rl/blob/master/src/env_wrapper.py
class MarioEnv(gym.Wrapper):
    def __init__(self, env=None, tilesEnv=False):
        """Reset mario environment without actually restarting fceux everytime.
        This speeds up unrolling by approximately 10 times.
        """
        super(MarioEnv, self).__init__(env)
        self.resetCount = -1
        # reward is distance travelled. So normalize it with total distance
        # https://github.com/ppaquette/gym-super-mario/blob/master/ppaquette_gym_super_mario/lua/super-mario-bros.lua
        # However, we will not use this reward at all. It is only for completion.
        self.maxDistance = 3000.0
        self.tilesEnv = tilesEnv

    def _reset(self):
        if self.resetCount < 0:
            print('\nDoing hard mario fceux reset (4 seconds wait) !')
            sys.stdout.flush()
            self.env.reset()
            time.sleep(4)
        obs, _, _, info = self.env.step(7)  # take right once to start game
        if info.get('ignore', False):  # assuming this happens only in beginning
            self.resetCount = -1
            self.env.close()
            return self._reset()
        self.resetCount = info.get('iteration', -1)
        if self.tilesEnv:
            return obs
        return obs[24:-12, 8:-8, :]

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info.get('ignore', True):
            return self.reset(), 0, False, info
        # print('info:', info)
        done = info['iteration'] > self.resetCount
        reward = float(reward) / self.maxDistance  # note: we do not use this rewards at all.
        if self.tilesEnv:
            return obs, reward, done, info
        return obs[24:-12, 8:-8, :], reward, done, info

    def _close(self):
        self.resetCount = -1
        return self.env.close()


def wrap_mario(env, stack=4, buttons=None):
    # env = MarioEnv(env)
    # buttons = env.BUTTONS
    env = MaxAndSkipEnv(env, skip=4, max_over=1)
    env = WarpFrame(env, res=42)
    # env = FrameStack(env, 4)
    env = GreyscaleRender(env)
    # discrete_action_wrapper = ToDiscreteMario()
    env = LimitedDiscreteActions(env, buttons)
    # env = discrete_action_wrapper(env)
    # No reward
    env = NoRewardEnv(env)
    print("Wrapping mario env")
    return env


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=self.visited_rooms.copy())
            self.visited_rooms.clear()
        return obs, rew, done, info


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class WarpFrame2(gym.ObservationWrapper):
    def __init__(self, env, res=84):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = res
        self.height = res
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

def make_montezuma(env, max_episode_steps=4500):
        # env = gym.make(env_id)
        env._max_episode_steps = max_episode_steps * 4
        assert 'NoFrameskip' in env.spec.id
        env = StickyActionEnv(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = MontezumaInfoWrapper(env, room_address=3)
        env = WarpFrame2(env)
        return env

def make_montezuma_ram(env, max_episode_steps=4500):
        # env = gym.make(env_id)
        env._max_episode_steps = max_episode_steps * 4
        assert 'NoFrameskip' in env.spec.id
        env = StickyActionEnv(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = MontezumaInfoWrapper(env, room_address=3)
        # env = WarpFrame2(env)
        return env

def make_atari(env, max_episode_steps=4500):
    # env = gym.make(env_id)
    env._max_episode_steps = max_episode_steps * 4
    assert 'NoFrameskip' in env.spec.id
    env = StickyActionEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    # env = MontezumaInfoWrapper(env, room_address=3)
    env = WarpFrame2(env)
    return env
