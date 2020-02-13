import gym
import numpy as np
from gym import spaces


class NChain(gym.Env):

    # Our wrapper handles the drawing
    # metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n=10, p=1, back=False):

        self.n = n # Size of NChain
        self.p = p # Probability of succesfully transitioning right
        self.back = back # If we unsuccesfully transition right, do we go back a state

        self.action_space = spaces.Discrete(2)

        self.limit = n + 9
        # Increase the limit if p < 1
        # self.limit *= 1 / p
        self.steps = 0

        self._seed = 56
        np.random.seed(self._seed)
        self.actions = [np.random.randint(2) for _ in range(n)] # If we should reverse the actions at this state

        self.observation_space = spaces.Box(low=0, high=1, shape=(n,))

    def update_limit(self, new_limit):
        self.limit = new_limit
        print("New Limit:", self.limit)

    def _step(self, a):
        self.steps += 1

        a_r = (a + self.actions[self.pos]) % 2

        r = 0
        if self.pos == 0 and a_r == 0:
            r = 0.001
        if self.pos == self.n - 1 and a_r == 1:
            r = 1

        new_pos = self.pos
        if a_r == 0:
            new_pos = max(0, self.pos - 1)
        elif a_r == 1:
            if np.random.random() < self.p:
                # Succesfully transition right
                new_pos = min(self.n - 1, self.pos + 1)
            else:
                if self.back:
                    # Go back
                    new_pos = max(0, self.pos - 1)
                else:
                    # Stay put
                    new_pos = self.pos

        self.pos = new_pos

        s = self._get_therm()

        info = {}
        finished = False
        if self.steps == self.limit:
            finished = True
            info["Steps_Termination"] = True

        return s, r, finished, info

    def _get_therm(self):
        s = np.zeros(shape=(self.n,), dtype=np.float32)
        s[:self.pos + 1] = 1
        return s

    def _reset(self):
        self.steps = 0
        self.pos = 1
        return self._get_therm()

    def _render(self, mode="rgb_array", close=False):
        pass

    def _close(self):
        pass

    def log_player_pos(self):
        return self.pos

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