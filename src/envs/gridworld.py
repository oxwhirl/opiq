import numpy as np
import gym
from gym import spaces
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridWorld(gym.Env):

    # --2d GridWorld--
    # 0 = Nothing
    # 1 = Wall
    # 2 = Goal
    # 3 = Player

    # Wrapper handles the drawing
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, randomise=False, num_actions=4, danger=False):
        self.reset()
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        if num_actions > 4:
            self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
        # Time-limit on the environment, 5 is arbitrary
        self.limit = self.grid.size * 10
        print("Limit:", self.limit)
        self.positive_reward = +1
        self.negative_reward = -0.005
        self.danger = danger

        # Gym Stuff
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.grid.shape)
        self.reward_range = (self.negative_reward, self.positive_reward)

        # Counting stuff
        self.counts = np.empty_like(self.grid)
        self.found_goal = False

        # Randomise actions
        self.total_actions = num_actions
        self.randomise = randomise

    def update_limit(self, new_limit):
        self.limit = new_limit
        print("New Limit:", self.limit)

    def get_randomised_action(self, a):
        state_seed = (937 * self.player_pos[0] + 79 * self.player_pos[1])
        all_actions = list(range(self.total_actions))
        for i in range(self.total_actions - 1):
            state_seed = (state_seed * 29) % 137
            to_swap = state_seed % (self.total_actions - i)
            to_swap += i
            temp = all_actions[i]
            all_actions[i] = all_actions[to_swap]
            all_actions[to_swap] = temp

        action_to_take = all_actions.index(a)
        action_to_take = min(action_to_take, 4 if self.total_actions > 4 else 3)

        return action_to_take

    def step(self, a):

        if self.randomise:
            a = self.get_randomised_action(a)

        info_dict = {}

        # Update counts
        self.counts[self.player_pos] += 1
        current_count = self.counts[self.player_pos]
        action_counts = []
        for aa in self.actions:
            new_player_pos = (self.player_pos[0] + aa[0], self.player_pos[1] + aa[1])
            # Clip
            if (
                new_player_pos[0] < 0
                or new_player_pos[0] >= self.grid.shape[0]
                or new_player_pos[1] < 0
                or new_player_pos[1] >= self.grid.shape[1]
            ):
                new_player_pos = self.player_pos

            # Into a wall
            if self.grid[new_player_pos] == 1:
                new_player_pos = self.player_pos

            action_counts.append(self.counts[new_player_pos])

        self.steps += 1
        new_player_pos = (
            self.player_pos[0] + self.actions[a][0],
            self.player_pos[1] + self.actions[a][1],
        )
        # Clip
        if (
            new_player_pos[0] < 0
            or new_player_pos[0] >= self.grid.shape[0]
            or new_player_pos[1] < 0
            or new_player_pos[1] >= self.grid.shape[1]
        ):
            new_player_pos = self.player_pos

        r = self.negative_reward

        finished = False

        # Into a wall
        if self.grid[new_player_pos] == 1:
            new_player_pos = self.player_pos
            if self.danger:
                finished = True
        # Into a goal
        elif self.grid[new_player_pos] == 2:
            r += self.positive_reward
            self.found_goal = True
            self.goals -= 1
            if self.goals == 0:
                finished = True

        self.grid[self.player_pos] = 0
        self.grid[new_player_pos] = 3
        self.player_pos = new_player_pos

        if self.danger and finished:
            # Gone into wall
            r = -1

        if self.steps >= self.limit and not finished:
            finished = True
            info_dict["Steps_Termination"] = True

        # Fill in info dict with the action selection statistics
        new_state_count = self.counts[new_player_pos]
        count_list = [current_count] + action_counts + [new_state_count]
        info_dict["Action_Counts"] = np.array(count_list)

        info_dict["Found Goal"] = self.found_goal

        return self.grid[:, :, np.newaxis] / 3, r, finished, info_dict

    def reset(self):
        self.steps = 0
        self.create_grid()
        player_pos_np = np.argwhere(self.grid == 3)[0]
        self.player_pos = (player_pos_np[0], player_pos_np[1])
        self.goals = (self.grid == 2).sum()
        self.num_goals = self.goals
        self.goals_order = np.argwhere(self.grid == 2)
        # print(self.goals_order)
        return self.grid[:, :, np.newaxis] / 3

    def render(self, mode="rgb_array", close=False):
        if mode == "rgb_array":
            grid = self.grid
            image = np.zeros(shape=(grid.shape[0], grid.shape[1], 3))
            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    if grid[x, y] != 0:
                        image[x, y] = (
                            255 * grid[x, y] / 3,
                            255 * grid[x, y] / 3,
                            255 * grid[x, y] / 3,
                        )
            return image
        else:
            pass
            # raise Exception("Cannot do human rendering")

    def state_to_image(self, state):
        grid = state
        image = np.zeros(shape=(grid.shape[0], grid.shape[1], 3))
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] != 0:
                    image[x, y] = (
                        255 * grid[x, y] / 3,
                        255 * grid[x, y] / 3,
                        255 * grid[x, y] / 3,
                    )
        return image

    def create_grid(self):
        self.grid = np.array([[3, 0], [1, 2]])

    def log_player_pos(self):
        goals_list = [self.grid[g[0], g[1]] == 2 for g in self.goals_order]
        # print(goals_list)
        player_pos = list(self.player_pos)
        joint = player_pos + goals_list
        # print(tuple(joint))
        return tuple(joint)

    def state_to_player_pos(self, state):
        internal_state = state[:, :, 0]
        goals_list = [
            internal_state[g[0], g[1]] > 0.6 and internal_state[g[0], g[1]] < 0.7
            for g in self.goals_order
        ]
        # print(goals_list)
        player_pos = list(np.argwhere(internal_state > 0.9)[0])
        joint = player_pos + goals_list
        # print(tuple(joint))
        return tuple(joint)

    def trained_on_states(self, player_visits, args):

        # interval = args.exp_replay_size

        if self.num_goals > 3:
            raise Exception("Cant do trained on states for >3 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        # Works for num_goals <= 3
        # print(self.grid.shape)
        canvas = np.zeros(
            (
                self.grid.shape[0] * self.num_goals,
                self.grid.shape[1] * self.num_goals,
                3,
            )
        )
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]

        # end_t = int(args.t_max * i / 100) * args.batch_size
        # start_t = int(args.t_max * (i - 1) / 100) * args.batch_size
        # print("\n\n\n\n",start_t, end_t)

        for visit in player_visits:
            # print(visit)
            px = visit[0]
            py = visit[1]

            np_goals = np.array(visit[2:])
            goal_colours = [ig for ig, c in enumerate(visit[2:]) if c == True]
            if goal_colours == []:
                goal_colours = [0, 1, 2]
            x_place = px + grid_x * (np_goals == False).sum()
            yy = 0
            if (np_goals == False).sum() == 1:
                yy = np.argwhere(np_goals == False)[0]
            elif (np_goals == False).sum() == 2:
                yy = np.argwhere(np_goals == True)[0]
            y_place = py + grid_y * yy

            # print(x_place, y_place, goal_colours, canvas.shape)
            canvas[x_place, y_place, goal_colours] += 1

        if np.max(canvas) == 0:
            return
        canvas = canvas / np.max(canvas)

        # TODO: Colour the unvisited goals
        for goal in self.goals_order:
            canvas[goal[0], goal[1], :] = 2 / 3
        if self.num_goals >= 2:
            for g in range(self.num_goals):
                for go_i, goal in enumerate(self.goals_order):
                    if go_i != g:
                        canvas[goal[0] + grid_x, goal[1] + g * grid_y, :] = 2 / 3
        if self.num_goals >= 3:
            for g in range(self.num_goals):
                for go_i, goal in enumerate(self.goals_order):
                    if go_i == g:
                        canvas[goal[0] + 2 * grid_x, goal[1] + g * grid_y, :] = 2 / 3

        # The walls
        for x in range(grid_x):
            for y in range(grid_y):
                if self.grid[x, y] == 1:
                    canvas[x, y, :] = 1 / 3
                    for zx in range(1, self.num_goals):
                        for zy in range(self.num_goals):
                            canvas[zx * grid_x + x, zy * grid_y + y, :] = 1 / 3

        # Seperate the mazes
        canvas = np.insert(
            canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=0
        )
        canvas = np.insert(
            canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=1
        )
        canvas[0:grid_x, grid_y + 1 :, :] = 0
        colour_maze = canvas

        colour_maze = np.clip(colour_maze, 0, 1) * 255
        # colour_maze = np.swapaxes(colour_maze, 0, 1)
        colour_images.append(colour_maze.astype(np.uint8))
        return colour_images[0]

    def xp_and_frontier_states(self):
        # We should have already computed the xp replay and frontier images
        xp_replay_image = self.xp_replay_image
        frontier_colours = self.frontier_image

        if frontier_colours.shape[1] != xp_replay_image.shape[1]:
            tiled_xp_replay_image = np.empty_like(frontier_colours)
            times_to_tile = frontier_colours.shape[1] // xp_replay_image.shape[1]
            for i in range(times_to_tile):
                tiled_xp_replay_image[
                    :,
                    i * xp_replay_image.shape[1] : (i + 1) * xp_replay_image.shape[1],
                    :,
                ] = xp_replay_image
            xp_replay_image = tiled_xp_replay_image

        overlayed_image = xp_replay_image + frontier_colours

        return overlayed_image

    def bonus_xp_and_frontier_states(self):
        # We should have already computer the xp replay and frontier images
        xp_replay_image = self.bonus_replay_image
        frontier_colours = self.frontier_image

        if frontier_colours.shape[1] != xp_replay_image.shape[1]:
            tiled_xp_replay_image = np.empty_like(frontier_colours)
            times_to_tile = frontier_colours.shape[1] // xp_replay_image.shape[1]
            for i in range(times_to_tile):
                tiled_xp_replay_image[
                    :,
                    i * xp_replay_image.shape[1] : (i + 1) * xp_replay_image.shape[1],
                    :,
                ] = xp_replay_image
            xp_replay_image = tiled_xp_replay_image

        overlayed_image = xp_replay_image + frontier_colours

        return overlayed_image

    def visits_and_frontier_states(self):
        visits_image = self.player_visits_image
        frontier_colours = self.frontier_image

        if frontier_colours.shape[1] != visits_image.shape[1]:
            tiled_xp_replay_image = np.empty_like(frontier_colours)
            times_to_tile = frontier_colours.shape[1] // visits_image.shape[1]
            for i in range(times_to_tile):
                tiled_xp_replay_image[
                    :, i * visits_image.shape[1] : (i + 1) * visits_image.shape[1], :
                ] = visits_image
            visits_image = tiled_xp_replay_image

        overlayed_image = visits_image + frontier_colours

        return overlayed_image

    def xp_replay_states(self, player_visits, args, bonus_replay=False):

        # interval = args.exp_replay_size

        if self.num_goals > 1:
            raise Exception("Cant do xp replay states for >1 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        # Works for num_goals <= 3
        # print(self.grid.shape)
        canvas = np.zeros(
            (
                self.grid.shape[0] * self.num_goals,
                self.grid.shape[1] * self.num_goals,
                3,
            )
        )
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]

        # end_t = int(args.t_max * i / 100)
        # start_t = max(0, end_t - args.exp_replay_size)

        # print("\n\n")
        # print(self.num_goals)
        # print(self.goals_order)
        # print(canvas.shape)
        # print("\n\n")

        for visit in player_visits:
            # print(visit)
            px = visit[0]
            py = visit[1]

            np_goals = np.array(visit[2:])
            goal_colours = [ig for ig, c in enumerate(visit[2:]) if c == True]
            if goal_colours == []:
                goal_colours = [0, 1, 2]
            x_place = px  #
            yy = 0
            if (np_goals == False).sum() == 1:
                yy = np.argwhere(np_goals == False)[0]
            elif (np_goals == False).sum() == 2:
                yy = np.argwhere(np_goals == True)[0]
            y_place = py  #

            # print(x_place, y_place, goal_colours, canvas.shape)
            if x_place >= grid_x or y_place >= grid_y:
                print(px, py, np_goals, goal_colours, x_place, y_place)
            canvas[x_place, y_place, goal_colours] = 1

        if np.max(canvas) == 0:
            return
        # canvas = canvas / (np.max(canvas) / scaling)

        # TODO: Colour the unvisited goals
        for goal in self.goals_order:
            canvas[goal[0], goal[1], :] = 2 / 3
        if self.num_goals >= 2:
            for g in range(self.num_goals):
                for go_i, goal in enumerate(self.goals_order):
                    if go_i != g:
                        canvas[goal[0] + grid_x, goal[1] + g * grid_y, :] = 2 / 3
        if self.num_goals >= 3:
            for g in range(self.num_goals):
                for go_i, goal in enumerate(self.goals_order):
                    if go_i == g:
                        canvas[goal[0] + 2 * grid_x, goal[1] + g * grid_y, :] = 2 / 3

        # The walls
        for x in range(grid_x):
            for y in range(grid_y):
                if self.grid[x, y] == 1:
                    canvas[x, y, :] = 1 / 3
                    for zx in range(1, self.num_goals):
                        for zy in range(self.num_goals):
                            canvas[zx * grid_x + x, zy * grid_y + y, :] = 1 / 3

        # Seperate the mazes
        canvas = np.insert(
            canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=0
        )
        canvas = np.insert(
            canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=1
        )
        canvas[0:grid_x, grid_y + 1 :, :] = 0

        # Flip red and blue to make the bonus replay states blue instead of red
        if bonus_replay:
            # Blue instead of red
            red_canvas = np.copy(canvas[:, :, 0])
            canvas[:, :, 0] = canvas[:, :, 2]
            canvas[:, :, 2] = red_canvas

        colour_maze = canvas

        colour_maze = np.clip(colour_maze, 0, 1) * 255
        # colour_maze = np.swapaxes(colour_maze, 0, 1)
        colour_images.append(colour_maze.astype(np.uint8))

        if not bonus_replay:
            self.xp_replay_image = np.copy(colour_images[0])
        else:
            self.bonus_replay_image = np.copy(colour_images[0])

        return colour_images[0]
        # save_video("{}/visitations/Goal_Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)

    def player_visits(self, player_visits, args):
        # Log the visitations
        # with open("{}/logs/Player_Positions.txt".format(args.log_path), "a") as file:
        #     file.write('\n'.join(" ".join(str(x) for x in t) for t in player_visits))

        scaling = 2

        if self.num_goals > 3:
            raise Exception("Cant do state visitations for >3 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        # Works for num_goals <= 3
        # print(self.grid.shape)
        canvas = np.zeros(
            (
                self.grid.shape[0] * self.num_goals,
                self.grid.shape[1] * self.num_goals,
                3,
            )
        )
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]

        for visit in player_visits:
            px = visit[0]
            py = visit[1]

            np_goals = np.array(visit[2:])
            goal_colours = [ig for ig, c in enumerate(visit[2:]) if c == True]
            if goal_colours == []:
                goal_colours = [0, 1, 2]
            x_place = px + grid_x * (np_goals == False).sum()
            yy = 0
            if (np_goals == False).sum() == 1:
                yy = np.argwhere(np_goals == False)[0]
            elif (np_goals == False).sum() == 2:
                yy = np.argwhere(np_goals == True)[0]
            y_place = py + grid_y * yy

            # print(x_place, y_place, goal_colours, canvas.shape)
            canvas[x_place, y_place, goal_colours] += 1

        if np.max(canvas) == 0:
            return
        canvas = canvas / (np.max(canvas) / scaling)

        # TODO: Colour the unvisited goals
        for goal in self.goals_order:
            canvas[goal[0], goal[1], :] = 2 / 3
        if self.num_goals >= 2:
            for g in range(self.num_goals):
                for go_i, goal in enumerate(self.goals_order):
                    if go_i != g:
                        canvas[goal[0] + grid_x, goal[1] + g * grid_y, :] = 2 / 3
        if self.num_goals >= 3:
            for g in range(self.num_goals):
                for go_i, goal in enumerate(self.goals_order):
                    if go_i == g:
                        canvas[goal[0] + 2 * grid_x, goal[1] + g * grid_y, :] = 2 / 3

        # The walls
        for x in range(grid_x):
            for y in range(grid_y):
                if self.grid[x, y] == 1:
                    canvas[x, y, :] = 1 / 3
                    for zx in range(1, self.num_goals):
                        for zy in range(self.num_goals):
                            canvas[zx * grid_x + x, zy * grid_y + y, :] = 1 / 3

        # Seperate the mazes
        canvas = np.insert(
            canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=0
        )
        canvas = np.insert(
            canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=1
        )
        canvas[0:grid_x, grid_y + 1 :, :] = 0
        colour_maze = canvas

        colour_maze = np.clip(colour_maze, 0, 1) * 255
        # colour_maze = np.swapaxes(colour_maze, 0, 1)
        colour_images.append(colour_maze.astype(np.uint8))

        self.player_visits_image = np.copy(colour_images[0])

        return colour_images[0]
        # save_video("{}/visitations/Goal_Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)

    def bonus_landscape(self, player_visits, exploration_bonuses, max_bonus, args):
        # interval = int(args.t_max / args.interval_size)
        # scaling = 2

        if self.num_goals > 3:
            raise Exception("Cant do bonus landscape for >3 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        # for i in range(0, args.t_max, interval // 10):
        # Works for num_goals <= 3
        # print(self.grid.shape)
        canvas = np.zeros(
            (
                self.grid.shape[0] * self.num_goals,
                self.grid.shape[1] * self.num_goals,
                3,
            )
        )
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]

        for visit, bonus in zip(player_visits, exploration_bonuses):
            relative_bonus = bonus / max_bonus
            px = visit[0]
            py = visit[1]

            np_goals = np.array(visit[2:])
            goal_colours = [ig for ig, c in enumerate(visit[2:]) if c == True]
            if goal_colours == []:
                goal_colours = [0, 1, 2]
            x_place = px + grid_x * (np_goals == False).sum()
            yy = 0
            if (np_goals == False).sum() == 1:
                yy = np.argwhere(np_goals == False)[0]
            elif (np_goals == False).sum() == 2:
                yy = np.argwhere(np_goals == True)[0]
            y_place = py + grid_y * yy

            # print(x_place, y_place, goal_colours, canvas.shape)
            canvas[x_place, y_place, goal_colours] = max(
                relative_bonus, canvas[x_place, y_place, goal_colours[0]]
            )

        if np.max(canvas) == 0:
            return
        # canvas = canvas / np.max(canvas)

        # TODO: Colour the unvisited goals
        for goal in self.goals_order:
            canvas[goal[0], goal[1], :] = 2 / 3
        if self.num_goals >= 2:
            for g in range(self.num_goals):
                for go_i, goal in enumerate(self.goals_order):
                    if go_i != g:
                        canvas[goal[0] + grid_x, goal[1] + g * grid_y, :] = 2 / 3
        if self.num_goals >= 3:
            for g in range(self.num_goals):
                for go_i, goal in enumerate(self.goals_order):
                    if go_i == g:
                        canvas[goal[0] + 2 * grid_x, goal[1] + g * grid_y, :] = 2 / 3

        # The walls
        for x in range(grid_x):
            for y in range(grid_y):
                if self.grid[x, y] == 1:
                    canvas[x, y, :] = 1 / 3
                    for zx in range(1, self.num_goals):
                        for zy in range(self.num_goals):
                            canvas[zx * grid_x + x, zy * grid_y + y, :] = 1 / 3

        # Seperate the mazes
        canvas = np.insert(
            canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=0
        )
        canvas = np.insert(
            canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=1
        )
        canvas[0:grid_x, grid_y + 1 :, :] = 0
        colour_maze = canvas

        colour_maze = np.clip(colour_maze, 0, 1) * 255
        # colour_maze = np.swapaxes(colour_maze, 0, 1)
        colour_images.append(colour_maze.astype(np.uint8))
        return colour_images[0]

    def count_state_action_space(self, count_model, args):

        actions = args.num_actions
        canvas = np.zeros(
            (
                self.grid.shape[0] * self.num_goals,
                self.grid.shape[1] * self.num_goals * actions,
                3,
            )
        )
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]
        grid = self.grid

        counts = np.zeros((self.grid.shape[0] * self.num_goals, self.grid.shape[1] * self.num_goals * actions))

        # TODO: Batch the states if we need more efficiency
        for a in range(actions):
            for x in range(grid_x):
                for y in range(grid_y):

                    if grid[x, y] == 1 or grid[x, y] == 2:
                        # If the position is a wall the player cannot ever be there
                        continue

                    state_copy = np.copy(self.grid)
                    state_copy[self.player_pos] = 0
                    state_copy[x, y] = 3
                    # print(state_copy)
                    state_copy = state_copy[np.newaxis, :, :] / 3

                    count = count_model.get_count(torch.tensor([state_copy], device=device, dtype=torch.float32), action=a)
                    # print(x,y,bonus)
                    canvas[x, y + (grid_y * a), 1] = count
                    counts[x, y + (grid_y * a)] = count

        # canvas /= np.max(canvas)
        max_count = 500
        canvas = canvas.clip(min=0, max=max_count)
        canvas /= max_count

        # Walls
        for a in range(actions):
            for x in range(grid_x):
                for y in range(grid_y):
                    if grid[x, y] == 1 or grid[x, y] == 2:
                        canvas[x, y + (grid_y * a), :] = grid[x, y] / 3

        canvas = np.clip(canvas, 0, 1) * 255

        return canvas, counts

    def np_q_vals(self, count_model, nn, args):
        actions = args.num_actions
        canvas = np.zeros(
            (
                self.grid.shape[0] * self.num_goals,
                self.grid.shape[1] * self.num_goals * actions,
                3,
            )
        )
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]
        grid = self.grid

        # TODO: Batch the states if we need more efficiency
        for x in range(grid_x):
            for y in range(grid_y):

                if grid[x, y] == 1 or grid[x, y] == 2:
                    # If the position is a wall the player cannot ever be there
                    continue

                state_copy = np.copy(self.grid)
                state_copy[self.player_pos] = 0
                state_copy[x, y] = 3
                # print(state_copy)
                state_copy = state_copy[np.newaxis, :, :] / 3

                state_tensor = torch.tensor([state_copy], device=device, dtype=torch.float32)
                # state_tensor = state_tensor.transpose(1,3).transpose(2,3)
                q_vals = nn(state_tensor)[0]
                # Inefficient
                for a in range(actions):
                    canvas[x, y + (grid_y * a), 0] = q_vals[a]

                    if args.optim_bootstrap:
                        count = count_model.get_count(state_tensor, action=a)
                        # print(x,y,bonus)
                        canvas[x, y + (grid_y * a), 0] += args.optim_bootstrap_tau / ((count + 1.0) ** args.optim_m)
        return canvas

    def q_value_estimates(self, count_model, nn, args):
        q_vals = self.np_q_vals(count_model, nn, args)
        canvas = np.copy(q_vals)
        actions = args.num_actions
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]
        grid = self.grid

        # canvas /= np.max(canvas)
        max_count = self.positive_reward if args.reward_clipping is False else +1
        max_count *= 1.5
        canvas = canvas.clip(min=0, max=max_count)
        canvas /= max_count

        # Walls
        for a in range(actions):
            for x in range(grid_x):
                for y in range(grid_y):
                    if grid[x, y] == 1 or grid[x, y] == 2:
                        canvas[x, y + (grid_y * a), :] = grid[x, y] / 3

        canvas = np.clip(canvas, 0, 1) * 255

        return canvas, q_vals
