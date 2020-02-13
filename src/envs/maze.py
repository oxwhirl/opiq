from .gridworld import GridWorld
import numpy as np
import random
import logging


class SnakingMaze(GridWorld):

    def __init__(self, size, rnd_seed=13, corridor_width=10, neg_reward=True, randomise=False, num_actions=4, danger=False):
        self.size = size
        self.rnd_seed = rnd_seed
        self._seed = self.rnd_seed
        self.corridor_width = corridor_width
        self.logger = logging.getLogger("Maze")
        super().__init__(randomise=randomise, num_actions=num_actions, danger=danger)
        self.logger.critical("Maze of size: {}".format(self.grid.shape))

        # Count number of states
        num_states = 0
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x, y] != 1:
                    num_states += 1

        self.logger.critical("Number of states: {}".format(num_states))

        # No negative reward at each timestep
        if not neg_reward:
            self.negative_reward = 0
        self.positive_reward = +10 # Clipped to +1 if we are reward clipping

    def create_grid(self):
        self.grid = np.ones(shape=(self.size * 10, self.size * 10))
        self.grid[1:-1, 1:-1] = 0
        # Goal
        self.grid[1, -2] = 2
        # Player
        self.grid[-2, 1] = 3

        # Copied from https://en.wikipedia.org/wiki/Maze_generation_algorithm#Python_code_examples
        num_rows = int(self.size)  # number of rows
        num_cols = int(self.size)  # number of columns
        corridor_width = self.corridor_width
        if self.size == 18:
            # Hack
            self.rnd_seed = 2
        random.seed(self.rnd_seed)
        # The array M is going to hold the array information for each cell.
        # The first four coordinates tell if walls exist on those sides
        # and the fifth indicates if the cell has been visited in the search.
        # M(LEFT, UP, RIGHT, DOWN, CHECK_IF_VISITED)
        M = np.zeros((num_rows, num_cols, 5), dtype=np.uint8)

        # The array image is going to be the output image to display
        image = np.zeros(
            (num_rows * corridor_width, num_cols * corridor_width), dtype=np.uint8
        )

        # Set starting row and column
        r = 0
        c = 0
        history = [(r, c)]  # The history is the stack of visited locations

        max_history_len = 0
        max_pair = (0, 0)
        offset = (0, 0)
        prev = (0, 0)
        prev_max = (0, 0)
        # Trace a path though the cells of the maze and open walls along the path.
        # We do this with a while loop, repeating the loop until there is no history,
        # which would mean we backtracked to the initial start.
        while history:
            M[r, c, 4] = 1  # designate this location as visited
            # check if the adjacent cells are valid for moving to
            check = []
            if c > 0 and M[r, c - 1, 4] == 0:
                check.append("L")
            if r > 0 and M[r - 1, c, 4] == 0:
                check.append("U")
            if c < num_cols - 1 and M[r, c + 1, 4] == 0:
                check.append("R")
            if r < num_rows - 1 and M[r + 1, c, 4] == 0:
                check.append("D")

            if len(history) > max_history_len:
                max_pair = (r, c)
                prev_max = (prev[0], prev[1])
                max_history_len = len(history)
            if len(check):  # If there is a valid cell to move to.
                # Mark the walls between cells as open if we move
                prev = (r, c)
                history.append([r, c])
                move_direction = random.choice(check)
                if move_direction == "L":
                    M[r, c, 0] = 1
                    c = c - 1
                    M[r, c, 2] = 1
                if move_direction == "U":
                    M[r, c, 1] = 1
                    r = r - 1
                    M[r, c, 3] = 1
                if move_direction == "R":
                    M[r, c, 2] = 1
                    c = c + 1
                    M[r, c, 0] = 1
                if move_direction == "D":
                    M[r, c, 3] = 1
                    r = r + 1
                    M[r, c, 1] = 1
            else:  # If there are no valid cells to move to.
                # retrace one step back in history if no move is possible
                r, c = history.pop()

        # Open the walls at the start and finish
        # M[0,0,0] = 1
        # M[num_rows-1,num_cols-1,2] = 1
        # print(M)

        # Generate the image for display
        for row in range(0, num_rows):
            for col in range(0, num_cols):
                cell_data = M[row, col]
                for i in range(
                    corridor_width * row + 1, corridor_width * row + corridor_width
                ):
                    image[
                        i,
                        range(
                            corridor_width * col + 1,
                            corridor_width * col + corridor_width,
                        ),
                    ] = 1
                    if cell_data[0] == 1:
                        image[
                            range(
                                corridor_width * row + 1,
                                corridor_width * row + corridor_width,
                            ),
                            corridor_width * col,
                        ] = 1
                    if cell_data[1] == 1:
                        image[
                            corridor_width * row,
                            range(
                                corridor_width * col + 1,
                                corridor_width * col + corridor_width,
                            ),
                        ] = 1
                    if cell_data[2] == 1:
                        image[
                            range(
                                corridor_width * row + 1,
                                corridor_width * row + corridor_width,
                            ),
                            corridor_width * col + corridor_width - 1,
                        ] = 1
                    if cell_data[3] == 1:
                        image[
                            corridor_width * row + corridor_width - 1,
                            range(
                                corridor_width * col + 1,
                                corridor_width * col + corridor_width,
                            ),
                        ] = 1

        image = 1 - image

        # Fill in the bottom and right with walls
        image[-1, :] = 1
        image[:, -1] = 1

        image[1, 1] = 3
        # print(max_pair, prev_max)
        rr, cc = max_pair
        xx = 0
        if rr > prev_max[0]:
            xx = 1
        yy = 0
        if cc > prev_max[1]:
            yy = 1
        image[
            rr * corridor_width + int((corridor_width) * xx) + (-1) ** xx,
            cc * corridor_width + int((corridor_width - 0) * yy) + (-1) ** yy,
        ] = 2
        # image[-2, -2] = 2
        self.grid = image
