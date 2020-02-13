import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, args):
        super(DQN, self).__init__()

        self.logger = logging.getLogger("Maze_DQN")
        self.input_frames = args.past_frames_input

        maze_size = args.state_shape[0]

        image_size = maze_size
        self.logger.critical(
            "Maze image input: ({1},{0},{0})".format(image_size, self.input_frames)
        )

        stride = 2
        stride_third = 1
        if maze_size <= 9:
            stride = 1
        if maze_size > 40:
            stride_third = 2

        channels = 32
        # if image_size < 25:
        #     channels = 16

        convs = 2
        if image_size > 26:
            convs = 3
        self.convs = convs

        self.conv1 = nn.Conv2d(self.input_frames, channels, 3, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride)
        if convs > 2:
            self.conv3 = nn.Conv2d(channels, channels, 3, stride=stride_third)

        for _ in range(2):
            image_size = int((image_size + 2 * 0 - 3) / stride + 1)
            self.logger.critical("After Conv: ({},{},{})".format(channels, image_size, image_size))

        if convs > 2:
            image_size = int((image_size + 2 * 0 - 3) / stride_third + 1)
            self.logger.critical("After Conv: ({},{},{})".format(channels, image_size, image_size))

        self.fc_size = image_size * image_size * channels
        self.fc_size_half = image_size * image_size * (channels // 2)
        self.logger.critical("FC {} -> {}".format(self.fc_size, self.fc_size_half))
        self.fc1 = nn.Linear(self.fc_size, self.fc_size_half)

        self.qvals = nn.Linear(self.fc_size_half, args.num_actions)
        self.logger.critical("FC {} -> {} (Q-Values)".format(self.fc_size_half, args.num_actions))

        if args.final_layer_bias > -1:
            optimistic_bias = torch.tensor([args.final_layer_bias for _ in range(args.num_actions)], dtype=torch.float)
            self.qvals.bias = nn.Parameter(optimistic_bias)
        # self.to(device)

    def forward(self, x):
        # x = x.to(device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.convs > 2:
            x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(-1, self.fc_size)

        x = F.relu(self.fc1(x))

        q = self.qvals(x)

        return q
