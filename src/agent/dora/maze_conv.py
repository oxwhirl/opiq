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
        channels = 16
        self.conv1 = nn.Conv2d(self.input_frames, channels, 3, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride)

        for _ in range(2):
            image_size = int((image_size + 2 * 0 - 3) / stride + 1)
            self.logger.critical("After Conv: ({},{},{})".format(channels, image_size, image_size))

        self.fc_size = image_size * image_size * channels
        assert self.fc_size == 400
        self.fc_size_half = image_size * image_size * (channels // 2)
        self.logger.critical("FC {} -> {}".format(self.fc_size, self.fc_size_half))
        self.fc1 = nn.Linear(self.fc_size, self.fc_size_half)

        self.qvals = nn.Linear(self.fc_size_half, args.num_actions)
        self.logger.critical("FC {} -> {} (Q-Values)".format(self.fc_size_half, args.num_actions))

        self.qvals.weight.data.fill_(0)

        # self.to(device)

    def forward(self, x):
        # x = x.to(device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.view(-1, self.fc_size)
        x = F.relu(self.fc1(x))
        q = self.qvals(x)

        return F.sigmoid(q)
