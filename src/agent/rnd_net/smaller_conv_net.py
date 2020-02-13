import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Target_RND(nn.Module):

    def __init__(self, args):
        super(Target_RND, self).__init__()
        maze_size = args.state_shape[0]

        image_size = maze_size
        stride = 2
        channels = 16
        self.conv1 = nn.Conv2d(1, channels, 3, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride)

        for _ in range(2):
            image_size = int((image_size + 2 * 0 - 3) / stride + 1)

        self.fc_size = image_size * image_size * channels
        assert self.fc_size == 400
        self.fc_size_half = image_size * image_size * (channels // 2)
        self.fc1 = nn.Linear(self.fc_size, self.fc_size_half)

        self.out = nn.Linear(self.fc_size_half, args.rnd_rep_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten
        x = x.view(-1, self.fc_size)
        x = F.relu(self.fc1(x))
        out = self.out(x)
        return out


class Predictor_RND(nn.Module):

    def __init__(self, args):
        super(Predictor_RND, self).__init__()

        # This predictor is intentionally smaller/differ than the target network

        maze_size = args.state_shape[0]

        image_size = maze_size
        stride = 2
        channels = 8
        self.conv1 = nn.Conv2d(1, channels, 3, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride)

        for _ in range(2):
            image_size = int((image_size + 2 * 0 - 3) / stride + 1)

        self.fc_size = image_size * image_size * channels
        assert self.fc_size == 200
        self.fc_size_half = image_size * image_size * (channels // 2)
        self.fc1 = nn.Linear(self.fc_size, self.fc_size_half)

        self.out = nn.Linear(self.fc_size_half, args.rnd_rep_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten
        x = x.view(-1, self.fc_size)
        x = F.relu(self.fc1(x))
        out = self.out(x)
        return out
