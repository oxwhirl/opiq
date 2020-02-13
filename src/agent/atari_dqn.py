# From https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb
import torch
import torch.nn as nn
import logging


class DQN(nn.Module):

    def __init__(self, args):
        super(DQN, self).__init__()

        self.logger = logging.getLogger("Atari_DQN")
        self.input_frames = args.past_frames_input

        self.input_shape = (args.past_frames_input, *args.state_shape[:2])
        self.num_actions = args.num_actions

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        if args.final_layer_bias > -1:
            optimistic_bias = torch.tensor([args.final_layer_bias for _ in range(args.num_actions)], dtype=torch.float)
            self.fc[-1].bias = nn.Parameter(optimistic_bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        q = self.fc(x)
        return q

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

