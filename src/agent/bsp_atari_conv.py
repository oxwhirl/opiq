import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BSPDQN(nn.Module):

    def __init__(self, args):
        super(BSPDQN, self).__init__()

        self.logger = logging.getLogger("Atari_BSP_DQN")
        self.input_frames = args.past_frames_input
        self.input_shape = (args.past_frames_input, *args.state_shape[:2])
        self.num_actions = args.num_actions

        maze_size = args.state_shape[0]

        image_size = maze_size
        self.logger.critical(
            "Atari image input: ({1},{0},{0})".format(image_size, self.input_frames)
        )
        self.bsp_k = args.bsp_k
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv_params = self.conv.parameters()
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )for _ in range(self.bsp_k)])

        self.prior_conv = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.prior_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )for _ in range(self.bsp_k)])
        self.prior_heads.requires_grad = False
        self.prior_beta = args.bsp_beta
        self.fc_size = self.feature_size()

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    # Scale the gradient of the shared conv layer by 1/K
    def scale_gradients(self):
        for c in self.conv_params:
            c.grad *= 1/self.bsp_k

    def forward(self, x):
        # x = x.to(device)
        c = self.conv(x)
        pc = self.prior_conv(x)

        c = c.view(-1, self.fc_size)
        pc = pc.view(-1, self.fc_size)

        qs = [self.heads[i](c) for i in range(self.bsp_k)]
        prior_qs = [self.prior_heads[i](pc) for i in range(self.bsp_k)]

        torch_qs = torch.stack(qs, dim=2)
        torch_prior_qs = torch.stack(prior_qs, dim=2)

        return torch_qs + self.prior_beta * torch_prior_qs
