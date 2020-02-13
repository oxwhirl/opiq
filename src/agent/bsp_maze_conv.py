import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BSPDQN(nn.Module):

    def __init__(self, args):
        super(BSPDQN, self).__init__()

        self.logger = logging.getLogger("Maze_BSP_DQN")
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

        self.fc_size = 400

        self.bsp_k = args.bsp_k
        self.conv = nn.Sequential(nn.Conv2d(self.input_frames, channels, 3, stride=stride),
                                        nn.ReLU(),
                                        nn.Conv2d(channels, channels, 3, stride=stride),
                                        nn.ReLU())
        self.conv_params = self.conv.parameters()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(400, 200), nn.ReLU(), nn.Linear(200, args.num_actions)) for _ in range(self.bsp_k)])

        self.prior_conv = nn.Sequential(nn.Conv2d(self.input_frames, channels, 3, stride=stride),
                                        nn.ReLU(),
                                        nn.Conv2d(channels, channels, 3, stride=stride),
                                        nn.ReLU())
        self.prior_heads = nn.ModuleList([nn.Sequential(nn.Linear(400, 200), nn.ReLU(), nn.Linear(200, args.num_actions)) for _ in range(self.bsp_k)])
        self.prior_conv.requires_grad = False
        self.prior_heads.requires_grad = False

        self.prior_beta = args.bsp_beta

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
