import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, args):
        super(DQN, self).__init__()

        input_dim = args.state_shape[0]
        num_actions = args.num_actions

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.Tanh(), nn.Linear(32, num_actions)
        )
        # self.model.to(device)

    def forward(self, x):
        # model_input = x.to(device)

        return self.model(x)
