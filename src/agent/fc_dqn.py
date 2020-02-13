import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, args):
        super(DQN, self).__init__()

        input_dim = args.state_shape[0]
        num_actions = args.num_actions

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, num_actions)
        )

        if args.final_layer_bias > -1:
            optimistic_bias = torch.tensor([args.final_layer_bias for _ in range(num_actions)], dtype=torch.float)
            self.model[-1].bias = nn.Parameter(optimistic_bias)


    def forward(self, x):
        # model_input = x.to(device)

        return self.model(x)
