import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Target_RND(nn.Module):

    def __init__(self, args):
        super(Target_RND, self).__init__()

        input_dim = args.state_shape[0]
        rep_size = args.rnd_rep_size

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, rep_size)
        )


    def forward(self, x):
        return self.model(x)


class Predictor_RND(nn.Module):

    def __init__(self, args):
        super(Predictor_RND, self).__init__()

        input_dim = args.state_shape[0]
        rep_size = args.rnd_rep_size

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, rep_size)
        )

    def forward(self, x):
        return self.model(x)
