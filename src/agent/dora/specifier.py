from .maze_conv import DQN as maze_conv

def get_net(name):
    nets = {
        "maze": maze_conv,
    }
    return nets[name]

