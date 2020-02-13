from .tiny_dqn import DQN as tiny_dqn
from .conv_dqn import DQN as conv_dqn
from .fc_dqn import DQN as fc_dqn
from .atari_dqn import DQN as atari_dqn
from .conv_dqn_bigger import DQN as conv_dqn_bigger
from .maze_conv import DQN as maze_dqn
from .fc_atari_dqn import DQN as fc_atari_dqn

from .bsp_maze_conv import BSPDQN as maze_bsp_dqn
from .bsp_atari_conv import BSPDQN as atari_bsp_dqn

names_to_models = {
    "tiny": tiny_dqn,
    "fc": fc_dqn,
    "fc_atari": fc_atari_dqn,
    "2conv": conv_dqn,
    "2convbigger": conv_dqn_bigger,
    "atari": atari_dqn,
    "maze": maze_dqn,
    "maze_bsp": maze_bsp_dqn,
    "atari_bsp": atari_bsp_dqn,
}


def get_model(name):
    return names_to_models[name]
