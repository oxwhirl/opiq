from .fc_net import Target_RND as target_fc
from .fc_net import Predictor_RND as pred_fc

from .conv_net import Target_RND as target_conv
from .conv_net import Predictor_RND as pred_conv

from .smaller_conv_net import Target_RND as smaller_target_conv
from .smaller_conv_net import Predictor_RND as smaller_pred_conv

from .bigger_conv_net import Target_RND as bigger_target_conv
from .bigger_conv_net import Predictor_RND as bigger_pred_conv

from .atari_conv_net import Target_RND as atari_target
from .atari_conv_net import Predictor_RND as atari_pred

def get_target(name):
    targets = {
        "fc": target_fc,
        "conv": target_conv,
        "smaller_pred_conv": smaller_target_conv,
        "bigger_pred_conv": bigger_target_conv,
        "atari": atari_target,
    }
    return targets[name]

def get_pred(name):
    preds = {
        "fc": pred_fc,
        "conv": pred_conv,
        "smaller_pred_conv": smaller_pred_conv,
        "bigger_pred_conv": bigger_pred_conv,
        "atari": atari_pred,
    }
    return preds[name]
