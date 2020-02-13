__all__ = ["get_test_action"]
import numpy as np

def get_test_action(agent_q_values, args):
    action = None
    if np.random.random() < args.test_epsilon:
        action = np.random.randint(args.num_actions)
    else:
        action = agent_q_values.argmax().cpu().item()
    return action
