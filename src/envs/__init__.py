from gym.envs.registration import register

# Randomised Chain
register(
    id="{}Chain-v0".format(100),
    entry_point="envs.nchain:NChain",
    kwargs={"n": 100}
)

# Maze
register(
    id="Maze-v0",
    entry_point="envs.maze:SnakingMaze",
    kwargs={"size": 8, "corridor_width": 3, "neg_reward": False, "randomise": True, "num_actions": 4},
    tags={"wrapper_config.TimeLimit.max_episode_steps": 8 * 8 * 3 * 3 * 10 + 100},
)

# Montezuma
register(
    id="Montezuma-v0",
    entry_point="envs.montezuma:Montezuma",
    kwargs={},
)


