from sacred import Experiment
from sacred.observers import FileStorageObserver
import sys
from math import sqrt
from utils.dict2namedtuple import convert
import gym
from agent.specifier import get_model
from action import eps_greedy, optimistic_action, bsp_action
from action.testing import get_test_action
from buffer.buffer import ReplayBuffer
from trainer.dqn_train import DQNTrainer
from utils.logging import configure_stats_logging, get_stats, save_image, save_q_vals, save_actual_counts, save_sa_count_vals
from count.rnd_count import ApproxRndCount as PseudoCount
from count.rnd_network_count import RndNetworkDistill
from count.dora_count import DoraCount
from count.atari_count import AtariCount
import envs
from envs.env_wrapper import EnvWrapper
import logging
import torch
import yaml
from utils.timehelper import time_left, time_str
import time
import numpy as np

ex = Experiment("Count_Explore")

# Load config from file
ex.add_config("src/config/default.yaml")
with open("src/config/default.yaml", "r") as f:
    temp_config = yaml.load(f)
file_config = None
if "config_file=" in sys.argv:
    for s in sys.argv:
        if s.startswith("config_file="):
            config_file = s.split("=")[-1]
            ex.add_config("src/config/{}.yaml".format(config_file))
            with open("src/config/{}.yaml".format(config_file), "r") as f:
                file_config = yaml.load(f)
            break

if file_config is not None:
    temp_config.update(file_config)

entires_to_update = ["save"]
for s in sys.argv:
    for entry in entires_to_update:
        if s.startswith("{}=".format(entry)):
            temp_config[entry] = s.split("=")[-1]

client = None

def save_results(config_dict):
    save = config_dict.get("save", False)

    # Sorry I removed the mongodb stuff!

    # Saving results to disk
    if save:
        ex.observers.append(FileStorageObserver.create("results/sacred"))
        logging.critical("Saving sacred to file")


save_results(temp_config)

@ex.automain
def main(_config, _run):
    config = convert(_config)
    _id = _run._id

    # Logging stuff
    logger = logging.getLogger("Main")
    if config.mongo:
        logging.disable(logging.WARNING)
    configure_stats_logging(
        str(_id) + "_" + config.name,
        log_interval=config.log_interval,
        sacred_info=_run.info,
        use_tb=config.tb,
    )
    stats = get_stats()

    logger.critical("ID: {}".format(_id))
    # Update config with environment specific information
    env = gym.make(config.env)
    num_actions = env.action_space.n
    config = config._replace(num_actions=num_actions)
    state_shape = env.observation_space.shape
    config = config._replace(state_shape=state_shape)
    # Wrap env
    env = EnvWrapper(env, debug=True, args=config)

    # Log the config
    config_str = "Config:\n\n"
    for k, v in sorted(config._asdict().items()):
        config_str += "     {}: {}\n".format(k, v)
    logger.critical(config_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.critical("Device: {}".format(device.type))

    # Make agent and target agent
    agent = get_model(config.agent)(config)
    target_agent = get_model(config.agent)(config)
    target_agent.load_state_dict(agent.state_dict())
    agent.to(device)
    target_agent.to(device)

    # Pseudocount stuff
    count_model = None
    if config.count_rewards:
        if config.atari_count:
            count_model = AtariCount(config)
        elif config.rnd_net_count:
            # assert config.count_state_only_rewards
            count_model = RndNetworkDistill(config, device)
        elif config.dora_count:
            count_model = DoraCount(config, device)
        else:
            count_model = PseudoCount(config)

    # Make action selector
    action_selector = None
    if config.action_selector == "eps_greedy":
        action_selector = eps_greedy.EpsGreedy(config)
    elif config.action_selector == "optimistic_action":
        action_selector = optimistic_action.OptimisticAction(count_model, config)
    elif config.action_selector == "bsp":
        action_selector = bsp_action.BSPAction(config)
    else:
        raise Exception("{} is not an Action Selector!".format(config.action_selector))

    # Make replay buffer
    # Check if the obs dtype of the environment is an int
    obs_dtype = getattr(env.wrapped_env, "obs_dtype", np.float32)
    obs_scaling = getattr(env.wrapped_env, "obs_scaling", 1)
    replay_buffer = ReplayBuffer(
        size=config.buffer_size, frame_history_len=config.past_frames_input, obs_dtype=obs_dtype, obs_scaling=obs_scaling, args=config
    )

    if config.dora_count:
        dora_buffer = ReplayBuffer(
            size=config.batch_size * 4, frame_history_len=config.past_frames_input, obs_dtype=obs_dtype, obs_scaling=obs_scaling, args=config
        )

    # Make trainer
    trainer = None
    if config.trainer == "DQN":
        trainer = DQNTrainer(agent=agent, target_agent=target_agent, args=config, count_model=count_model, buffer=replay_buffer)
    else:
        raise Exception
    testing_buffer = ReplayBuffer(
        size=(config.past_frames_input + 1), frame_history_len=config.past_frames_input, args=config
    )

    # Testing stuff
    testing_env = EnvWrapper(env=gym.make(config.env), debug=True, args=config)
    if config.test_augmented:
        assert config.action_selector == "optimistic_action"

    # Player Positions
    positions = set()
    action_positions = set()

    T = 0
    start_time = time.time()
    last_time = start_time

    # Lots of code duplication :(
    logging.critical("Filling buffer with {:,} random experiences.".format(config.buffer_burn_in))
    state = env.reset()
    assert config.buffer_burn_in == 0
    for t in range(config.buffer_burn_in):
        buffer_idx = replay_buffer.store_frame(state)
        stacked_states = replay_buffer.encode_recent_observation()
        tensor_state = torch.tensor(stacked_states, device=device).unsqueeze(0)
        action = np.random.randint(config.num_actions)
        next_state, reward, terminated, info = env.step(action)
        terminal_to_store = terminated
        if "Steps_Termination" in info and info["Steps_Termination"]:
            terminal_to_store = False

        intrinsic_reward = 0
        pseudo_count = 0
        if config.count_rewards:
            pseudo_count = count_model.visit(tensor_state, action)
            if getattr(count_model, "reward_directly", False):
                intrinsic_reward = pseudo_count
            else:
                count_bonus = config.count_beta / sqrt(pseudo_count)
                intrinsic_reward = count_bonus

        replay_buffer.store_effect(buffer_idx, action, reward - config.reward_baseline, intrinsic_reward, terminal_to_store, pseudo_count)
        state = next_state
        if terminated:
            state = env.reset()
            logger.warning("Random action burn in t: {:,}".format(t))

    state = env.reset()
    episode = 0
    episode_reward = 0
    intrinsic_episode_reward = 0
    episode_length = 0
    env_positive_reward = 0
    max_episode_reward = 0
    if config.bsp:
        bsp_k = np.random.randint(config.bsp_k)
        action_selector.update_k(bsp_k)

    logging.critical("Beginning training.")

    while T < config.t_max:

        # Store the current state
        buffer_idx = replay_buffer.store_frame(state)
        if config.dora_count:
            dora_idx = dora_buffer.store_frame(state)

        # Get the stacked input vector
        stacked_states = replay_buffer.encode_recent_observation()

        # Get output from agent
        with torch.no_grad():
            tensor_state = torch.tensor(stacked_states, device=device).unsqueeze(0)
            agent_output = agent(tensor_state)
            # agent_output = agent(torch.Tensor(stacked_states).unsqueeze(0))

        # Select action
        action, action_info = action_selector.select_actions(
            agent_output, T, info={"state": tensor_state}
        )

        # Take an environment step
        next_state, reward, terminated, info = env.step(action)
        T += 1
        stats.update_t(T)
        episode_reward += reward
        episode_length += 1
        terminal_to_store = terminated
        if "Steps_Termination" in info and info["Steps_Termination"]:
            logger.warning("Terminating because of episode limit")
            terminal_to_store = False

        # Log if a positive reward was ever received from environment. ~Finding goal
        if reward > 0.1:
            env_positive_reward = 1
        stats.update_stats("Positive_Reward", env_positive_reward)

        # Calculate count based intrinsic motivation
        intrinsic_reward = 0
        pseudo_count = 0
        if config.count_rewards:
            pseudo_count = count_model.visit(tensor_state, action)
            if getattr(count_model, "reward_directly", False):
                # The count-model is giving us the intrinsic reward directly
                intrinsic_reward = pseudo_count[0]
            else:
                # Count-model is giving us the pseudo-count
                count_bonus = config.count_beta / sqrt(pseudo_count)
                intrinsic_reward = count_bonus
            intrinsic_episode_reward += intrinsic_reward

        # Render training
        if config.render_train_env:
            debug_info = {}
            debug_info.update(action_info)
            env.render(debug_info=debug_info)

        # Add what happened to the buffer
        replay_buffer.store_effect(buffer_idx, action, reward - config.reward_baseline, intrinsic_reward, terminal_to_store, pseudo_count)
        if config.dora_count:
            dora_buffer.store_effect(dora_idx, action, reward - config.reward_baseline, intrinsic_reward, terminal_to_store, pseudo_count)

        # Update state
        state = next_state

        # If terminated
        if terminated:
            # If we terminated due to episode limit, we need to add the current state in
            if "Steps_Termination" in info and info["Steps_Termination"]:
                buffer_idx = replay_buffer.store_frame(state)
                replay_buffer.store_effect(buffer_idx, 0, 0, 0, True, 0, dont_sample=True)
                if config.dora_count:
                    dora_idx = dora_buffer.store_frame(state)
                    dora_buffer.store_effect(dora_idx, 0, 0, 0, True, 0, dont_sample=True)

            logger.warning("T: {:,}, Episode Reward: {:.2f}".format(T, episode_reward))
            state = env.reset()
            max_episode_reward = max(max_episode_reward, episode_reward)
            stats.update_stats("Episode Reward", episode_reward)
            stats.update_stats("Max Episode Reward", max_episode_reward)
            stats.update_stats("Episode Length", episode_length)
            stats.update_stats("Intrin Eps Reward", intrinsic_episode_reward)
            episode_reward = 0
            episode_length = 0
            intrinsic_episode_reward = 0
            episode += 1
            stats.update_stats("Episode", episode)
            if config.bsp:
                bsp_k = np.random.randint(config.bsp_k)
                action_selector.update_k(bsp_k)

        # Train if possible
        for _ in range(config.training_iters):
            sampled_batch = None

            if T % config.update_freq != 0:
                # Only train every update_freq timesteps
                continue
            if replay_buffer.can_sample(config.batch_size):
                sampled_batch = replay_buffer.sample(config.batch_size, nstep=config.n_step)

            if sampled_batch is not None:
                trainer.train(sampled_batch)

            if config.dora_count:
                if dora_buffer.can_sample(config.batch_size):
                    sampled_batch = replay_buffer.sample(config.batch_size, nstep=config.n_step)
                if sampled_batch is not None:
                    count_model.train(sampled_batch)

        # Update target networks if necessary
        if T % config.target_update_interval == 0:
            trainer.update_target_agent()
            if config.dora_count:
                count_model.update_target_agent()

        # Logging
        if config.bsp:
            agent_output = agent_output[:,:,bsp_k]
        q_vals_numpy = agent_output.detach().cpu()[0].numpy()
        if num_actions < 20:
            for action_id in range(config.num_actions):
                stats.update_stats("Q-Value_{}".format(action_id), q_vals_numpy[action_id])
        else:
            stats.update_stats("Q-Value_Mean", np.mean(q_vals_numpy))
        player_pos = env.log_visitation()
        positions.add(player_pos)
        action_positions.add((player_pos, action))
        stats.update_stats("States Visited", len(positions))
        stats.update_stats("State_Actions Visited", len(action_positions))
        stats.update_stats("Player Position", player_pos)
        # Log all env stats returned
        for k, v in info.items():
            if k != "Steps_Termination":
                stats.update_stats(k, v)

        if config.save_count_gifs > 0 and T % config.save_count_gifs == 0:
            if count_model is not None:
                state_action_counts, count_nums = env.count_state_action_space(count_model)
                if state_action_counts is not None:
                    save_image(state_action_counts, image_name="SA_Counts__{}_Size__{}_T".format(config.count_size, T), direc_name="State_Action_Counts")
                    save_sa_count_vals(count_nums, name="SA_PCounts__{}_Size__{}_T".format(config.count_size, T), direc_name="Sa_Count_Estimates")

                actual_counts = env.state_counts()
                if actual_counts is not None:
                    save_actual_counts(actual_counts, name="Counts__{}_T".format(T), direc_name="Actual_Counts")

                q_val_img, q_vals = env.q_value_estimates(count_model, agent)
                if q_val_img is not None:
                    save_image(q_val_img, image_name="Q_Vals__{}_Size__{}_T".format(config.count_size, T), direc_name="Q_Value_Estimates")
                if q_vals is not None:
                    save_q_vals(q_vals, name="Q_Vals__{}_Size__{}_T".format(config.count_size, T), direc_name="Q_Value_Estimates")

        # Testing
        with torch.no_grad():
            if T % config.testing_interval == 0:

                prefixes = [""]
                if config.test_augmented:
                    prefixes += ["Aug_"]

                for prefix in prefixes:
                    total_test_reward = 0
                    total_test_length = 0
                    for _ in range(config.test_episodes):
                        test_episode_reward = 0
                        test_episode_length = 0
                        test_state = testing_env.reset()
                        test_env_terminated = False

                        while not test_env_terminated:
                            test_buffer_idx = testing_buffer.store_frame(test_state)
                            stacked_test_states = testing_buffer.encode_recent_observation()
                            test_tensor_state = torch.tensor(stacked_test_states, device=device).unsqueeze(0)
                            testing_agent_output = agent(test_tensor_state)

                            if prefix == "Aug_" or config.bsp:
                                test_action, _ = action_selector.select_actions(testing_agent_output, T, info={"state": test_tensor_state}, testing=True)
                            else:
                                test_action = get_test_action(testing_agent_output, config)

                            next_test_state, test_reward, test_env_terminated, _ = testing_env.step(test_action)
                            if config.render_test_env:
                                testing_env.render()

                            test_episode_length += 1
                            test_episode_reward += test_reward

                            testing_buffer.store_effect(
                                test_buffer_idx, test_action, test_reward, 0, test_env_terminated, 0
                            )

                            test_state = next_test_state

                        total_test_length += test_episode_length
                        total_test_reward += test_episode_reward

                    mean_test_reward = total_test_reward / config.test_episodes
                    mean_test_length = total_test_length / config.test_episodes

                    logger.error(
                        "{}Testing -- T: {:,}/{:,}, Test Reward: {:.2f}, Test Length: {:,}".format(
                            prefix,T, config.t_max, mean_test_reward, mean_test_length
                        )
                    )

                    stats.update_stats("{}Test Reward".format(prefix), mean_test_reward, always_log=True)
                    stats.update_stats(
                        "{}Test Episode Length".format(prefix), mean_test_length, always_log=True
                    )

                logger.error("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, T - config.testing_interval, T, config.t_max),
                    time_str(time.time() - start_time)))
                last_time = time.time()

        if T % (config.log_interval * 4) == 0:
            stats.print_stats()

    logger.critical("Closing envs")
    env.close()
    testing_env.close()

    logger.critical("Finished training.")

    if client is not None:
        logger.critical("Attempting to close pymongo client")
        client.close()
        logger.critical("Pymongo client closed")

    logger.critical("Exiting")