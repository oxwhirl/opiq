# This repo contains the code for Optimistic Exploration even with a Pessimistic Initialisation.

The [paper](https://openreview.net/forum?id=r1xGP6VYwH) is located here. 

To run experiments, first build the docker container by running `build.sh` from inside the docker folder.

The following launches a docker container on GPU $GPU_ID (set to 0 if you want to run on the first gpu) with the parameters specified (shown as ...):

`bash run.sh $GPU_ID python3 src/main.py with ... repeat_id=0`

To run OPIQ on the Maze:

`bash run.sh 0 python3 src/main.py with t_max=1050000 env=Maze-v0 agent=2conv batch_size=64 epsilon_time_length=50000 epsilon_finish=0.01 test_episodes=1 testing_interval=20000 log_interval=10000 count_rewards=True count_beta=0.1 count_size=128 recompute_count_rewards=True n_step=3 buffer_size=250000 double_q=False count_state_only_rewards=False save=True reward_clipping=False target_update_interval=1000 gamma=0.99 episode_limit=250 name=opiq action_selector=optimistic_action optim_m=2 optim_action_tau=100 optim_bootstrap=True optim_bootstrap_tau=0.01  label=maze repeat_id=1`

Montezuma:

`bash run.sh 0 python3 src/main.py with t_max=13050000 env=Montezuma-v0 past_frames_input=4 agent=atari batch_size=32 update_freq=4 epsilon_time_length=1000000 epsilon_finish=0.01 test_epsilon=0.01 test_episodes=5 testing_interval=100000 log_interval=50000 count_rewards=True count_beta=0.01 atari_count=True recompute_count_rewards=True n_step=3 buffer_size=1000000 double_q=False count_state_only_rewards=False save=True reward_clipping=True target_update_interval=8000 gamma=0.99 mmc=True name=OPIQ action_selector=optimistic_action optim_m=2 optim_action_tau=0.1 mmc_beta=0.01 optim_bootstrap=True optim_bootstrap_tau=0.01  label=montezuma_opiq`

The config file `default.yaml` in src/config contains all the possible flags/hyperparameters.
Sorry about the lack of documentation and general messiness of the code, if you have any questions don't hesitate to raise an issue or send me an email: [tabish.rashid@cs.ox.ac.uk](mailto:tabish.rashid@cs.ox.ac.uk).

Bibtex:
```
@inproceedings{rashid2019optimistic,
  title={Optimistic Exploration even with a Pessimistic Initialisation},
  author={Rashid, Tabish and Peng, Bei and Boehmer, Wendelin and Whiteson, Shimon},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```