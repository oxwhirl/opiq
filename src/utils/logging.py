import logging
from collections import defaultdict, deque
from tensorboard_logger import configure, log_value
import datetime
import numpy as np
import os
from PIL import Image


__all__ = ["configure_stats_logging", "get_stats"]


class Stats:

    def __init__(self, log_interval, sacred_info, use_tb):
        self.T = 0
        self.logger = logging.getLogger("stats")

        # Stats
        self.stats = defaultdict(lambda: deque(maxlen=20))
        self.stats_ts = defaultdict(lambda: deque(maxlen=20))

        self.log_t = defaultdict(lambda: 0)
        self.log_interval = log_interval
        self.logger.error("Logging every {} steps".format(log_interval))

        # TB
        self.tb = use_tb

        # Sacred
        self.sacred_info = sacred_info

        # Fill in defaults for Episode
        self.stats["Episode"].append(0)
        self.stats["Episode_T"].append(0)

    def update_stats(self, key, value, always_log=False):

        if value is None:
            return

        if (not always_log and self.T - self.log_t[key] >= self.log_interval) or always_log:
            # Tb
            if self.tb:
                if not type(value) is tuple:
                    log_value(name=key, value=value, step=self.T)

            # Sacred
            if key in self.sacred_info:
                self.sacred_info[key+"_T"].append(self.T)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info[key] = [value]
                self.sacred_info[key+"_T"] = [self.T]

            self.log_t[key] = self.T

        self.stats[key].append(value)
        self.stats_ts[key].append(self.T)

    def update_t(self, T):
        self.T = T

    def print_stats(self):
        log_str = "Recent Stats | T: {:,} | Episode: {:,}\n".format(self.T, self.stats["Episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "Episode" or k == "Player Position":
                continue
            i += 1
            # window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(np.mean([x for x in self.stats[k]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.logger.error(log_str)



stats = None
run_results_name = ""


def configure_stats_logging(name, log_interval, sacred_info, use_tb) -> None:
    global stats
    stats = Stats(log_interval=log_interval, sacred_info=sacred_info, use_tb=use_tb)
    date = "{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    global run_results_name
    run_results_name = "{}__{}".format(name, date)
    if use_tb:
        configure("results/tb/{}".format(run_results_name), flush_secs=10)


def save_image(image, image_name, direc_name="all",):
    global run_results_name
    global stats
    stats.logger.critical("Saving state-action counts: {} in directory {}".format(image_name, direc_name))
    results_directory = "results/images/{}/{}".format(run_results_name, direc_name)
    os.makedirs(results_directory, exist_ok=True)

    img = Image.fromarray(image.astype(np.uint8))
    img.save("{}/{}.png".format(results_directory, image_name))


def save_q_vals(q_vals, name, direc_name="all"):
    global run_results_name
    global stats
    stats.logger.critical("Saving Q-Values: {} in directory {}".format(name, direc_name))
    results_directory = "results/q_values/{}/{}".format(run_results_name, direc_name)
    os.makedirs(results_directory, exist_ok=True)

    np.savetxt("{}/{}.txt".format(results_directory, name), q_vals[:, :, 0])


def save_actual_counts(counts, name, direc_name="all"):
    global run_results_name
    global stats
    stats.logger.critical("Saving Counts: {} in directory {}".format(name, direc_name))
    results_directory = "results/actual_counts/{}/{}".format(run_results_name, direc_name)
    os.makedirs(results_directory, exist_ok=True)

    np.savetxt("{}/{}.txt".format(results_directory, name), counts)


def save_sa_count_vals(c_vals, name, direc_name="all"):
    global run_results_name
    global stats
    stats.logger.critical("Saving SA PCounts: {} in directory {}".format(name, direc_name))
    results_directory = "results/sa_count_p_values/{}/{}".format(run_results_name, direc_name)
    os.makedirs(results_directory, exist_ok=True)

    np.savetxt("{}/{}.txt".format(results_directory, name), c_vals)


def get_stats() -> Stats:
    global stats
    return stats
