import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smc.smc_soc import Engine
from smc.environment import Environment
from smc.gen_soc import Generator
from smc.utils.plotter import Plotter2

import matplotlib.pyplot as plt
from collections import Counter


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging
    
    def log(self, log_str: str):
        if self.logging:
            print(log_str)


gen_config = {
    "omega": 2.0,
    "prop_random": 0.1,
    "train": False,
}

smc_config = {
    "num_particles": 30,
    "init_theta": (0.5, 0.5),
    "ess_threshold": 0.5,
    "skill": True,
    "mode": "soc",
    "prior": "uniform",
}

max_trials = 70

hist_config = {
    "num_runs": 100,
    "trials_per_run": 15,
    "show_plot": True,
}


def get_num_boxes_opened(history):
    """
    Try to extract the final number of opened boxes from engine history.
    Adjust this if your history format is different.
    """

    if not history:
        return 0

    last = history[-1]

    # Case 1: history item stores opened-box count directly
    if isinstance(last, dict):
        for key in ["num_boxes_opened", "boxes_opened", "opened_boxes"]:
            if key in last:
                value = last[key]
                if isinstance(value, int):
                    return value
                if isinstance(value, (list, set, tuple)):
                    return len(value)

        # Case 2: history item stores box states in a dict/list
        if "box_states" in last:
            box_states = last["box_states"]
            if isinstance(box_states, dict):
                return sum(1 for v in box_states.values() if v)
            if isinstance(box_states, list):
                return sum(1 for v in box_states if v)

    # Fallback:
    # if your history format is different, print the last entry once and inspect it
    raise ValueError(
        "Could not infer number of opened boxes from history. "
        "Please inspect history[-1] and adjust get_num_boxes_opened()."
    )


def histogram_fit(
    num_runs: int = 100,
    trials_per_run: int = 15,
    show_plot: bool = True,
):
    """
    Run the model num_runs times, each time for trials_per_run trials.
    Plot a histogram of how many boxes were opened by the end of each run.
    """

    opened_counts = []

    for run_idx in range(num_runs):
        environment = Environment(include_inspect=False)
        generator = Generator(gen_config, environment)

        # turn logging off for batch runs
        logger = Logger(logging=False)

        smc_engine = Engine(smc_config, environment, generator, logger)
        history = smc_engine.run(max_trials=trials_per_run)

        num_opened = get_num_boxes_opened(history)
        opened_counts.append(num_opened)

    freq = Counter(opened_counts)

    plt.figure(figsize=(8, 5))
    plt.bar(freq.keys(), freq.values(), width=0.8)
    plt.xlabel("Number of boxes opened")
    plt.ylabel("Frequency")
    plt.title(f"Histogram over {num_runs} runs ({trials_per_run} trials each)")
    plt.xticks(sorted(freq.keys()))
    plt.grid(axis="y", alpha=0.3)

    if show_plot:
        plt.show()

    return opened_counts, freq


if __name__ == '__main__':

    # normal single run
    environment = Environment(include_inspect=False)
    generator = Generator(gen_config, environment)
    logger = Logger(logging=True)

    smc_engine = Engine(smc_config, environment, generator, logger)
    history = smc_engine.run(max_trials=max_trials)

    plotter2 = Plotter2(history)
    plotter2.plot_boxes_opened_over_trials(show=True)
    plotter2.plot_hypothesis_probs_over_trials(show=True)
    plotter2.plot_theta_over_trials(show=True)

    # histogram fit
    histogram_fit(
        num_runs=hist_config["num_runs"],
        trials_per_run=hist_config["trials_per_run"],
        show_plot=hist_config["show_plot"],
    )