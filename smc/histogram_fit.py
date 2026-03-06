from smc_soc import Engine
from environment import Environment
from gen_soc import Generator

import os
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm


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

hist_config = {
    "num_runs": 100,
    "trials_per_run": 10,
    "show_plot": True,
    "save_plot": True,
    "save_dir": "training_results\histogram_method",
    "file_name": "histogram_100runs_10trials_30particle.png",
    "figsize": (8, 5),
}


def get_num_boxes_opened(history):
    if not history:
        return 0

    last = history[-1]

    if isinstance(last, dict) and "opened" in last:
        return last["opened"]

    raise ValueError(
        f"Could not infer number of opened boxes from history[-1]: {last}"
    )


def histogram_fit(
    gen_config: dict,
    smc_config: dict,
    hist_config: dict,
):
    """
    Run the model num_runs times.
    Each run lasts trials_per_run trials.
    Plot histogram of final number of boxes opened.
    """
    num_runs = hist_config["num_runs"]
    trials_per_run = hist_config["trials_per_run"]
    show_plot = hist_config.get("show_plot", True)
    save_plot = hist_config.get("save_plot", False)
    save_dir = hist_config.get("save_dir", "outputs")
    file_name = hist_config.get("file_name", "histogram.png")
    figsize = hist_config.get("figsize", (8, 5))

    opened_counts = []

    for run_idx in tqdm(range(num_runs), desc="Running histogram trials"):
        environment = Environment(include_inspect=False)
        generator = Generator(gen_config, environment)
        logger = Logger(logging=False)

        smc_engine = Engine(smc_config, environment, generator, logger)
        history = smc_engine.run(max_trials=trials_per_run)

        num_opened = get_num_boxes_opened(history)
        opened_counts.append(num_opened)

    freq = Counter(opened_counts)

    x_vals = sorted(freq.keys())
    y_vals = [freq[x] for x in x_vals]

    plt.figure(figsize=figsize)
    plt.bar(x_vals, y_vals, width=0.8)
    plt.xlabel(f"Number of boxes opened after {trials_per_run} trials")
    plt.ylabel("Frequency")
    plt.title(f"Histogram over {num_runs} runs")
    plt.xticks(x_vals)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = None
    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return opened_counts, freq, save_path


if __name__ == "__main__":
    opened_counts, freq, save_path = histogram_fit(
        gen_config=gen_config,
        smc_config=smc_config,
        hist_config=hist_config,
    )

    print("Opened box counts over runs:")
    print(opened_counts)

    print("Frequency table:")
    print(dict(sorted(freq.items())))

    if save_path is not None:
        print(f"Histogram saved to: {save_path}")