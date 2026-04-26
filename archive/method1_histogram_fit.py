from smc_soc import Engine
from environment import Environment
from gen_soc import Generator

import os
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from datetime import datetime


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
    "trials_per_run": 100,
    "show_plot": False,
    "save_plot": False,
    "save_dir": r"training_results\histogram_method",
    "file_name": "histogram_attempts_to_open_all_100runs_10trials_30particle.png",
    "figsize": (8, 5),
    "target_opened": 5,          # number of boxes that means task completed
    "only_successful_runs": True # only plot runs that opened all 5
}


class HistogramFitter:
    """
    Fit:
        Run the BED / SMC model multiple times and record
        the number of attempts needed to open all boxes.

    Plot:
        Plot histogram of attempts-to-open-all-boxes.
    """
    @staticmethod
    def make_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def __init__(self, gen_config: dict, smc_config: dict, hist_config: dict):
        self.gen_config = gen_config
        self.smc_config = smc_config
        self.hist_config = hist_config

        self.attempt_counts = []
        self.failed_runs = 0
        self.freq = None
        self.save_path = None

    @staticmethod
    def get_attempt_to_open_all(history, target_opened: int = 5):
        """
        Return the 1-based trial number at which all boxes were first opened.
        If never reached, return None.

        Assumes each history item is a dict and may contain key 'opened'.
        """
        if not history:
            return None

        for trial_idx, step in enumerate(history, start=1):
            if isinstance(step, dict) and step.get("opened", None) == target_opened:
                return trial_idx

        return None

    def fit(self):
        """
        Run the model num_runs times and collect the number of attempts
        needed to open all boxes.

        Returns:
            attempt_counts: list[int]
                attempts needed for successful runs only if only_successful_runs=True,
                otherwise includes all successful runs and leaves failed runs counted separately
            freq: Counter
            failed_runs: int
        """
        num_runs = self.hist_config["num_runs"]
        trials_per_run = self.hist_config["trials_per_run"]
        target_opened = self.hist_config.get("target_opened", 5)
        only_successful_runs = self.hist_config.get("only_successful_runs", True)

        self.attempt_counts = []
        self.failed_runs = 0

        for _ in tqdm(range(num_runs), desc="Running histogram trials"):
            environment = Environment(include_inspect=False)
            generator = Generator(self.gen_config, environment)
            logger = Logger(logging=False)

            smc_engine = Engine(self.smc_config, environment, generator, logger)
            history = smc_engine.run(max_trials=trials_per_run)

            attempt_to_finish = self.get_attempt_to_open_all(
                history, target_opened=target_opened
            )

            if attempt_to_finish is None:
                self.failed_runs += 1
                if not only_successful_runs:
                    # If you later want to include failures in some other way,
                    # you can store trials_per_run + 1 or another marker here.
                    pass
            else:
                self.attempt_counts.append(attempt_to_finish)

        self.freq = Counter(self.attempt_counts)
        return self.attempt_counts, self.freq, self.failed_runs

    def plot(self):
        """
        Plot histogram from fitted attempt counts.
        """
        if self.freq is None:
            raise ValueError("Call fit() before plot().")

        show_plot = self.hist_config.get("show_plot", True)
        save_plot = self.hist_config.get("save_plot", False)
        save_dir = self.hist_config.get("save_dir", "outputs")
        file_name = self.hist_config.get("file_name", "histogram.png")
        figsize = self.hist_config.get("figsize", (8, 5))
        num_runs = self.hist_config["num_runs"]
        trials_per_run = self.hist_config["trials_per_run"]
        only_successful_runs = self.hist_config.get("only_successful_runs", True)

        x_vals = sorted(self.freq.keys())
        y_vals = [self.freq[x] for x in x_vals]

        plt.figure(figsize=figsize)
        plt.bar(x_vals, y_vals, width=0.8)

        plt.xlabel("Number of attempts needed to open all 5 boxes")
        plt.ylabel("Frequency")

        if only_successful_runs:
            plt.title(
                f"Histogram over successful model runs only\n"
                f"({len(self.attempt_counts)}/{num_runs} runs opened all 5 within {trials_per_run} trials)"
            )
        else:
            plt.title(f"Histogram over {num_runs} runs")

        plt.xticks(x_vals if x_vals else range(1, trials_per_run + 1))
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        self.save_path = None
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            self.save_path = os.path.join(save_dir, file_name)
            plt.savefig(self.save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return self.save_path


if __name__ == "__main__":
    fitter = HistogramFitter(
        gen_config=gen_config,
        smc_config=smc_config,
        hist_config=hist_config,
    )

    attempt_counts, freq, failed_runs = fitter.fit()
    save_path = fitter.plot()

    print("Attempts needed to open all 5 boxes:")
    print(attempt_counts)

    print("Frequency table:")
    print(dict(sorted(freq.items())))

    print(f"Failed runs (did not open all 5 within limit): {failed_runs}")

    if save_path is not None:
        print(f"Histogram saved to: {save_path}")