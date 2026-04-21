from smc_sp_vs2 import Engine
from llm.llm_vs2 import LLM
from environment import Environment

import csv
import os
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        if self.logging:
            print(log_str)


# LLM has zero awareness of SMC config
smc_config = {
    "num_particles": 1,
    "init_theta": (19, 1),
    "ess_threshold": 0.5,
    "act_mode": "sample",
}

llm_config = {
    "model": "qwen-plus",      # or "deepseek-chat"
    "temperature": 0.1,
    "max_tokens": 512,
}

max_trials = 70
num_run = 98
save_dir = r"C:\Users\MSN\Documents\Python\smc-s\training_results\nips_qwen\the_way_one_run"


def save_history_to_csv(history, csv_path):
    if not history:
        print("No history to save.")
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # union of all keys across rows
    fieldnames = []
    seen = set()
    for row in history:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)

    print(f"Saved history to: {csv_path}")


def save_trials_histogram(trials_per_run, save_dir, model_name, timestamp):
    if not trials_per_run:
        print("No trials data to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(
        save_dir,
        f"{model_name}_summary_{timestamp}_trials_to_open_5_boxes_histogram.png"
    )

    counts = Counter(trials_per_run)
    x = sorted(counts.keys())
    y = [counts[v] for v in x]

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.xlabel("Trial number to open 5 boxes")
    plt.ylabel("Total number of runs")
    plt.title("Histogram of trials needed to open 5 boxes across runs")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved histogram to: {plot_path}")


if __name__ == '__main__':
    NUM_RUNS = num_run
    os.makedirs(save_dir, exist_ok=True)

    trials_per_run = []
    model_name = llm_config["model"].replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for run_idx in range(NUM_RUNS):
        print(f"\n===== RUN {run_idx + 1} =====")

        environment = Environment(include_inspect=False)
        logger = Logger(logging=(run_idx == 0))

        llm = LLM(
            model=llm_config["model"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
        )

        smc_engine = Engine(smc_config, environment, llm, logger)
        history = smc_engine.run(max_trials=max_trials)

        for row in history:
            row["run_number"] = run_idx + 1

        run_save_path = os.path.join(
            save_dir,
            f"{model_name}_run_{run_idx + 1:03d}_{timestamp}.csv"
        )

        save_history_to_csv(history, run_save_path)
        trials_per_run.append(len(history))

    save_trials_histogram(trials_per_run, save_dir, model_name, timestamp)