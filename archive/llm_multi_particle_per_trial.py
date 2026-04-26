from smc_sp import Engine
from environment import Environment
from llm.llm import LLM

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


# THIS config now really cuts in, because we pass it to smc_sp.Engine(...)
smc_config = {
    "num_particles": 1,       # particle number 
    "init_theta": (19, 1),     
    "ess_threshold": 0.5,     
    "act_mode": "sample",      
}

llm_config = {
    "model": "deepseek-chat",#qwen2.5-72b-instruct
    "temperature": 0.1,
    "max_tokens": 512,
}

max_trials = 70
num_run = 10
save_dir = r"C:\Users\MSN\Documents\Python\smc-s\training_results\nips_qwen\per_trial_deepseek32_1particle"


def save_history_to_csv(history, csv_path):
    if not history:
        print("No history to save.")
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

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
        print("\n" + "#" * 80)
        print(f"START OF RUN {run_idx + 1}/{NUM_RUNS}")
        print("#" * 80)

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
            row["model_name"] = llm_config["model"]
            row["num_particles"] = smc_config["num_particles"]
            row["init_theta_alpha"] = smc_config["init_theta"][0]
            row["init_theta_beta"] = smc_config["init_theta"][1]
            row["ess_threshold"] = smc_config["ess_threshold"]
            row["act_mode"] = smc_config["act_mode"]

        run_save_path = os.path.join(
            save_dir,
            f"{model_name}_particles_{smc_config['num_particles']}_run_{run_idx + 1:03d}_{timestamp}.csv"
        )

        save_history_to_csv(history, run_save_path)
        trials_per_run.append(len(history))

        final_opened = history[-1]["opened"] if history else 0

        print(f"END OF RUN {run_idx + 1}/{NUM_RUNS}")
        print(f"Trials recorded: {len(history)}")
        print(f"Boxes opened: {final_opened}")
        print(f"Saved CSV: {run_save_path}")

    save_trials_histogram(trials_per_run, save_dir, model_name, timestamp)