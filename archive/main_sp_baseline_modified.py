from llm_ps import LlmPS
from environment import Environment
from llm.llm import LLM

import csv
import os
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
from types import MethodType


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        if self.logging:
            print(log_str)

smc_config = {
    "num_particles": 1,
    "init_theta": (19, 1),
    "ess_threshold": 0.5,
    "act_mode": "sample",
}
llm_config = {
    "model": "deepseek-chat",#"qwen2.5-72b-instruct"
    "temperature": 0.1,
    "max_tokens": 512,
}

patch_config = {
    "max_refine_fails_per_trial": 3,
}

max_trials = 70
num_run = 1
save_dir = r"training_results\LLM_baseline_results"


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


def patch_spbaseline_with_early_stop(
    baseline: SPBaseline,
    *,
    max_refine_fails_per_trial: int = 3,
):
    def _fresh_llm(self):
        return LLM(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def run_per_trial_llm(self, max_trials: int) -> dict:
        self.trial_count = 0
        self.history = []
        self._early_stop = False
        self._early_stop_reason = None

        while (
            not self.env.is_solved()
            and self.trial_count < max_trials
            and not self._early_stop
        ):
            print("\n" + "=" * 70)
            print(f"START OF NEW TRIAL: {self.trial_count}")
            print("=" * 70)

            self.logger.log(f"TRIAL {self.trial_count}")

            self.llm = self._fresh_llm()

            llm_mode = "generate" if self.trial_count == 0 else "refine"
            accepted = True
            refine_attempts = 0
            early_stop_triggered = False

            if self.trial_count == 0:
                self.hypothesis, h_name = self.llm.generate([])
            else:
                accepted = False

                while True:
                    self.hypothesis, new_name = self.llm.refine(self.evidence, self.hypothesis)
                    refine_attempts += 1

                    if self._accept_h():
                        accepted = True
                        break

                    if refine_attempts > self.max_refine_fails_per_trial:
                        self.logger.log(
                            f"[EARLY STOP] refine failed more than "
                            f"{self.max_refine_fails_per_trial} times in this trial"
                        )
                        self._early_stop = True
                        self._early_stop_reason = "too_many_refine_failures"
                        early_stop_triggered = True
                        break

                if self._early_stop:
                    self.history.append({
                        "t": self.trial_count,
                        "opened": len(self.env.success_pairs),
                        "hypothesis": self.hypothesis,
                        "action": None,
                        "outcome": None,
                        "llm_mode": llm_mode,
                        "fresh_llm_instance": True,
                        "refine_attempts": refine_attempts,
                        "accepted_after_refine": accepted,
                        "early_stop_triggered": early_stop_triggered,
                        "early_stop_reason": self._early_stop_reason,
                    })

                    print(f"END OF TRIAL {self.trial_count}: EARLY STOP")
                    print(f"Opened boxes so far: {len(self.env.success_pairs)}")
                    break

            self.logger.log(f"{self.hypothesis}")

            key, box = self._select_action()
            outcome = self.env.test_action(key, box)

            self.evidence.append((key, box, outcome))

            self.logger.log(f"Action chosen: ({key.id}, {box.id})")
            self.logger.log(f"Outcome: {outcome}")
            self.logger.log(f"Boxes opened: {self.env.success_pairs}")

            self.trial_count += 1

            self.history.append({
                "t": self.trial_count,
                "opened": len(self.env.success_pairs),
                "hypothesis": self.hypothesis,
                "action": [key.id, box.id],
                "outcome": bool(outcome),
                "llm_mode": llm_mode,
                "fresh_llm_instance": True,
                "refine_attempts": refine_attempts,
                "accepted_after_refine": accepted,
                "early_stop_triggered": early_stop_triggered,
                "early_stop_reason": self._early_stop_reason,
            })

            print(f"END OF TRIAL {self.trial_count}")
            print(f"Chosen action: ({key.id}, {box.id})")
            print(f"Outcome: {outcome}")
            print(f"Opened boxes so far: {len(self.env.success_pairs)}")

        return {
            "solved": self.env.is_solved() and not self._early_stop,
            "trials": self.trial_count,
            "opened": len(self.env.success_pairs),
            "success_pairs": list(self.env.success_pairs),
            "history": self.history,
            "early_stop": self._early_stop,
            "early_stop_reason": self._early_stop_reason,
        }

    baseline.model_name = baseline.llm.model
    baseline.temperature = baseline.llm.temperature
    baseline.max_tokens = baseline.llm.max_tokens

    baseline.max_refine_fails_per_trial = max_refine_fails_per_trial

    baseline._fresh_llm = MethodType(_fresh_llm, baseline)
    baseline.run = MethodType(run_per_trial_llm, baseline)
    return baseline


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

        baseline = LlmPS(
            env=environment,
            llm=llm,
            logger=logger,
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
        )

        result = baseline.run(max_trials=max_trials)

        # baseline = patch_spbaseline_with_early_stop(
        #     baseline,
        #     max_refine_fails_per_trial=patch_config["max_refine_fails_per_trial"],
        # )

        result = baseline.run(max_trials=max_trials)
        history = result["history"]

        for row in history:
            row["run_number"] = run_idx + 1
            row["solved"] = result["solved"]
            row["final_opened"] = result["opened"]
            row["run_early_stop"] = result["early_stop"]
            row["run_early_stop_reason"] = result["early_stop_reason"]

        run_save_path = os.path.join(
            save_dir,
            f"{model_name}_per_trial_run_{run_idx + 1:03d}_{timestamp}.csv"
        )

        save_history_to_csv(history, run_save_path)
        trials_per_run.append(result["trials"])

        print(f"END OF RUN {run_idx + 1}/{NUM_RUNS}")
        print(f"Solved: {result['solved']}")
        print(f"Trials recorded: {result['trials']}")
        print(f"Boxes opened: {result['opened']}")
        print(f"Early stop: {result['early_stop']}")
        print(f"Saved CSV: {run_save_path}")

    save_trials_histogram(trials_per_run, save_dir, model_name, timestamp)