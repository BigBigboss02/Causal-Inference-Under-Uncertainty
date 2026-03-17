import os
import copy
import itertools
import json
from datetime import datetime
from collections import Counter

import matplotlib.pyplot as plt
from tqdm import tqdm

from smc_soc import Engine
from environment import Environment
from gen_soc import Generator


# =========================
# Base configs
# =========================

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

prob_config = {
    "num_runs": 1000,
    "trials_per_run": 70,
    "show_plot": False,
    "save_plot": True,
    "save_dir": r"training_results\trial_probability_method",
    "file_name": "trial_probability.png",
    "figsize": (10, 6),
    "max_boxes": 5,
    "bar_plot_trial": 9,   # optional single-trial bar plot
}


THETA_LIST = [
    (9, 1),
    (19, 1),
    (8, 1),
    (7, 1),
    (6, 1),
]

GENERATOR_PRIOR_LIST = [0.01, 0.1, 0.3, 0.5, 0.8]
TRUE_RULE_PRIOR_LIST = [0.01, 0.1, 0.2]
# THETA_LIST = [
#     (9, 1),
# ]

# GENERATOR_PRIOR_LIST = [0.01]
# TRUE_RULE_PRIOR_LIST = [0.01]
SAVING_DIR = r"training_results\trial_probability_method\pertrialprobability_experiments_batch_1000_run_11032026"


# =========================
# Logger
# =========================

class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        if self.logging:
            print(log_str)


# =========================
# Method 2 fitter
# =========================

class TrialProbabilityFitter:
    """
    Method 2:
    For each model run, compute the cumulative number of unique boxes opened
    by each trial:
        K_n in {0,1,2,3,4,5}

    Across many runs, estimate:
        P(K_n = k)
    for n = 1..trials_per_run and k = 0..max_boxes
    by simple empirical frequency.
    """

    @staticmethod
    def make_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def __init__(self, gen_config: dict, smc_config: dict, prob_config: dict):
        self.gen_config = gen_config
        self.smc_config = smc_config
        self.prob_config = prob_config

        self.trajectories = []
        self.count_table = None
        self.prob_table = None
        self.save_path = None
        self.bar_save_path = None

    @staticmethod
    def pad_trajectory_with_final_value(trajectory, max_trial):
        """
        If a run stops before max_trial, keep its final opened-count constant.

        This preserves the monotone cumulative interpretation:
            once a run has opened k boxes, it should not disappear later.
        """
        if len(trajectory) == 0:
            return [0] * max_trial

        if len(trajectory) >= max_trial:
            return trajectory[:max_trial]

        final_value = trajectory[-1]
        return trajectory + [final_value] * (max_trial - len(trajectory))

    @staticmethod
    def model_history_to_opened_trajectory(history, max_trial=70, max_boxes=5):
        """
        Convert one Engine.run(...) history into a monotone cumulative trajectory:
            trajectory[n-1] = number of unique boxes opened by trial n

        Preferred case:
            each history step already contains step["opened"]

        Fallback case:
            reconstruct from step["box"] or step["box_id"] and step["outcome"]

        Adjust this function if your actual Engine.run() output uses different keys.
        """
        if not history:
            return [0] * max_trial

        # Case 1: history already stores cumulative opened count
        first_step = history[0]
        if isinstance(first_step, dict) and "opened" in first_step:
            trajectory = []
            for step in history:
                opened = int(step.get("opened", 0))
                opened = max(0, min(opened, max_boxes))
                trajectory.append(opened)

            return TrialProbabilityFitter.pad_trajectory_with_final_value(
                trajectory, max_trial=max_trial
            )

        # Case 2: reconstruct from chosen box + outcome
        opened_boxes = set()
        trajectory = []

        for step in history:
            if not isinstance(step, dict):
                raise ValueError(
                    "Unexpected history step format. "
                    "Expected dict steps from Engine.run()."
                )

            # Try common key names
            box_id = None
            if "box" in step:
                box_id = step["box"]
            elif "box_id" in step:
                box_id = step["box_id"]

            outcome = step.get("outcome", None)

            # If this trial successfully opened a box, add it to the opened set
            if outcome == 1 and box_id is not None:
                opened_boxes.add(box_id)

            trajectory.append(min(len(opened_boxes), max_boxes))

        return TrialProbabilityFitter.pad_trajectory_with_final_value(
            trajectory, max_trial=max_trial
        )

    def fit(self):
        """
        Run the model num_runs times and estimate:
            count_table[n][k]
            prob_table[n][k]
        """
        num_runs = self.prob_config["num_runs"]
        trials_per_run = self.prob_config["trials_per_run"]
        max_boxes = self.prob_config.get("max_boxes", 5)

        self.trajectories = []

        for _ in tqdm(range(num_runs), desc="Running trial-probability simulations"):
            # Keep the exact Method 1 call pattern
            environment = Environment(include_inspect=False)
            generator = Generator(self.gen_config, environment)
            logger = Logger(logging=False)

            smc_engine = Engine(self.smc_config, environment, generator, logger)
            history = smc_engine.run(max_trials=trials_per_run)

            traj = self.model_history_to_opened_trajectory(
                history=history,
                max_trial=trials_per_run,
                max_boxes=max_boxes,
            )
            self.trajectories.append(traj)

        self.count_table = {}
        self.prob_table = {}

        for trial_n in range(1, trials_per_run + 1):
            opened_counts = [traj[trial_n - 1] for traj in self.trajectories]
            freq = Counter(opened_counts)

            self.count_table[trial_n] = {
                k: freq.get(k, 0)
                for k in range(max_boxes + 1)
            }

            self.prob_table[trial_n] = {
                k: freq.get(k, 0) / num_runs if num_runs > 0 else 0.0
                for k in range(max_boxes + 1)
            }

        return self.trajectories, self.count_table, self.prob_table

    def plot_probability_lines(self):
        """
        Plot P(K_n = k), k=0..max_boxes, over trials n=1..trials_per_run.
        """
        if self.prob_table is None:
            raise ValueError("Call fit() before plot_probability_lines().")

        show_plot = self.prob_config.get("show_plot", True)
        save_plot = self.prob_config.get("save_plot", False)
        save_dir = self.prob_config.get("save_dir", "outputs")
        file_name = self.prob_config.get("file_name", "trial_probability.png")
        figsize = self.prob_config.get("figsize", (10, 6))
        trials_per_run = self.prob_config["trials_per_run"]
        max_boxes = self.prob_config.get("max_boxes", 5)
        num_runs = self.prob_config["num_runs"]

        trials = list(range(1, trials_per_run + 1))

        plt.figure(figsize=figsize)

        for k in range(0, max_boxes + 1):
            probs = [self.prob_table[n][k] for n in trials]
            plt.plot(trials, probs, marker="o", label=f"{k} boxes")

        plt.xlabel("Trial number (n)")
        plt.ylabel("Probability")
        plt.title(
            f"Model probability of opening k boxes by trial n\n"
            f"({num_runs} SMC runs, {trials_per_run} trials per run)"
        )
        plt.legend()
        plt.grid(alpha=0.3)
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

    def plot_distribution_at_trial(self, trial_n=None):
        """
        Plot bar chart of P(K_n = k) at one chosen trial.
        """
        if self.prob_table is None:
            raise ValueError("Call fit() before plot_distribution_at_trial().")

        show_plot = self.prob_config.get("show_plot", True)
        save_plot = self.prob_config.get("save_plot", False)
        save_dir = self.prob_config.get("save_dir", "outputs")
        max_boxes = self.prob_config.get("max_boxes", 5)
        trials_per_run = self.prob_config["trials_per_run"]

        if trial_n is None:
            trial_n = self.prob_config.get("bar_plot_trial", 9)

        if not (1 <= trial_n <= trials_per_run):
            raise ValueError(f"trial_n must be in [1, {trials_per_run}], got {trial_n}")

        x_vals = list(range(0, max_boxes + 1))
        y_vals = [self.prob_table[trial_n][k] for k in x_vals]

        plt.figure(figsize=(8, 5))
        plt.bar(x_vals, y_vals, width=0.8)
        plt.xlabel(f"Number of boxes opened by trial {trial_n}")
        plt.ylabel("Probability")
        plt.title(f"Distribution of K_{trial_n}")
        plt.xticks(x_vals)
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        self.bar_save_path = None
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            bar_name = f"distribution_at_trial_{trial_n}.png"
            self.bar_save_path = os.path.join(save_dir, bar_name)
            plt.savefig(self.bar_save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return self.bar_save_path


# =========================
# Method 2 experiment runner
# =========================

class TrialProbabilityExperimentRunner:
    """
    Sweep over the SAME hyperparameter set as Method 1:
      1. theta prior
      2. generator prior
      3. true rule prior (prior_order)

    Each condition:
      - runs SMC num_runs times
      - each run lasts trials_per_run trials
      - estimates P(K_n = k) by empirical frequency
    """

    @staticmethod
    def make_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def make_config_tag(theta, generator_prior, true_rule_prior):
        theta_str = f"{theta[0]}_{theta[1]}"
        gen_str = str(generator_prior).replace(".", "p")
        rule_str = str(true_rule_prior).replace(".", "p")

        return (
            f"theta_{theta_str}"
            f"__gen_{gen_str}"
            f"__trueprior_{rule_str}"
        )

    def __init__(
        self,
        base_gen_config: dict,
        base_smc_config: dict,
        base_prob_config: dict,
        theta_list=None,
        generator_prior_list=None,
        true_rule_prior_list=None,
    ):
        self.base_gen_config = copy.deepcopy(base_gen_config)
        self.base_smc_config = copy.deepcopy(base_smc_config)
        self.base_prob_config = copy.deepcopy(base_prob_config)

        self.theta_list = theta_list or THETA_LIST
        self.generator_prior_list = generator_prior_list or GENERATOR_PRIOR_LIST
        self.true_rule_prior_list = true_rule_prior_list or TRUE_RULE_PRIOR_LIST

        self.results = []

    @staticmethod
    def assign_rule_priors(gcfg: dict, prior_order: float):
        """
        Set the 5 rule priors so they sum to 1:
            prior_order = given value
            the other 4 priors = (1 - prior_order) / 4
        """
        if not (0.0 <= prior_order <= 1.0):
            raise ValueError(f"prior_order must be in [0, 1], got {prior_order}")

        other_prior = (1.0 - prior_order) / 4.0

        gcfg["prior_color"] = other_prior
        gcfg["prior_order"] = prior_order
        gcfg["prior_shape"] = other_prior
        gcfg["prior_number"] = other_prior
        gcfg["prior_sim_color_total"] = other_prior

        return gcfg

    def build_configs(
        self,
        theta,
        generator_prior,
        true_rule_prior,
        num_runs=None,
        trials_per_run=None,
    ):
        gcfg = copy.deepcopy(self.base_gen_config)
        scfg = copy.deepcopy(self.base_smc_config)
        pcfg = copy.deepcopy(self.base_prob_config)

        # same hyperparameter logic as Method 1
        scfg["init_theta"] = theta
        gcfg["prop_random"] = generator_prior
        gcfg = self.assign_rule_priors(gcfg, prior_order=true_rule_prior)

        if num_runs is not None:
            pcfg["num_runs"] = num_runs
        if trials_per_run is not None:
            pcfg["trials_per_run"] = trials_per_run

        pcfg["save_plot"] = True
        pcfg["show_plot"] = False

        timestamp = self.make_timestamp()
        config_tag = self.make_config_tag(
            theta=theta,
            generator_prior=generator_prior,
            true_rule_prior=true_rule_prior,
        )

        experiment_dir = os.path.join(SAVING_DIR, f"{config_tag}__{timestamp}")
        pcfg["save_dir"] = experiment_dir
        pcfg["file_name"] = f"{config_tag}__{timestamp}.png"

        return gcfg, scfg, pcfg, experiment_dir, config_tag, timestamp

    @staticmethod
    def compute_expected_opened_curve(prob_table, max_trial, max_boxes=5):
        """
        E[K_n] = sum_k k * P(K_n = k)
        """
        expected_curve = {}
        for n in range(1, max_trial + 1):
            expected_curve[n] = sum(
                k * prob_table[n][k]
                for k in range(max_boxes + 1)
            )
        return expected_curve

    def run_one(
        self,
        theta,
        generator_prior,
        true_rule_prior,
        num_runs=None,
        trials_per_run=None,
    ):
        gcfg, scfg, pcfg, experiment_dir, config_tag, timestamp = self.build_configs(
            theta=theta,
            generator_prior=generator_prior,
            true_rule_prior=true_rule_prior,
            num_runs=num_runs,
            trials_per_run=trials_per_run,
        )

        fitter = TrialProbabilityFitter(
            gen_config=gcfg,
            smc_config=scfg,
            prob_config=pcfg,
        )

        trajectories, count_table, prob_table = fitter.fit()
        save_path = fitter.plot_probability_lines()
        bar_save_path = fitter.plot_distribution_at_trial(
            trial_n=pcfg.get("bar_plot_trial", 9)
        )

        expected_curve = self.compute_expected_opened_curve(
            prob_table=prob_table,
            max_trial=pcfg["trials_per_run"],
            max_boxes=pcfg.get("max_boxes", 5),
        )

        result = {
            "timestamp": timestamp,
            "config_tag": config_tag,
            "theta": list(theta),
            "generator_prior": generator_prior,
            "true_rule_prior": true_rule_prior,
            "num_runs": pcfg["num_runs"],
            "trials_per_run": pcfg["trials_per_run"],
            "prior_color": gcfg["prior_color"],
            "prior_order": gcfg["prior_order"],
            "prior_shape": gcfg["prior_shape"],
            "prior_number": gcfg["prior_number"],
            "prior_sim_color_total": gcfg["prior_sim_color_total"],
            "num_trajectories": len(trajectories),
            "probability_table": {
                str(n): {str(k): prob_table[n][k] for k in prob_table[n]}
                for n in prob_table
            },
            "count_table": {
                str(n): {str(k): count_table[n][k] for k in count_table[n]}
                for n in count_table
            },
            "expected_opened_curve": {
                str(n): expected_curve[n]
                for n in expected_curve
            },
            "save_path": save_path,
            "bar_save_path": bar_save_path,
        }

        os.makedirs(experiment_dir, exist_ok=True)

        result_json_path = os.path.join(
            experiment_dir,
            f"{config_tag}__{timestamp}.json"
        )

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        result["result_json_path"] = result_json_path

        self.results.append(result)
        return result

    def run_all(self, num_runs=None, trials_per_run=None):
        all_conditions = list(
            itertools.product(
                self.theta_list,
                self.generator_prior_list,
                self.true_rule_prior_list,
            )
        )

        print(f"Running {len(all_conditions)} experiment conditions...")

        for idx, (theta, generator_prior, true_rule_prior) in enumerate(all_conditions, start=1):
            print(
                f"[{idx}/{len(all_conditions)}] "
                f"theta={theta}, "
                f"generator_prior={generator_prior}, "
                f"prior_order={true_rule_prior}, "
                f"num_runs={num_runs if num_runs is not None else self.base_prob_config['num_runs']}, "
                f"trials_per_run={trials_per_run if trials_per_run is not None else self.base_prob_config['trials_per_run']}"
            )

            self.run_one(
                theta=theta,
                generator_prior=generator_prior,
                true_rule_prior=true_rule_prior,
                num_runs=num_runs,
                trials_per_run=trials_per_run,
            )

        return self.results

    def save_results_json(self, out_path=None):
        if out_path is None:
            out_path = os.path.join(
                self.base_prob_config.get("save_dir", SAVING_DIR),
                "experiments",
                "trial_probability_experiment_results.json",
            )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        return out_path

    def print_brief_table(self):
        print("\n=== Trial Probability Experiment Summary ===")
        for r in self.results:
            final_trial = str(r["trials_per_run"])
            e_final = r["expected_opened_curve"][final_trial]
            print(
                f"theta={tuple(r['theta'])}, "
                f"gen_prior={r['generator_prior']}, "
                f"prior_order={r['prior_order']:.3f}, "
                f"other_priors={r['prior_color']:.3f} | "
                f"E[K_{r['trials_per_run']}]={e_final:.3f}"
            )


if __name__ == "__main__":
    runner = TrialProbabilityExperimentRunner(
        base_gen_config=gen_config,
        base_smc_config=smc_config,
        base_prob_config=prob_config,
    )

    # Same hyperparameter grid, each condition:
    #   1000 runs, 70 trials each
    runner.run_all(num_runs=1000, trials_per_run=70)
    runner.print_brief_table()
    json_path = runner.save_results_json()

    print(f"\nSaved sweep results to: {json_path}")