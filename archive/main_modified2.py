import os
import json
from datetime import datetime
from collections import Counter

from smc_soc import Engine
from environment import Environment
from gen_soc import Generator


# ============================================================
# USER-EDIT VARIABLES
# ============================================================

# Output folder
SAVE_ROOT = r"training_results\CCN_plots_data\smc_s\31stMar"

# Main experiment controls
NUM_RUNS = 1000
MAX_TRIALS = 70

# SMC-S hyperparameters to change
# C. Lower skill + low randomness
SIM_THETA = (2,2)
SIM_PROP_RANDOM = 0.6


# run single or multiple settings:
# SIM_TRUE_RULE_PRIOR = 0.011
SIM_TRUE_RULE_PRIORS = [round(x / 1000, 3) for x in range(40, 59, 2)]  # 0.040 to 0.058
REPEATS_PER_SETTING = 20




# Optional custom output filename.
# Set to None to use automatic timestamped naming.
OUTPUT_JSON_NAME = None

# ============================================================
# SMC-S CCN PANEL SETTINGS (A–D)
# ============================================================
# A. Very low randomness + very high skill
#    SIM_THETA = (19, 1)
#    SIM_PROP_RANDOM = 0.01
#    SIM_TRUE_RULE_PRIOR = 0.05
#
# B. Lesioned skill + moderate randomness
#    SIM_THETA = (2, 2)
#    SIM_PROP_RANDOM = 0.6
#    SIM_TRUE_RULE_PRIOR = 0.05
#
# C. Lower skill + low randomness
#    SIM_THETA = (4, 2)
#    SIM_PROP_RANDOM = 0.05
#    SIM_TRUE_RULE_PRIOR = 0.05
#
# D. Balanced skill + balanced randomness
#    SIM_THETA = (6, 1)
#    SIM_PROP_RANDOM = 0.1
#    SIM_TRUE_RULE_PRIOR = 0.05
# ============================================================


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        pass


gen_config = {
    "omega": 2.0,
    "prop_random": 0.8,
    "true_prior": 0.2,   # keep this because Generator expects it
    "train": False,
    # These will be overwritten by assign_rule_priors(...)
    "prior_color": 20,
    "prior_order": 20,
    "prior_shape": 10,
    "prior_number": 32,
    "prior_sim_color_total": 8,
}

smc_config = {
    "num_particles": 30,
    "init_theta": (1, 1),
    "ess_threshold": 0.5,
    "skill": True,
    "mode": "soc",
}


def make_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_config_tag(theta, generator_prior, true_rule_prior, num_runs, trials_per_run):
    theta_str = f"{theta[0]}_{theta[1]}"
    gen_str = str(generator_prior).replace(".", "p")
    prior_str = str(true_rule_prior).replace(".", "p")
    return (
        f"theta_{theta_str}"
        f"__gen_{gen_str}"
        f"__priororder_{prior_str}"
        f"__runs_{num_runs}"
        f"__trials_{trials_per_run}"
    )


def assign_rule_priors(gcfg: dict, prior_order: float):
    """
    Force the 5 rule priors to sum to 1:
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


def summarise_attempt_counts(attempt_counts):
    freq = Counter(attempt_counts)
    return {str(k): freq[k] for k in sorted(freq.keys())}


def detect_success(history, trials_per_run):
    """
    Success rule:
    1. Prefer final 'opened' count if available.
    2. Otherwise fall back to early stopping logic: solved if len(history) < trials_per_run.
    """
    if history and isinstance(history[-1], dict) and "opened" in history[-1]:
        final_opened = int(history[-1].get("opened", 0))
        return final_opened >= 5

    return len(history) < trials_per_run


# if __name__ == '__main__':
#     num_runs = NUM_RUNS
#     trials_per_run = MAX_TRIALS
#     logger = Logger(logging=False)

#     # Apply user-set hyperparameters
#     local_gen_config = dict(gen_config)
#     local_smc_config = dict(smc_config)

#     local_smc_config["init_theta"] = SIM_THETA
#     local_gen_config["prop_random"] = SIM_PROP_RANDOM
#     true_rule_prior = SIM_TRUE_RULE_PRIOR

#     # keep both for compatibility:
#     # - true_prior is used by current Generator
#     # - prior_order + family priors are used by your experiment logic
#     local_gen_config["true_prior"] = true_rule_prior
#     local_gen_config = assign_rule_priors(local_gen_config, true_rule_prior)

#     trial_counts = []
#     num_successful_runs = 0
#     num_failed_runs = 0

#     for _ in range(num_runs):
#         environment = Environment(include_inspect=False)
#         generator = Generator(local_gen_config, environment)
#         smc_engine = Engine(local_smc_config, environment, generator, logger)
#         history = smc_engine.run(max_trials=trials_per_run)

#         n_steps = len(history)
#         trial_counts.append(n_steps)

#         if detect_success(history, trials_per_run):
#             num_successful_runs += 1
#         else:
#             num_failed_runs += 1

#     timestamp = make_timestamp()
#     theta = local_smc_config["init_theta"]
#     generator_prior = local_gen_config["prop_random"]

#     config_tag = make_config_tag(
#         theta=theta,
#         generator_prior=generator_prior,
#         true_rule_prior=true_rule_prior,
#         num_runs=num_runs,
#         trials_per_run=trials_per_run,
#     )

#     experiment_dir = os.path.join(SAVE_ROOT, f"{config_tag}__{timestamp}")
#     os.makedirs(experiment_dir, exist_ok=True)

#     result = {
#         "timestamp": timestamp,
#         "config_tag": config_tag,
#         "theta": list(theta),
#         "generator_prior": generator_prior,
#         "true_rule_prior": true_rule_prior,
#         "num_runs": num_runs,
#         "trials_per_run": trials_per_run,
#         "true_prior": local_gen_config["true_prior"],
#         "prior_color": local_gen_config["prior_color"],
#         "prior_order": local_gen_config["prior_order"],
#         "prior_shape": local_gen_config["prior_shape"],
#         "prior_number": local_gen_config["prior_number"],
#         "prior_sim_color_total": local_gen_config["prior_sim_color_total"],
#         "num_successful_runs": num_successful_runs,
#         "num_failed_runs": num_failed_runs,
#         "attempt_counts": trial_counts,
#         "frequency_table": summarise_attempt_counts(trial_counts),
#         "save_path": None,
#         "mean_attempts": sum(trial_counts) / len(trial_counts) if trial_counts else None,
#         "min_attempts": min(trial_counts) if trial_counts else None,
#         "max_attempts": max(trial_counts) if trial_counts else None,
#     }

#     if OUTPUT_JSON_NAME is None:
#         result_json_path = os.path.join(
#             experiment_dir,
#             f"{config_tag}__{timestamp}.json"
#         )
#     else:
#         os.makedirs(SAVE_ROOT, exist_ok=True)
#         result_json_path = os.path.join(SAVE_ROOT, OUTPUT_JSON_NAME)

#     os.makedirs(os.path.dirname(result_json_path), exist_ok=True)

#     with open(result_json_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2)

#     print(f"Saved JSON result to: {result_json_path}")
if __name__ == '__main__':
    num_runs = NUM_RUNS
    trials_per_run = MAX_TRIALS
    logger = Logger(logging=False)

    for true_rule_prior in SIM_TRUE_RULE_PRIORS:
        print(f"\nRunning prior={true_rule_prior}")

        # Fresh configs each time
        local_gen_config = dict(gen_config)
        local_smc_config = dict(smc_config)

        local_smc_config["init_theta"] = SIM_THETA
        local_gen_config["prop_random"] = SIM_PROP_RANDOM

        # Apply prior
        local_gen_config["true_prior"] = true_rule_prior
        local_gen_config = assign_rule_priors(local_gen_config, true_rule_prior)

        trial_counts = []
        num_successful_runs = 0
        num_failed_runs = 0

        for _ in range(num_runs):
            environment = Environment(include_inspect=False)
            generator = Generator(local_gen_config, environment)
            smc_engine = Engine(local_smc_config, environment, generator, logger)

            history = smc_engine.run(max_trials=trials_per_run)

            n_steps = len(history)
            trial_counts.append(n_steps)

            if detect_success(history, trials_per_run):
                num_successful_runs += 1
            else:
                num_failed_runs += 1

        # ===== Save result =====
        timestamp = make_timestamp()
        theta = local_smc_config["init_theta"]
        generator_prior = local_gen_config["prop_random"]

        config_tag = make_config_tag(
            theta=theta,
            generator_prior=generator_prior,
            true_rule_prior=true_rule_prior,
            num_runs=num_runs,
            trials_per_run=trials_per_run,
        )

        experiment_dir = os.path.join(SAVE_ROOT, f"{config_tag}__{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)

        result = {
            "timestamp": timestamp,
            "config_tag": config_tag,
            "theta": list(theta),
            "generator_prior": generator_prior,
            "true_rule_prior": true_rule_prior,
            "num_runs": num_runs,
            "trials_per_run": trials_per_run,
            "true_prior": local_gen_config["true_prior"],
            "prior_color": local_gen_config["prior_color"],
            "prior_order": local_gen_config["prior_order"],
            "prior_shape": local_gen_config["prior_shape"],
            "prior_number": local_gen_config["prior_number"],
            "prior_sim_color_total": local_gen_config["prior_sim_color_total"],
            "num_successful_runs": num_successful_runs,
            "num_failed_runs": num_failed_runs,
            "attempt_counts": trial_counts,
            "frequency_table": summarise_attempt_counts(trial_counts),
            "mean_attempts": sum(trial_counts) / len(trial_counts) if trial_counts else None,
            "min_attempts": min(trial_counts) if trial_counts else None,
            "max_attempts": max(trial_counts) if trial_counts else None,
        }

        result_json_path = os.path.join(
            experiment_dir,
            f"{config_tag}__{timestamp}.json"
        )

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"Saved: {result_json_path}")