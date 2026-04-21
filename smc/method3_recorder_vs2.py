from smc_soc import Engine
from environment import Environment
from gen_soc import Generator

from collections import Counter
from copy import deepcopy
from itertools import product
from tqdm import tqdm

import json
import math
import os
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        pass


# =========================================================
# Base configs
# =========================================================
gen_config = {
    "omega": 2.0,
    "prop_random": 0.8,   # will be overwritten by sweep
    "true_prior": 0.2,    # will be overwritten by sweep
    "train": False,
    # when train is False, following parameters are disabled and a uniform distribution is used instead:
    "prior_color": 20,
    "prior_order": 20,
    "prior_shape": 10,
    "prior_number": 32,
    "prior_sim_color_total": 0,  # chaotic factor
}

smc_config = {
    "num_particles": 30,
    "init_theta": (1, 1),   # will be overwritten by sweep
    "ess_threshold": 0.5,
    "skill": True,
    "mode": "soc",
    "prior": "uniform",
    # "model": "gpt-4o", #or "deepseek-chat"
}

max_trials = 70
num_runs = 100
base_save_dir = r"training_results\smc_trace_sweeps_redo\SoC-Lesioned"

# =========================================================
# Hyperparameter lists for sweep
# =========================================================
# TRUE_PRIOR_LIST = [0.01, 0.02,0.03,0.04,0.05 ,0.06,0.07,0.08,0.09]
# PROP_RANDOM_LIST = [0.05, 0.2, 0.4, 0.6, 0.8]
# INIT_THETA_LIST = [
#     (1, 1),
#     (2, 1),
#     (6, 1),
#     (9, 1),
#     (19, 1),
# ]
# #no gen setup
# TRUE_PRIOR_LIST = [0.02]
# PROP_RANDOM_LIST = [0.01]
# part1 = [(1, b) for b in range(90, 9, -10)]
# part2 = [(1, b) for b in range(9, 0, -1)]
# part3 = [(a, 1) for a in range(2,10,1)]
# part4 = [(a, 1) for a in range(10, 91, 10)]
# INIT_THETA_LIST = part1 + part2 + part3 + part4
# print(INIT_THETA_LIST)

# #no skill setup
TRUE_PRIOR_LIST = [0.02]
PROP_RANDOM_LIST = [0.01]
INIT_THETA_LIST = [
    (99, 1),
]
# #fully leisioned setup
# TRUE_PRIOR_LIST = [0.02]
# PROP_RANDOM_LIST = [0.01]
# # INIT_THETA_LIST = [(90, 1)]
# TRUE_PRIOR_LIST = [0.02]
# PROP_RANDOM_LIST = [i / 100 for i in range(0, 100,5)]
# INIT_THETA_LIST = [
#     (1, 2),
#     (1, 1),
#     (2, 1),
#     (3, 1),
#     (4, 1),
#     (5, 1),
# ]

# TRUE_PRIOR_LIST = [0.01, 0.02,0.03,0.04,0.05 ,0.06,0.07,0.08,0.09]
# PROP_RANDOM_LIST = [0.1, 0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8,0.9]
# INIT_THETA_LIST = [
#     (1, 2),
#     (1, 3),
#     (1, 1),
#     (2, 1),
#     (3, 1),
#     (4, 1),
#     (5, 1),
#     (6, 1),
#     (9, 1),
#     (15,1),
#     (19, 1),
# ]

def observe_child_trial_manual(smc, key, box, outcome):
    """
    Same update order as Engine.run()
    """
    smc.evidence.append((key, box, outcome))

    if outcome:
        smc.proposal.prune_proposal_dist(key, box)
        smc.succ_count[(key.id, box.id)] += 1
    else:
        smc.fail_count[(key.id, box.id)] += 1

    if smc.skill:
        smc._compute_theta()

    smc._update_particle_weights(key, box, outcome)


def get_particle_weights_dict(smc):
    """
    Save all particle slots explicitly.
    """
    out = {}
    for i, p in enumerate(smc.particles):
        out[f"particle_{i+1}"] = {
            "name": p.name,
            "weight": float(p.weight),
        }
    return out


def get_all_action_ig_dict(smc):
    """
    Compute IG for all currently available actions.
    Mirrors opened-box filtering in Engine._select_action().
    """
    out = {}
    opened = {box_id for _, box_id in smc.env.success_pairs}

    for (key, box) in smc.env.actions:
        if key == "inspect":
            continue
        if box.id in opened:
            continue
        ig = smc._compute_info_gain(key, box)
        out[f"{key.id}->{box.id}"] = float(ig)
    return out


def collect_trial_major_trace(num_runs, gen_config, smc_config, max_trials, logger, pbar=None):
    """
    Output structure:
    {
      "trial_1": {
        "run_1": {...},
        "run_2": {...},
        ...
      },
      "trial_2": {...},
      ...
      "trial_70": {...}
    }
    """
    trial_major = {f"trial_{t}": {} for t in range(1, max_trials + 1)}
    trial_counts = []

    for run_idx in range(1, num_runs + 1):
        env = Environment(include_inspect=False)
        gen = Generator(gen_config, env)
        smc = Engine(smc_config, env, gen, logger)

        for t in range(1, max_trials + 1):
            if smc.env.is_solved():
                break

            particle_weights = get_particle_weights_dict(smc)
            all_action_ig = get_all_action_ig_dict(smc)

            key, box = smc._select_action()
            chosen_action = f"{key.id}->{box.id}"
            chosen_action_ig = all_action_ig.get(chosen_action, None)

            opened_before = len(smc.env.success_pairs)
            theta_before = (
                float(smc.alpha / (smc.alpha + smc.beta))
                if (smc.alpha + smc.beta) > 0 else 0.0
            )

            outcome = smc.env.test_action(key, box)

            trial_major[f"trial_{t}"][f"run_{run_idx}"] = {
                "particle_weights": particle_weights,
                "all_action_ig": all_action_ig,
                "chosen_action": chosen_action,
                "chosen_action_ig": chosen_action_ig,
                "outcome": bool(outcome),
                "opened_before": opened_before,
                "theta_before": theta_before,
            }

            observe_child_trial_manual(smc, key, box, outcome)

        completed_trials = sum(
            1 for t in range(1, max_trials + 1)
            if f"run_{run_idx}" in trial_major[f"trial_{t}"]
        )
        trial_counts.append(completed_trials)

        if pbar is not None:
            pbar.update(1)

    return trial_major, trial_counts


def sanitize_float_for_name(x):
    s = f"{x}".replace("-", "m").replace(".", "p")
    return s


def make_run_folder(base_dir, gen_config, smc_config, num_runs, max_trials):
    alpha0, beta0 = smc_config["init_theta"]
    folder_name = (
        f"theta_{sanitize_float_for_name(alpha0)}_{sanitize_float_for_name(beta0)}"
        f"__gen_{sanitize_float_for_name(gen_config['prop_random'])}"
        f"__trueprior_{sanitize_float_for_name(gen_config['true_prior'])}"
        f"__particles_{smc_config['num_particles']}"
        f"__runs_{num_runs}"
        f"__trials_{max_trials}"
    )
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_trial_major_trace(filepath, trial_major, gen_config, smc_config, num_runs, max_trials):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    payload = {
        "meta": {
            "num_runs": num_runs,
            "max_trials": max_trials,
            "gen_config": gen_config,
            "smc_config": smc_config,
        },
        "trial_major_trace": trial_major,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved JSON to: {filepath}")


def save_histogram_png(filepath, trial_counts, gen_config, smc_config, num_runs, max_trials):
    histogram = Counter(trial_counts)
    xs = sorted(histogram.keys())
    ys = [histogram[x] for x in xs]

    alpha0, beta0 = smc_config["init_theta"]

    plt.figure(figsize=(10, 6))
    plt.bar(xs, ys)
    plt.xlabel("Trials to solve")
    plt.ylabel("Count")
    plt.title("Trials to solve across runs")
    plt.suptitle(
        f"alpha0={alpha0}, beta0={beta0}, "
        f"prop_random={gen_config['prop_random']}, "
        f"true_prior={gen_config['true_prior']}, "
        f"particles={smc_config['num_particles']}, "
        f"runs={num_runs}, max_trials={max_trials}",
        fontsize=10,
        y=0.94,
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {filepath}")


def run_one_hyperparam_set(true_prior, prop_random, init_theta, base_gen_config, base_smc_config,
                           num_runs, max_trials, logger, base_save_dir):
    this_gen_config = deepcopy(base_gen_config)
    this_smc_config = deepcopy(base_smc_config)

    this_gen_config["true_prior"] = true_prior
    this_gen_config["prop_random"] = prop_random
    this_smc_config["init_theta"] = init_theta

    save_dir = make_run_folder(
        base_dir=base_save_dir,
        gen_config=this_gen_config,
        smc_config=this_smc_config,
        num_runs=num_runs,
        max_trials=max_trials,
    )

    json_path = os.path.join(save_dir, "trial_major_particles_and_ig.json")
    png_path = os.path.join(save_dir, "trials_histogram.png")

    trial_major_trace, trial_counts = collect_trial_major_trace(
        num_runs=num_runs,
        gen_config=this_gen_config,
        smc_config=this_smc_config,
        max_trials=max_trials,
        logger=logger,
    )

    save_trial_major_trace(
        filepath=json_path,
        trial_major=trial_major_trace,
        gen_config=this_gen_config,
        smc_config=this_smc_config,
        num_runs=num_runs,
        max_trials=max_trials,
    )

    save_histogram_png(
        filepath=png_path,
        trial_counts=trial_counts,
        gen_config=this_gen_config,
        smc_config=this_smc_config,
        num_runs=num_runs,
        max_trials=max_trials,
    )


if __name__ == '__main__':
    logger = Logger(logging=False)

    sweep_settings = list(product(TRUE_PRIOR_LIST, PROP_RANDOM_LIST, INIT_THETA_LIST))
    total_sets = len(sweep_settings)

    print(f"Total hyperparameter sets: {total_sets}")

    with tqdm(total=total_sets, desc="Hyperparameter sweep") as pbar:
        for true_prior, prop_random, init_theta in sweep_settings:
            pbar.set_postfix({
                "true_prior": true_prior,
                "prop_random": prop_random,
                "init_theta": init_theta,
            })

            run_one_hyperparam_set(
                true_prior=true_prior,
                prop_random=prop_random,
                init_theta=init_theta,
                base_gen_config=gen_config,
                base_smc_config=smc_config,
                num_runs=num_runs,
                max_trials=max_trials,
                logger=logger,
                base_save_dir=base_save_dir,
            )

            pbar.update(1)