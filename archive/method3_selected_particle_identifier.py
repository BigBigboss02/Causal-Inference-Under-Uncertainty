from smc_soc import Engine
from environment import Environment
from gen_soc import Generator

from copy import deepcopy
from collections import Counter
import json
import os


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
    "prop_random": 0.4,
    "true_prior": 0.02,
    "train": False,
    "prior_color": 20,
    "prior_order": 20,
    "prior_shape": 10,
    "prior_number": 32,
    "prior_sim_color_total": 0,
}

smc_config = {
    "num_particles": 30,
    "init_theta": (2, 1),
    "ess_threshold": 0.5,
    "skill": True,
    "mode": "soc",
    "prior": "uniform",
}

max_trials = 70
num_runs = 100

base_save_dir = r"training_results\smc_selected_hypothesis"


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


def sanitize_float_for_name(x):
    return f"{x}".replace("-", "m").replace(".", "p")


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


def get_top_particles(particle_weights):
    """
    Find highest-weight particle(s), keeping ties.
    """
    if not particle_weights:
        return None, []

    max_weight = max(v["weight"] for v in particle_weights.values())

    top_particles = []
    for particle_id, info in particle_weights.items():
        if info["weight"] == max_weight:
            top_particles.append({
                "particle_id": particle_id,
                "name": info["name"],
                "weight": float(info["weight"]),
            })

    return float(max_weight), top_particles


def get_selected_hypothesis_names(top_particles):
    """
    Return unique hypothesis names among top particles.
    """
    names = sorted(list(set(p["name"] for p in top_particles)))
    return names


def record_final_trace_of_each_run(num_runs, gen_config, smc_config, max_trials, logger):
    """
    For each run, record ONLY the final particle state after the run ends.

    Output:
    {
      "run_selection_summary": {
        "run_1": ["number_match"],
        "run_2": ["generator_89", "generator_92"],
        ...
      },
      "run_final_traces": {
        "run_1": {
          "completed_trials": 17,
          "solved": true,
          "final_theta": 0.91,
          "particle_weights": {...},
          "max_weight": 0.42,
          "top_particles": [...],
          "selected_hypothesis_names": ["number_match"]
        },
        ...
      }
    }
    """
    run_final_traces = {}
    run_selection_summary = {}

    for run_idx in range(1, num_runs + 1):
        env = Environment(include_inspect=False)
        gen = Generator(gen_config, env)
        smc = Engine(smc_config, env, gen, logger)

        completed_trials = 0

        for t in range(1, max_trials + 1):
            if smc.env.is_solved():
                break

            key, box = smc._select_action()
            outcome = smc.env.test_action(key, box)
            observe_child_trial_manual(smc, key, box, outcome)
            completed_trials += 1

        particle_weights = get_particle_weights_dict(smc)
        max_weight, top_particles = get_top_particles(particle_weights)
        selected_hypothesis_names = get_selected_hypothesis_names(top_particles)

        final_theta = (
            float(smc.alpha / (smc.alpha + smc.beta))
            if (smc.alpha + smc.beta) > 0 else 0.0
        )

        run_key = f"run_{run_idx}"

        run_selection_summary[run_key] = selected_hypothesis_names

        run_final_traces[run_key] = {
            "completed_trials": completed_trials,
            "solved": bool(smc.env.is_solved()),
            "final_theta": final_theta,
            "particle_weights": particle_weights,
            "max_weight": max_weight,
            "top_particles": top_particles,
            "selected_hypothesis_names": selected_hypothesis_names,
        }

    return {
        "run_selection_summary": run_selection_summary,
        "run_final_traces": run_final_traces,
    }


def add_selection_counts(run_selection_summary):
    """
    Count how often each selected hypothesis pattern appears across runs.
    """
    counter = Counter()

    for _, names in run_selection_summary.items():
        label = " | ".join(names) if names else "None"
        counter[label] += 1

    return dict(counter)


def save_final_trace_of_each_run(filepath, trace_payload, gen_config, smc_config, num_runs, max_trials):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    run_selection_summary = trace_payload["run_selection_summary"]
    run_final_traces = trace_payload["run_final_traces"]
    selection_counts = add_selection_counts(run_selection_summary)

    payload = {
        "meta": {
            "num_runs": num_runs,
            "max_trials": max_trials,
            "gen_config": gen_config,
            "smc_config": smc_config,
            "record_type": "final_particle_state_of_each_run",
        },
        "run_selection_summary": run_selection_summary,
        "selection_counts_across_runs": selection_counts,
        "run_final_traces": run_final_traces,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved JSON to: {filepath}")


def run_one_config(true_prior, prop_random, init_theta, base_gen_config, base_smc_config,
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

    json_path = os.path.join(save_dir, "final_particle_state_each_run.json")

    trace_payload = record_final_trace_of_each_run(
        num_runs=num_runs,
        gen_config=this_gen_config,
        smc_config=this_smc_config,
        max_trials=max_trials,
        logger=logger,
    )

    save_final_trace_of_each_run(
        filepath=json_path,
        trace_payload=trace_payload,
        gen_config=this_gen_config,
        smc_config=this_smc_config,
        num_runs=num_runs,
        max_trials=max_trials,
    )


if __name__ == "__main__":
    logger = Logger(logging=False)

    true_prior = 0.02
    prop_random = 0.4
    init_theta = (2, 1)

    run_one_config(
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