import json
import os
import math
from itertools import product
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

from environment import Environment
from gen_soc import Generator
from smc_soc_train import Engine


class Logger:
    def __init__(self, logging: bool = False):
        self.logging = logging

    def log(self, log_str: str):
        if self.logging:
            print(log_str)


# =========================================================
# Hyperparameter lists for sweep
# =========================================================

# Full sweep:
TRUE_PRIOR_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
PROP_RANDOM_LIST = [0.05, 0.2, 0.4, 0.6, 0.8]
INIT_THETA_LIST = [
    (1, 1),
    (2, 2),
    (5, 2),
    (9, 1),
    (19, 1),
]



# =========================================================
# Fixed config parts
# =========================================================
BASE_GEN_CONFIG = {
    "omega": 2.0,
    "train": False,
    "prior_color": 20,
    "prior_order": 20,
    "prior_shape": 10,
    "prior_number": 32,
    "prior_sim_color_total": 0,
}

BASE_SMC_CONFIG = {
    "num_particles": 30,
    "ess_threshold": 0.5,
    "skill": True,
    "mode": "soc",
    "prior": "uniform",
}

num_runs = 1

kids_data_path = r"data\Dolly_KeyEviModel_7.3.24.json"
root_output_dir = r"training_results\smc_human_data_lead"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to: {path}")


def convert_one_kid_sequence(kid_seq: List[List[Any]]) -> List[Dict[str, Any]]:
    """
    Input format for one kid:
    [
        ["red", "red", 0, {...}],
        ["red", "red", 1, {...}],
        ...
    ]

    Convert to replay format:
    [
        {"key": "red", "box": "red", "outcome": False},
        {"key": "red", "box": "red", "outcome": True},
        ...
    ]
    """
    converted = []

    for entry in kid_seq:
        if not isinstance(entry, list) or len(entry) < 3:
            continue

        key_id = entry[0]
        box_id = entry[1]
        outcome = bool(entry[2])

        converted.append({
            "key": key_id,
            "box": box_id,
            "outcome": outcome,
        })

    return converted


def compress_trace(full_trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep:
    - kid_action
    - model_chosen_action
    - kid_action_ig
    - all_action_ig
    """
    compact = {}

    for trial_name, run_dict in full_trace.items():
        for run_name, info in run_dict.items():
            kid_action = info["kid_action"]
            all_action_ig = info.get("all_action_ig", {})
            kid_action_ig = all_action_ig.get(kid_action, None)

            compact.setdefault(run_name, {})
            compact[run_name][trial_name] = {
                "kid_action": kid_action,
                "model_chosen_action": info["chosen_action"],
                "kid_action_ig": kid_action_ig,
                "all_action_ig": all_action_ig,
            }

    return compact


def compute_log_likelihood(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Likelihood rule:
    - if kid_action == model_chosen_action:
        prob = 1 - 0.001 * 64 = 0.936
    - else:
        prob = 0.001

    Then log-likelihood is sum of log(prob).
    """
    eps = 0.001
    num_actions = 65
    match_prob = 1.0 - eps * (num_actions - 1)  # 0.936

    ll_result = {
        "meta": {
            "match_prob": match_prob,
            "mismatch_prob": eps,
            "num_actions_assumed": num_actions,
            "log_match_prob": math.log(match_prob),
            "log_mismatch_prob": math.log(eps),
        },
        "kid_ll": {},
        "total_ll": 0.0,
    }

    for kid_id, runs in result["kid_major_trace"].items():
        ll_result["kid_ll"][kid_id] = {}

        for run_name, trials in runs.items():
            run_ll = 0.0
            trial_ll = {}

            for trial_name, info in trials.items():
                kid_action = info["kid_action"]
                model_action = info["model_chosen_action"]

                if kid_action == model_action:
                    prob = match_prob
                else:
                    prob = eps

                log_prob = math.log(prob)
                run_ll += log_prob

                trial_ll[trial_name] = {
                    "kid_action": kid_action,
                    "model_chosen_action": model_action,
                    "assigned_prob": prob,
                    "log_prob": log_prob,
                    "match": (kid_action == model_action),
                }

            ll_result["kid_ll"][kid_id][run_name] = {
                "run_ll": run_ll,
                "trial_ll": trial_ll,
            }

            ll_result["total_ll"] += run_ll

    return ll_result


def float_tag(x: float) -> str:
    """
    0.05 -> 0p05
    0.2  -> 0p2
    """
    return str(x).replace(".", "p")


def make_setting_name(init_theta: Tuple[int, int], prop_random: float, true_prior: float) -> str:
    alpha0, beta0 = init_theta
    return (
        f"theta_{alpha0}_{beta0}"
        f"__gen_{float_tag(prop_random)}"
        f"__trueprior_{float_tag(true_prior)}"
    )


def run_one_setting(
    kids_data: Dict[str, Any],
    gen_config: Dict[str, Any],
    smc_config: Dict[str, Any],
    num_runs: int,
) -> Dict[str, Any]:
    result = {
        "meta": {
            "num_runs": num_runs,
            "gen_config": gen_config,
            "smc_config": smc_config,
        },
        "kid_major_trace": {},
    }

    for kid_id, kid_seq in tqdm(
        kids_data.items(),
        desc="Kids",
        leave=False,
        total=len(kids_data),
    ):
        if not isinstance(kid_seq, list) or len(kid_seq) == 0:
            continue

        replay_seq = convert_one_kid_sequence(kid_seq)
        result["kid_major_trace"][kid_id] = {}

        for run_idx in range(1, num_runs + 1):
            logger = Logger(logging=False)
            env = Environment(include_inspect=False)
            gen = Generator(gen_config, env)
            engine = Engine(smc_config, env, gen, logger)

            full_trace = engine.replay_actions(
                replay_seq,
                run_name=f"run_{run_idx}"
            )

            compact_trace = compress_trace(full_trace)
            result["kid_major_trace"][kid_id].update(compact_trace)

    result["log_likelihood"] = compute_log_likelihood(result)
    return result


def main():
    kids_data = load_json(kids_data_path)

    config_setups = []
    for init_theta, prop_random, true_prior in product(
        INIT_THETA_LIST,
        PROP_RANDOM_LIST,
        TRUE_PRIOR_LIST,
    ):
        gen_config = dict(BASE_GEN_CONFIG)
        gen_config["prop_random"] = prop_random
        gen_config["true_prior"] = true_prior

        smc_config = dict(BASE_SMC_CONFIG)
        smc_config["init_theta"] = init_theta

        config_setups.append({
            "setting_name": make_setting_name(init_theta, prop_random, true_prior),
            "gen_config": gen_config,
            "smc_config": smc_config,
            "init_theta": init_theta,
            "prop_random": prop_random,
            "true_prior": true_prior,
        })

    print(f"Total config setups: {len(config_setups)}")

    summary = []

    for idx, setup in enumerate(
        tqdm(config_setups, desc="Hyperparameter Sweep", total=len(config_setups)),
        start=1,
    ):
        setting_name = setup["setting_name"]
        gen_config = setup["gen_config"]
        smc_config = setup["smc_config"]

        result = run_one_setting(
            kids_data=kids_data,
            gen_config=gen_config,
            smc_config=smc_config,
            num_runs=num_runs,
        )

        output_dir = os.path.join(root_output_dir, setting_name)
        output_path = os.path.join(output_dir, "kids_model_action_summary.json")
        save_json(output_path, result)

        summary.append({
            "setting_name": setting_name,
            "init_theta": list(setup["init_theta"]),
            "prop_random": setup["prop_random"],
            "true_prior": setup["true_prior"],
            "total_ll": result["log_likelihood"]["total_ll"],
            "output_path": output_path,
        })

    summary_path = os.path.join(root_output_dir, "sweep_summary.json")
    save_json(summary_path, {"summary": summary})

    if summary:
        best = max(summary, key=lambda x: x["total_ll"])
        print("\nBEST SETTING:")
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()