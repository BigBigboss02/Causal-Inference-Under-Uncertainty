import json
import math
import os
from typing import Dict, List, Any


# =========================================================
# Configs
# =========================================================
root_sweep_dir = r"training_results\smc_trace_sweeps"
filepath2 = r"data\Dolly_KeyEviModel_7.3.24.json"   # kids json

# trace filename expected inside each hyperparameter folder
trace_json_name = "trial_major_particles_and_ig.json"

# output filename to save inside each hyperparameter folder
output_json_name = "output_loglik.json"

# kept for compatibility, no longer used in likelihood assignment
temperature = 1.0

# probability assigned to every non-chosen action
missing_action_prob = 0.001


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_kid_action(kid_trial_entry: List[Any]) -> str:
    """
    Expected kid trial format:
    [key_id, box_id, outcome, {...}]
    Example:
    ["red", "red", 0, {...}]
    """
    if not isinstance(kid_trial_entry, list) or len(kid_trial_entry) < 2:
        raise ValueError(f"Invalid kid trial entry: {kid_trial_entry}")
    key_id = kid_trial_entry[0]
    box_id = kid_trial_entry[1]
    return f"{key_id}->{box_id}"


def build_kid_trials(kids_data: Dict[str, List[List[Any]]]) -> Dict[str, Dict[str, str]]:
    """
    Returns:
    {
      "trial_1": {"D001": "red->red", "D002": "...", ...},
      "trial_2": {...},
      ...
    }
    """
    out: Dict[str, Dict[str, str]] = {}
    for kid_id, seq in kids_data.items():
        if not isinstance(seq, list):
            continue
        for idx, entry in enumerate(seq, start=1):
            trial_name = f"trial_{idx}"
            action = extract_kid_action(entry)
            out.setdefault(trial_name, {})
            out[trial_name][kid_id] = action
    return out


def trial_sort_key(trial_name: str) -> int:
    return int(trial_name.split("_")[1])


def run_sort_key(run_name: str) -> int:
    return int(run_name.split("_")[1])


def compute_ll(trace_data: Dict[str, Any],
               kids_data: Dict[str, Any],
               temperature: float = 1.0,
               missing_action_prob: float = 1e-12) -> Dict[str, Any]:
    """
    For each run r and trial t:
      LL_run_trial(r,t) = sum_over_kids_at_trial_t log P_r,t(kid_action)

    where P_r,t(.) is defined as:
      - chosen_action gets probability 1 - eps * (N - 1)
      - every other action gets probability eps

    Then:
      LL_run(r) = sum_t LL_run_trial(r,t)
      LL_total = sum_r LL_run(r)
    """
    trial_major_trace = trace_data["trial_major_trace"]
    kid_trials = build_kid_trials(kids_data)

    sorted_trials = sorted(trial_major_trace.keys(), key=trial_sort_key)

    all_run_names = set()
    for trial_name in sorted_trials:
        all_run_names.update(trial_major_trace[trial_name].keys())
    sorted_runs = sorted(all_run_names, key=run_sort_key)

    result = {
        "meta": {
            "temperature": temperature,
            "missing_action_prob": missing_action_prob,
            "trace_meta": trace_data.get("meta", {}),
            "action_model": "chosen_action_with_fixed_noise",
        },
        "trial_ll": {},
        "run_ll": {},
        "total_ll": 0.0,
    }

    run_totals = {run_name: 0.0 for run_name in sorted_runs}

    for trial_name in sorted_trials:
        result["trial_ll"][trial_name] = {}

        kids_at_trial = kid_trials.get(trial_name, {})
        runs_dict = trial_major_trace[trial_name]

        for run_name in sorted_runs:
            if run_name not in runs_dict:
                result["trial_ll"][trial_name][run_name] = {
                    "num_kids_used": 0,
                    "trial_ll": 0.0,
                    "kid_details": {},
                }
                continue

            run_info = runs_dict[run_name]
            all_action_ig = run_info.get("all_action_ig", {})
            chosen_action = run_info.get("chosen_action")

            trial_ll_sum = 0.0
            num_kids_used = 0
            kid_details = {}

            num_actions = len(all_action_ig)

            if num_actions <= 0:
                action_dist = {}
            elif chosen_action is None or chosen_action not in all_action_ig:
                uniform_p = 1.0 / num_actions
                action_dist = {a: uniform_p for a in all_action_ig.keys()}
            else:
                other_action_prob = missing_action_prob
                chosen_action_prob = 1.0 - other_action_prob * (num_actions - 1)

                if chosen_action_prob <= 0:
                    raise ValueError(
                        f"missing_action_prob={missing_action_prob} is too large for "
                        f"{num_actions} actions; chosen_action_prob becomes {chosen_action_prob}"
                    )

                action_dist = {a: other_action_prob for a in all_action_ig.keys()}
                action_dist[chosen_action] = chosen_action_prob

            for kid_id, kid_action in kids_at_trial.items():
                prob = action_dist.get(kid_action, missing_action_prob)
                prob = max(prob, missing_action_prob)
                log_prob = math.log(prob)
                kid_action_ig = all_action_ig.get(kid_action, None)

                kid_details[kid_id] = {
                    "action": kid_action,
                    "action_ig": kid_action_ig,
                    "model_chosen_action": chosen_action,
                    "assigned_prob": prob,
                    "log_prob": log_prob,
                }

                trial_ll_sum += log_prob
                num_kids_used += 1

            result["trial_ll"][trial_name][run_name] = {
                "num_kids_used": num_kids_used,
                "trial_ll": trial_ll_sum,
                "kid_details": kid_details,
            }

            run_totals[run_name] += trial_ll_sum

    result["run_ll"] = run_totals
    result["total_ll"] = sum(run_totals.values())

    return result


def save_json(path: str, payload: Dict[str, Any]) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to: {path}")


def find_trace_jsons(root_dir: str, trace_json_name: str) -> List[str]:
    """
    Find all trace json files under sweep root.
    """
    found = []
    for current_root, _, files in os.walk(root_dir):
        if trace_json_name in files:
            found.append(os.path.join(current_root, trace_json_name))
    return sorted(found)


if __name__ == "__main__":
    kids_data = load_json(filepath2)

    trace_paths = find_trace_jsons(root_sweep_dir, trace_json_name)

    if not trace_paths:
        raise FileNotFoundError(
            f"No '{trace_json_name}' found under: {root_sweep_dir}"
        )

    print(f"Found {len(trace_paths)} trace json file(s).")

    for trace_path in trace_paths:
        print(f"\nProcessing: {trace_path}")
        trace_data = load_json(trace_path)

        result = compute_ll(
            trace_data=trace_data,
            kids_data=kids_data,
            temperature=temperature,
            missing_action_prob=missing_action_prob,
        )

        out_path = os.path.join(os.path.dirname(trace_path), output_json_name)
        save_json(out_path, result)