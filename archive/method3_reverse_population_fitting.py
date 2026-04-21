import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List


# =========================
# CONFIG
# =========================
MAX_TRIALS = 70
EPS = 1e-4

KID_JSON_PATH = Path(r"data\Dolly_KeyEviModel_7.3.24.json")
MODEL_JSON_PATH = Path(
    r"neurips_drafts\data\theta_90_1__gen_0p85__trueprior_0p02__particles_30__runs_100__trials_70\trial_major_particles_and_ig.json"
)
OUTPUT_PATH = Path(r"training_results\kids_score_reverse\no_skill.json")


# =========================
# IO
# =========================
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================
# MODEL → P(n | t)
# =========================
def parse_trial_index(trial_key: str) -> int:
    return int(trial_key.split("_")[-1])


def build_model_distribution(model_data: Dict[str, Any]) -> Dict[int, Dict[int, float]]:
    trial_major_trace = model_data["trial_major_trace"]

    run_trial_opened = defaultdict(dict)

    for trial_key, runs_dict in trial_major_trace.items():
        t = parse_trial_index(trial_key)

        for run_key, rec in runs_dict.items():
            opened_before = int(rec["opened_before"])
            outcome = bool(rec["outcome"])
            opened_after = opened_before + (1 if outcome else 0)
            run_trial_opened[run_key][t] = opened_after

    run_to_counts = {}
    for run_key, by_t in run_trial_opened.items():
        dense = []
        last = 0
        for t in range(1, MAX_TRIALS + 1):
            if t in by_t:
                last = by_t[t]
            dense.append(last)
        run_to_counts[run_key] = dense

    n_runs = len(run_to_counts)
    dist = {}

    for t in range(1, MAX_TRIALS + 1):
        counts = {n: 0 for n in range(6)}

        for seq in run_to_counts.values():
            counts[seq[t - 1]] += 1

        probs = {}
        for n in range(6):
            p = counts[n] / n_runs
            if p == 0:
                p = EPS
            probs[n] = p

        z = sum(probs.values())
        probs = {n: probs[n] / z for n in range(6)}
        dist[t] = probs

    return dist


# =========================
# KID → opened counts
# =========================
def kid_to_counts(kid_trials: List[List[Any]]) -> List[int]:
    opened = set()
    counts = []

    for trial in kid_trials[:MAX_TRIALS]:
        _, box_id, outcome, *_ = trial

        if int(outcome) == 1 and box_id not in opened:
            opened.add(box_id)

        counts.append(len(opened))

    if len(counts) < MAX_TRIALS:
        last = counts[-1] if counts else 0
        counts.extend([last] * (MAX_TRIALS - len(counts)))

    return counts


# =========================
# SCORING
# =========================
def score_kid(kid_counts: List[int], dist: Dict[int, Dict[int, float]]) -> Dict[str, Any]:
    total_ll = 0.0
    trial_details = []

    for t in range(1, MAX_TRIALS + 1):
        n = kid_counts[t - 1]
        p = dist[t].get(n, EPS)
        if p == 0:
            p = EPS

        logp = math.log(p)
        total_ll += logp

        trial_details.append(
            {
                "trial": t,
                "opened_boxes": n,
                "likelihood": p,
                "log_likelihood": logp,
            }
        )

    return {
        "loglik": total_ll,
        "opened_counts": kid_counts,
        "trial_details": trial_details,
    }


# =========================
# MAIN
# =========================
def main():
    print("[INFO] Loading data...")
    kid_data = load_json(KID_JSON_PATH)
    model_data = load_json(MODEL_JSON_PATH)

    print("[INFO] Building model distribution P(n | t)...")
    dist = build_model_distribution(model_data)

    print("[INFO] Scoring kids...")
    results = {}

    for kid_id, trials in kid_data.items():
        counts = kid_to_counts(trials)
        result = score_kid(counts, dist)
        results[kid_id] = result

    all_ll = [v["loglik"] for v in results.values()]
    summary = {
        "num_kids": len(all_ll),
        "mean_loglik": sum(all_ll) / len(all_ll),
        "total_loglik": sum(all_ll),
        "best_kid": max(results.items(), key=lambda x: x[1]["loglik"])[0],
        "worst_kid": min(results.items(), key=lambda x: x[1]["loglik"])[0],
    }

    output = {
        "what_this_does": "Scores each child by comparing their cumulative opened-box counts across trials to the model's trial-wise distribution over opened-box counts.",
        "model_path": str(MODEL_JSON_PATH),
        "kid_path": str(KID_JSON_PATH),
        "distribution": {
            f"trial_{t}": dist[t] for t in range(1, MAX_TRIALS + 1)
        },
        "kid_scores": results,
        "summary": summary,
    }

    save_json(output, OUTPUT_PATH)

    print("\n===== DONE =====")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Mean LL: {summary['mean_loglik']:.4f}")
    print(f"Best kid: {summary['best_kid']}")
    print(f"Worst kid: {summary['worst_kid']}")


if __name__ == "__main__":
    main()