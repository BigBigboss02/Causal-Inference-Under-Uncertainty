import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List


MAX_TRIALS = 70
EPS = 1e-4

SWEEP_ROOT = Path(r"training_results\smc_trace_sweeps\small_true_prior")
KID_JSON_PATH = Path(r"data\Dolly_KeyEviModel_7.3.24.json")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_trial_index(trial_key: str) -> int:
    return int(trial_key.split("_")[-1])


def find_trace_json(trace_dir: Path) -> Path:
    """
    Find the main trace json containing trial_major_trace.
    """
    candidates = sorted(trace_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No json files found in {trace_dir}")

    for path in candidates:
        try:
            data = load_json(path)
            if isinstance(data, dict) and "trial_major_trace" in data:
                return path
        except Exception:
            continue

    raise FileNotFoundError(
        f"No json containing 'trial_major_trace' found in {trace_dir}"
    )


def extract_open_counts_per_run(
    trace_data: Dict[str, Any],
    max_trials: int = MAX_TRIALS,
) -> Dict[str, List[int]]:
    """
    For each run, build:
        run_key -> [n_1, n_2, ..., n_70]
    where n_t is number of boxes opened AFTER trial t.

    opened_after = opened_before + 1 if outcome == True else opened_before

    If a run ends early, pad remaining trials with the last opened count.
    """
    if "trial_major_trace" not in trace_data:
        raise KeyError("Missing 'trial_major_trace' in trace data.")

    trial_major_trace = trace_data["trial_major_trace"]
    run_trial_opened = defaultdict(dict)

    for trial_key, runs_dict in trial_major_trace.items():
        t = parse_trial_index(trial_key)

        for run_key, rec in runs_dict.items():
            opened_before = int(rec["opened_before"])
            outcome = bool(rec["outcome"])
            opened_after = opened_before + (1 if outcome else 0)
            run_trial_opened[run_key][t] = opened_after

    run_to_counts: Dict[str, List[int]] = {}

    for run_key, by_t in run_trial_opened.items():
        dense = []
        last = 0
        for t in range(1, max_trials + 1):
            if t in by_t:
                last = by_t[t]
            dense.append(last)
        run_to_counts[run_key] = dense

    return run_to_counts


def build_trial_distribution(
    run_to_counts: Dict[str, List[int]],
    max_trials: int = MAX_TRIALS,
    eps: float = EPS,
) -> Dict[int, Dict[int, float]]:
    """
    Build:
        P(n | t, theta) = #runs with n boxes at trial t / N

    Then smooth zero probabilities and renormalize.
    """
    if not run_to_counts:
        raise ValueError("No run data found.")

    n_runs = len(run_to_counts)
    dist: Dict[int, Dict[int, float]] = {}

    for t in range(1, max_trials + 1):
        counts = {n: 0 for n in range(6)}

        for opened_seq in run_to_counts.values():
            n_opened = int(opened_seq[t - 1])
            if n_opened < 0 or n_opened > 5:
                raise ValueError(f"Invalid opened count {n_opened} at trial {t}.")
            counts[n_opened] += 1

        empirical = {n: counts[n] / n_runs for n in range(6)}

        smoothed = {n: (empirical[n] if empirical[n] > 0 else eps) for n in range(6)}
        z = sum(smoothed.values())
        smoothed = {n: smoothed[n] / z for n in range(6)}

        dist[t] = smoothed

    return dist


def print_trial_distribution(
    folder_name: str,
    dist: Dict[int, Dict[int, float]],
    max_trials: int = MAX_TRIALS,
) -> None:
    print("\n" + "=" * 100)
    print(f"EMPIRICAL DISTRIBUTION P(n | t, theta) FOR: {folder_name}")
    print("=" * 100)

    header = "trial".ljust(8) + "".join([f"n={n}".rjust(14) for n in range(6)])
    print(header)
    print("-" * len(header))

    for t in range(1, max_trials + 1):
        row = f"{t}".ljust(8)
        for n in range(6):
            row += f"{dist[t][n]:14.6f}"
        print(row)

    print("=" * 100 + "\n")


def child_to_opened_counts(
    child_trials: List[List[Any]],
    max_trials: int = MAX_TRIALS,
) -> List[int]:
    """
    Convert one child's raw data into:
        [n_1, ..., n_70]
    where n_t is number of UNIQUE boxes opened by trial t.
    """
    opened_boxes = set()
    counts = []

    for trial in child_trials[:max_trials]:
        if not isinstance(trial, list) or len(trial) < 3:
            raise ValueError(f"Malformed child trial: {trial}")

        box_id = trial[1]
        outcome = int(trial[2])

        if outcome == 1 and box_id not in opened_boxes:
            opened_boxes.add(box_id)

        counts.append(len(opened_boxes))

    if len(counts) < max_trials:
        last = counts[-1] if counts else 0
        counts.extend([last] * (max_trials - len(counts)))

    return counts


def build_all_children_counts(
    kid_data: Dict[str, List[List[Any]]],
    max_trials: int = MAX_TRIALS,
) -> Dict[str, List[int]]:
    out = {}
    for child_id, trials in kid_data.items():
        out[child_id] = child_to_opened_counts(trials, max_trials=max_trials)
    return out


def compute_child_loglik(
    child_id: str,
    child_counts: List[int],
    dist: Dict[int, Dict[int, float]],
    max_trials: int = MAX_TRIALS,
) -> Dict[str, Any]:
    total_loglik = 0.0
    trial_details = []

    for t in range(1, max_trials + 1):
        n_jt = int(child_counts[t - 1])
        p = float(dist[t][n_jt])
        logp = math.log(p)
        total_loglik += logp

        trial_details.append(
            {
                "trial": t,
                "n_opened_child": n_jt,
                "p_n_given_t_theta": p,
                "log_prob": logp,
            }
        )

    return {
        "child_id": child_id,
        "opened_counts": child_counts,
        "loglik": total_loglik,
        "trial_details": trial_details,
    }


def population_summary(child_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    vals = [(cid, rec["loglik"]) for cid, rec in child_results.items()]
    mean_ll = sum(v for _, v in vals) / len(vals)

    best_child_id, best_ll = max(vals, key=lambda x: x[1])
    worst_child_id, worst_ll = min(vals, key=lambda x: x[1])

    total_ll = sum(v for _, v in vals)

    return {
        "num_children": len(vals),
        "mean_loglik": mean_ll,
        "total_loglik": total_ll,
        "best_child_id": best_child_id,
        "best_loglik": best_ll,
        "worst_child_id": worst_child_id,
        "worst_loglik": worst_ll,
    }


def process_one_folder(
    folder_path: Path,
    kid_data: Dict[str, List[List[Any]]],
    print_distribution: bool = False,
) -> Dict[str, Any]:
    trace_json_path = find_trace_json(folder_path)
    trace_data = load_json(trace_json_path)

    run_to_counts = extract_open_counts_per_run(trace_data, max_trials=MAX_TRIALS)
    dist = build_trial_distribution(run_to_counts, max_trials=MAX_TRIALS, eps=EPS)

    if print_distribution:
        print_trial_distribution(folder_path.name, dist, max_trials=MAX_TRIALS)

    child_to_counts_map = build_all_children_counts(kid_data, max_trials=MAX_TRIALS)

    child_results: Dict[str, Dict[str, Any]] = {}
    for child_id, child_counts in child_to_counts_map.items():
        result = compute_child_loglik(
            child_id=child_id,
            child_counts=child_counts,
            dist=dist,
            max_trials=MAX_TRIALS,
        )
        child_results[child_id] = result

    summary = population_summary(child_results)

    output = {
        "trace_dir": str(folder_path),
        "trace_json": str(trace_json_path),
        "kid_json": str(KID_JSON_PATH),
        "max_trials": MAX_TRIALS,
        "eps": EPS,
        "distribution_p_n_given_t": {
            f"trial_{t}": {str(n): dist[t][n] for n in range(6)}
            for t in range(1, MAX_TRIALS + 1)
        },
        "child_results": child_results,
        "population_summary": summary,
    }

    per_folder_out = folder_path / "population_fit_results.json"
    save_json(output, per_folder_out)

    return {
        "folder_name": folder_path.name,
        "folder_path": str(folder_path),
        "trace_json": str(trace_json_path),
        "num_runs": len(run_to_counts),
        "num_children": summary["num_children"],
        "mean_loglik": summary["mean_loglik"],
        "total_loglik": summary["total_loglik"],
        "best_child_id": summary["best_child_id"],
        "best_child_loglik": summary["best_loglik"],
        "worst_child_id": summary["worst_child_id"],
        "worst_child_loglik": summary["worst_loglik"],
        "per_folder_result_json": str(per_folder_out),
    }


def main() -> None:
    if not SWEEP_ROOT.exists():
        raise FileNotFoundError(f"Sweep root not found: {SWEEP_ROOT}")

    kid_data = load_json(KID_JSON_PATH)

    subfolders = sorted([p for p in SWEEP_ROOT.iterdir() if p.is_dir()])
    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in: {SWEEP_ROOT}")

    all_results = []
    failed_folders = []

    print(f"[INFO] Found {len(subfolders)} folders under {SWEEP_ROOT}")

    for i, folder in enumerate(subfolders, start=1):
        print(f"\n[INFO] Processing {i}/{len(subfolders)}: {folder.name}")
        try:
            result = process_one_folder(
                folder_path=folder,
                kid_data=kid_data,
                print_distribution=False,  # change to True if you want every folder printed
            )
            all_results.append(result)
            print(
                f"[OK] {folder.name} | total_loglik={result['total_loglik']:.6f} | "
                f"mean_loglik={result['mean_loglik']:.6f}"
            )
        except Exception as e:
            failed_folders.append(
                {
                    "folder_name": folder.name,
                    "folder_path": str(folder),
                    "error": str(e),
                }
            )
            print(f"[FAILED] {folder.name} | {e}")

    if not all_results:
        raise RuntimeError("No folders were processed successfully.")

    ranked_by_total = sorted(all_results, key=lambda x: x["total_loglik"], reverse=True)
    ranked_by_mean = sorted(all_results, key=lambda x: x["mean_loglik"], reverse=True)

    best_total = ranked_by_total[0]
    best_mean = ranked_by_mean[0]

    summary = {
        "sweep_root": str(SWEEP_ROOT),
        "kid_json": str(KID_JSON_PATH),
        "max_trials": MAX_TRIALS,
        "eps": EPS,
        "num_successful_folders": len(all_results),
        "num_failed_folders": len(failed_folders),
        "best_by_total_loglik": best_total,
        "best_by_mean_loglik": best_mean,
        "all_results_ranked_by_total_loglik": ranked_by_total,
        "failed_folders": failed_folders,
    }

    summary_path = SWEEP_ROOT / "sweep_population_fit_summary.json"
    save_json(summary, summary_path)

    print("\n" + "=" * 100)
    print("SWEEP COMPLETE")
    print("=" * 100)
    print(f"Successful folders: {len(all_results)}")
    print(f"Failed folders:     {len(failed_folders)}")
    print("\nMost positive TOTAL log-likelihood:")
    print(f"  Folder: {best_total['folder_name']}")
    print(f"  Value:  {best_total['total_loglik']:.6f}")
    print("\nMost positive MEAN log-likelihood:")
    print(f"  Folder: {best_mean['folder_name']}")
    print(f"  Value:  {best_mean['mean_loglik']:.6f}")
    print(f"\nSaved summary to: {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()