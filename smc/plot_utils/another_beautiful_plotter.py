import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# --------------------------
# PATHS
# --------------------------
KID_JSON_PATH = Path(r"data\Dolly_KeyEviModel_7.3.24.json")
TRACE_DIR = Path(
    r"training_results\smc_trace_sweeps\small_true_prior\theta_2_1__gen_0p4__trueprior_0p02__particles_30__runs_100__trials_70"
)


# --------------------------
# LOAD JSON
# --------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------
# FIND TRACE FILE
# --------------------------
def find_trace_json(folder):
    for p in folder.glob("*.json"):
        data = load_json(p)
        if "trial_major_trace" in data:
            return p
    raise ValueError("No valid trace file found")


# --------------------------
# KIDS: trials to solve
# --------------------------
def get_kids_trials_to_solve(kid_data):
    trials_to_solve = []

    for child_id, trials in kid_data.items():
        opened_boxes = set()
        solved_trial = None

        for i, trial in enumerate(trials):
            box_id = trial[1]
            outcome = int(trial[2])

            if outcome == 1:
                opened_boxes.add(box_id)

            if len(opened_boxes) == 5:
                solved_trial = i + 1
                break

        if solved_trial is None:
            solved_trial = len(trials)

        trials_to_solve.append(solved_trial)

    return trials_to_solve


# --------------------------
# MODEL: trials to solve
# --------------------------
def get_model_trials_to_solve(trace_data):
    trial_major_trace = trace_data["trial_major_trace"]

    run_solved = {}

    for trial_key, runs in trial_major_trace.items():
        t = int(trial_key.split("_")[-1])

        for run_key, rec in runs.items():
            opened_before = rec["opened_before"]
            outcome = rec["outcome"]

            opened_after = opened_before + (1 if outcome else 0)

            if run_key not in run_solved:
                if opened_after == 5:
                    run_solved[run_key] = t

    # fill unfinished runs
    max_trial = 70
    trials_to_solve = []

    for run_key in run_solved:
        trials_to_solve.append(run_solved[run_key])

    return trials_to_solve


# --------------------------
# BEAUTIFUL PLOT
# --------------------------
def plot_histogram(kids_trials, model_trials):
    plt.figure(figsize=(10, 6))

    bins = np.arange(0, 71, 1)

    # Kids
    plt.hist(
        kids_trials,
        bins=bins,
        alpha=0.8,
        label="Children",
        edgecolor="white",
    )

    # Model (scaled)
    weights = np.ones_like(model_trials) # normalize
    plt.hist(
        model_trials,
        bins=bins,
        weights=weights,
        alpha=0.5,
        label="Model",
        edgecolor="white",
    )

    # Styling
    plt.title("Trials to Solve Distribution", fontsize=18, weight="bold")
    plt.xlabel("Number of Trials", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    plt.xlim(0, 70)

    plt.grid(alpha=0.2)
    plt.legend()

    plt.tight_layout()
    plt.show()


# --------------------------
# MAIN
# --------------------------
def main():
    kid_data = load_json(KID_JSON_PATH)
    trace_path = find_trace_json(TRACE_DIR)
    trace_data = load_json(trace_path)

    kids_trials = get_kids_trials_to_solve(kid_data)
    model_trials = get_model_trials_to_solve(trace_data)

    plot_histogram(kids_trials, model_trials)


if __name__ == "__main__":
    main()