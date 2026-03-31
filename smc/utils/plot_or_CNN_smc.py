import json
import os
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
KID_DATA_PATH = r"C:\Users\MSN\Documents\Python\smc-s\data\Dolly_KeyEviModel_7.3.24.json"

MODEL_JSON_PATH = r"training_results\CCN_plots_data\smc_s\theta_2_2__gen_0p6__priororder_0p05__runs_1000__trials_70__20260328_092157\theta_2_2__gen_0p6__priororder_0p05__runs_1000__trials_70__20260328_092157.json"

OUTPUT_PLOT_PATH = r"C:\Users\MSN\Documents\Python\smc-s\training_results\CCN_plots_data\smc_s\theta_2_2__gen_0p6__priororder_0p05__runs_1000__trials_70__20260328_092157\theta_2_2__gen_0p6__priororder_0p05__runs_1000__trials_70__20260328_092157_overlay.png"

X_MAX_DISPLAY = 70
MODEL_SCALE = 0.1   # normalize 1000 model runs to ~100-child scale


# -----------------------------
# Child data loader
# -----------------------------
def load_kid_trials_needed(json_path: str):
    """
    Expects:
        {
            "kid_id_1": [[...], [...], ...],
            "kid_id_2": [[...], [...], ...],
            ...
        }

    Each trial is assumed to have:
        trial[1] = box_id
        trial[2] = outcome (0 or 1)

    Returns:
        trials_needed_list: list of trial counts for kids who opened all 5 boxes
        success_kids: number of kids who opened all 5 boxes
        total_kids: total number of kids
    """
    with open(json_path, "r", encoding="utf-8") as f:
        kid_data = json.load(f)

    trials_needed_list = []
    total_kids = len(kid_data)
    success_kids = 0

    for kid_id, trials in kid_data.items():
        opened_boxes = set()
        trial_count_to_5 = None

        for i, trial in enumerate(trials, start=1):
            box_id = trial[1]
            outcome = int(trial[2])

            if outcome == 1:
                opened_boxes.add(box_id)

            if len(opened_boxes) == 5:
                trial_count_to_5 = i
                break

        if trial_count_to_5 is not None:
            success_kids += 1
            trials_needed_list.append(trial_count_to_5)

    return trials_needed_list, success_kids, total_kids


# -----------------------------
# SMC-S model data loader
# -----------------------------
def load_model_trials_needed(json_path: str):
    """
    Expected SMC-S JSON format:
        {
          ...
          "attempt_counts": [...],
          "num_runs": 1000,
          ...
        }

    Returns:
        attempt_counts
        num_runs
        raw_data
    """
    with open(json_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    if "attempt_counts" not in model_data:
        raise ValueError("Expected 'attempt_counts' in model JSON.")

    attempt_counts = [int(x) for x in model_data["attempt_counts"]]
    num_runs = int(model_data.get("num_runs", len(attempt_counts)))

    return attempt_counts, num_runs, model_data


# -----------------------------
# Overlay plot
# -----------------------------
def plot_kids_vs_model(
    kid_trials_needed,
    success_kids,
    total_kids,
    model_attempt_counts,
    num_runs,
    save_path,
    model_title="SMC-S"
):
    # clip to display range
    kid_plot_data = [x for x in kid_trials_needed if x <= X_MAX_DISPLAY]
    model_plot_data = [x for x in model_attempt_counts if x <= X_MAX_DISPLAY]

    if len(kid_plot_data) == 0:
        raise ValueError("No child data left after clipping.")
    if len(model_plot_data) == 0:
        raise ValueError("No model data left after clipping.")

    bins = range(0, X_MAX_DISPLAY + 2)

    # fixed 1/10 normalization as requested
    model_weights = [MODEL_SCALE] * len(model_plot_data)

    plt.figure(figsize=(3.1, 2.8))

    # children: gray
    plt.hist(
        kid_plot_data,
        bins=bins,
        color="#9e9e9e",
        edgecolor="#9e9e9e",
        alpha=0.95
    )

    # model: orange
    plt.hist(
        model_plot_data,
        bins=bins,
        weights=model_weights,
        color="#f4a259",
        edgecolor="#f4a259",
        alpha=0.90
    )

    # title
    plt.title(model_title, fontsize=12)

    # axes
    plt.xlim(0, X_MAX_DISPLAY)
    plt.xticks([0, 20, 40, 60], fontsize=8)
    plt.yticks(fontsize=8)

    # clean style
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    kid_trials_needed, success_kids, total_kids = load_kid_trials_needed(KID_DATA_PATH)
    model_attempt_counts, num_runs, model_data = load_model_trials_needed(MODEL_JSON_PATH)

    print(f"Child data: {success_kids}/{total_kids} kids opened all 5 boxes")
    print(f"Model attempt counts used: {len(model_attempt_counts)}")
    print(f"Model num_runs in JSON: {num_runs}")
    print(f"Model normalization scale applied: {MODEL_SCALE}")

    plot_kids_vs_model(
        kid_trials_needed=kid_trials_needed,
        success_kids=success_kids,
        total_kids=total_kids,
        model_attempt_counts=model_attempt_counts,
        num_runs=num_runs,
        save_path=OUTPUT_PLOT_PATH,
        model_title="SMC-S"
    )

    print(f"Saved overlay plot to:\n{OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()