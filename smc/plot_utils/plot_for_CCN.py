import json
import os
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
KID_DATA_PATH = r"C:\Users\MSN\Documents\Python\smc-s\data\Dolly_KeyEviModel_7.3.24.json"
MODEL_JSON_PATH = r"C:\Users\MSN\Documents\Python\smc-s\training_results\LLM_results\qwen_plus\spbaseline_histogram_results.json"
OUTPUT_PLOT_PATH = r"C:\Users\MSN\Documents\Python\smc-s\training_results\LLM_results\qwen_plus\spbaseline_kids_overlay.png"

X_MAX_DISPLAY = 70


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

            if len(opened_boxes) == 5 and trial_count_to_5 is None:
                trial_count_to_5 = i
                break

        if trial_count_to_5 is not None:
            success_kids += 1
            trials_needed_list.append(trial_count_to_5)

    return trials_needed_list, success_kids, total_kids


# -----------------------------
# Model data loader
# -----------------------------
def load_model_trials_needed(json_path: str):
    """
    Supports several model JSON formats.

    Priority:
    1. attempt_counts: direct list of completion trial counts
    2. runs: use solved=True and read trials
    3. histogram: expand {"6":14, "7":28, ...} into [6,6,...,7,7,...]

    Returns:
        trials_needed_list
        num_runs
        raw_data
    """
    with open(json_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    # Case 1: old histogram experiment format
    if "attempt_counts" in model_data:
        attempt_counts = model_data.get("attempt_counts", [])
        num_runs = model_data.get("num_runs", len(attempt_counts))
        return attempt_counts, num_runs, model_data

    # Case 2: LLM result format with runs
    if "runs" in model_data:
        runs = model_data.get("runs", [])
        attempt_counts = [
            int(run["trials"])
            for run in runs
            if run.get("solved", False) and "trials" in run
        ]
        num_runs = model_data.get("num_runs", len(runs))
        return attempt_counts, num_runs, model_data

    # Case 3: fallback to histogram dict
    if "histogram" in model_data:
        hist = model_data.get("histogram", {})
        attempt_counts = []
        for k, v in hist.items():
            trial_num = int(k)
            count = int(v)
            attempt_counts.extend([trial_num] * count)

        num_runs = model_data.get("num_runs", len(attempt_counts))
        return attempt_counts, num_runs, model_data

    raise ValueError("Unsupported model JSON format.")


# -----------------------------
# Overlay plot
# -----------------------------
def plot_kids_vs_model(
    kid_trials_needed,
    success_kids,
    total_kids,
    model_attempt_counts,
    num_runs,
    save_path
):
    kid_plot_data = [x for x in kid_trials_needed if x <= X_MAX_DISPLAY]
    model_plot_data = [x for x in model_attempt_counts if x <= X_MAX_DISPLAY]

    if len(kid_plot_data) == 0:
        raise ValueError("No child data left after clipping.")
    if len(model_plot_data) == 0:
        raise ValueError("No model data left after clipping.")
    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")

    bins = range(0, X_MAX_DISPLAY + 2)

    # normalize model to child scale
    scale_factor = total_kids / num_runs
    model_weights = [scale_factor] * len(model_plot_data)

    plt.figure(figsize=(3.1, 2.8))  # compact like your example

    # --- children: gray ---
    plt.hist(
        kid_plot_data,
        bins=bins,
        color="#9e9e9e",
        edgecolor="#9e9e9e",
        alpha=0.95
    )

    # --- model: orange ---
    plt.hist(
        model_plot_data,
        bins=bins,
        weights=model_weights,
        color="#f4a259",
        edgecolor="#f4a259",
        alpha=0.90
    )

    # --- title ---
    plt.title("Qwen-3.5 Plus", fontsize=12)

    # --- axes ---
    plt.xlim(0, X_MAX_DISPLAY)
    plt.xticks([0, 20, 40, 60])
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)

    # clean style (like your screenshot)
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
    print(f"Model solved runs counted for histogram: {len(model_attempt_counts)}")
    print(f"Model num_runs: {num_runs}")

    plot_kids_vs_model(
        kid_trials_needed=kid_trials_needed,
        success_kids=success_kids,
        total_kids=total_kids,
        model_attempt_counts=model_attempt_counts,
        num_runs=num_runs,
        save_path=OUTPUT_PLOT_PATH
    )

    print(f"Saved overlay plot to:\n{OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()