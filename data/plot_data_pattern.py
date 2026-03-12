import json
import matplotlib.pyplot as plt
from collections import Counter


DATA_PATH = r"C:\Users\MSN\Documents\Python\smc-s\data\Dolly_KeyEviModel_7.3.24.json"


def plot_kid_histograms(json_path: str, show: bool = True):
    with open(json_path, "r", encoding="utf-8") as f:
        kid_data = json.load(f)

    # -------- Plot 1: boxes opened per kid --------
    opened_counts = []

    # -------- Plot 2: trials needed to open all 5 boxes --------
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

        # record boxes opened
        opened_counts.append(len(opened_boxes))

        # record trials to 5
        if trial_count_to_5 is not None:
            success_kids += 1
            trials_needed_list.append(trial_count_to_5)

    # -------- Histogram 1 --------
    freq_boxes = Counter(opened_counts)

    x_vals = list(range(0, 6))
    y_vals = [freq_boxes.get(x, 0) for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.bar(x_vals, y_vals, width=0.8)
    plt.xlabel("Number of boxes opened")
    plt.ylabel("Number of kids")
    plt.xticks(x_vals)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    # -------- Histogram 2 --------
    if len(trials_needed_list) > 0:

        freq_trials = Counter(trials_needed_list)

        x_vals = list(range(min(freq_trials.keys()), max(freq_trials.keys()) + 1))
        y_vals = [freq_trials.get(x, 0) for x in x_vals]

        plt.figure(figsize=(8, 5))
        plt.bar(x_vals, y_vals, width=0.8)
        plt.xlabel("Trials needed to open all 5 boxes")
        plt.ylabel("Number of kids")
        plt.xticks(x_vals)
        plt.grid(axis="y", alpha=0.3)

        # only title
        plt.title(f"Kids opened 5/5 boxes: {success_kids}/{total_kids}")

        plt.tight_layout()

        if show:
            plt.show()

    return freq_boxes


if __name__ == "__main__":
    plot_kid_histograms(DATA_PATH, show=True)