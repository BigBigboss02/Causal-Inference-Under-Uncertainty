#expectaion of probabiulity calculated by simple algorithtic frequency

import json
import matplotlib.pyplot as plt
from collections import Counter


DATA_PATH = r"C:\Users\MSN\Documents\Python\smc-s\data\Dolly_KeyEviModel_7.3.24.json"


def cumulative_opened_boxes(trials):
    """
    For one child, compute the cumulative number of unique boxes opened
    after each trial.

    K_n = number of unique boxes successfully opened by trial n.

    This trajectory is monotone non-decreasing.
    """
    opened_boxes = set()
    trajectory = []

    for trial in trials:
        box_id = trial[1]
        outcome = int(trial[2])

        if outcome == 1:
            opened_boxes.add(box_id)

        trajectory.append(len(opened_boxes))

    return trajectory


def pad_trajectory_with_final_value(trajectory, max_trial):
    """
    Extend a child's trajectory to length max_trial by repeating
    the final value.

    This is the key mathematical fix:
    if a child stops early, their number of opened boxes should remain
    at the final achieved value, not disappear from later trials.
    """
    if len(trajectory) == 0:
        return [0] * max_trial

    if len(trajectory) >= max_trial:
        return trajectory[:max_trial]

    final_value = trajectory[-1]
    padding = [final_value] * (max_trial - len(trajectory))
    return trajectory + padding


def compute_probability_table(json_path, max_trial=100, max_boxes=5):
    """
    Compute empirical probabilities P(K_n = k) for n = 1..max_trial,
    where k = 0..max_boxes.

    IMPORTANT:
    Every child is included at every trial n by padding their trajectory
    with their final value. This preserves monotonicity properly.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        kid_data = json.load(f)

    # Build padded monotone trajectories for all children first
    all_trajectories = []

    for kid_id, trials in kid_data.items():
        traj = cumulative_opened_boxes(trials)
        padded_traj = pad_trajectory_with_final_value(traj, max_trial=max_trial)
        all_trajectories.append(padded_traj)

    total_kids = len(all_trajectories)
    prob_table = {}
    count_table = {}

    for trial_n in range(1, max_trial + 1):
        opened_counts = []

        for traj in all_trajectories:
            opened_counts.append(traj[trial_n - 1])

        freq = Counter(opened_counts)

        count_table[trial_n] = {
            k: freq.get(k, 0)
            for k in range(max_boxes + 1)
        }

        prob_table[trial_n] = {
            k: freq.get(k, 0) / total_kids if total_kids > 0 else 0.0
            for k in range(max_boxes + 1)
        }

    return prob_table, count_table, total_kids


def plot_probability_lines(prob_table, max_trial=100, max_boxes=5):
    """
    Plot P(K_n = k) for all k = 0..5.

    Since each child's K_n is monotone non-decreasing and padded forward,
    the probability mass should evolve sensibly over time.
    """
    trials = list(range(1, max_trial + 1))

    plt.figure(figsize=(10, 6))

    for k in range(0, max_boxes + 1):
        probs = [prob_table[n][k] for n in trials]
        plt.plot(trials, probs, marker="o", label=f"{k} boxes")

    plt.xlabel("Trial number (n)")
    plt.ylabel("Probability")
    plt.title("Empirical probability of opening k boxes by trial n")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_distribution_at_trial(prob_table, trial_n, max_boxes=5):
    """
    Plot the distribution over k at a fixed trial n.
    """
    x_vals = list(range(0, max_boxes + 1))
    y_vals = [prob_table[trial_n][k] for k in x_vals]

    plt.figure(figsize=(8, 5))
    plt.bar(x_vals, y_vals, width=0.8)
    plt.xlabel(f"Number of boxes opened by trial {trial_n}")
    plt.ylabel("Probability")
    plt.title(f"Distribution of K_{trial_n}")
    plt.xticks(x_vals)
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    prob_table, count_table, total_kids = compute_probability_table(
        json_path=DATA_PATH,
        max_trial=70,
        max_boxes=5
    )

    print(f"Total kids included at every trial: {total_kids}")
    print("Probabilities at trial 9:")
    print(prob_table[9])

    print("\nCounts at trial 9:")
    print(count_table[9])

    plot_probability_lines(
        prob_table=prob_table,
        max_trial=70,
        max_boxes=5
    )

    plot_distribution_at_trial(
        prob_table=prob_table,
        trial_n=9,
        max_boxes=5
    )