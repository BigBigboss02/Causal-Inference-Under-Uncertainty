import json
import matplotlib.pyplot as plt
from collections import Counter


DATA_PATH = r"C:\Users\MSN\Documents\Python\smc-s\data\Dolly_KeyEviModel_7.3.24.json"


def plot_boxes_opened_histogram(json_path: str, show: bool = True):
    with open(json_path, "r", encoding="utf-8") as f:
        kid_data = json.load(f)

    opened_counts = []

    for kid_id, trials in kid_data.items():
        opened_boxes = set()

        for trial in trials:
            box_id = trial[1]
            outcome = int(trial[2])

            if outcome == 1:
                opened_boxes.add(box_id)

        opened_counts.append(len(opened_boxes))

    freq = Counter(opened_counts)

    x_vals = list(range(0, 6))
    y_vals = [freq.get(x, 0) for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.bar(x_vals, y_vals, width=0.8)
    plt.xlabel("Number of boxes opened")
    plt.ylabel("Number of kids")
    plt.title("Histogram of boxes opened across kids")
    plt.xticks(x_vals)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return freq


if __name__ == "__main__":
    freq = plot_boxes_opened_histogram(DATA_PATH, show=True)
    print("Boxes opened frequency:")
    print(dict(sorted(freq.items())))