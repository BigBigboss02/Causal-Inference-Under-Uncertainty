import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ===== CHANGE THIS PATH =====
data_dir = r"C:\Users\MSN\Documents\Python\smc-s\training_results\nips_qwen"

# ============================

trials_per_run = []

for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_dir, filename)

        df = pd.read_csv(file_path)

        # number of trials = number of rows
        trials_per_run.append(len(df))

print("Collected trials per run:", trials_per_run)

# ===== histogram =====
counts = Counter(trials_per_run)
x = sorted(counts.keys())
y = [counts[v] for v in x]

plt.figure(figsize=(8, 5))
plt.bar(x, y)

plt.xlabel("Trial number to open 5 boxes")
plt.ylabel("Number of runs")
plt.title("Distribution of trials needed across runs")

plt.tight_layout()

save_path = os.path.join(data_dir, "trials_histogram.png")
plt.savefig(save_path, dpi=200)
plt.show()

print(f"Saved plot to: {save_path}")