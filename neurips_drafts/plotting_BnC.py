import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Load real CSV for Figure 1
# =========================
csv_path = r"neurips_drafts\data\BnC_data.csv"
df = pd.read_csv(csv_path)

# Use your renamed columns
models = ["SoC-Gen-L", "SoC-Rel", "SoC-Gen", "SoC-F"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# If best_model column already exists, use it
if "best_model" in df.columns:
    best_counts = df["best_model"].value_counts().reindex(models, fill_value=0)
else:
    # Otherwise infer best model from most positive log-likelihood
    best_model_idx = df[models].values.argmax(axis=1)
    best_counts = pd.Series(best_model_idx).value_counts().sort_index()
    best_counts = pd.Series(
        [best_counts.get(i, 0) for i in range(len(models))],
        index=models
    )

# --- 1. Model selection counts from real CSV ---
plt.figure(figsize=(5, 4))
plt.bar(models, best_counts.values, color=colors)
plt.ylabel("Number of children", fontsize=14)
plt.title("Best-fitting model counts", fontsize=16)
plt.xticks(rotation=20, fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()


models = ["SoC-Gen", "SoC-Rel", "SoC-F", "SoC-L"]
colors = ["tab:green", "tab:orange", "tab:red", "tab:blue"]

# handle naming mismatch
if "SoC-L" not in df.columns and "SoC-Gen-L" in df.columns:
    df["SoC-L"] = df["SoC-Gen-L"]

# =========================
# Build score matrix
# score = (-log-likelihood) / 1000
# =========================
logL = df[models].values
score = (-logL) / 1000.0

n_children = score.shape[0]
n_models = score.shape[1]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Load data
# =========================
csv_path = r"neurips_drafts\data\BnC_data.csv"
df = pd.read_csv(csv_path)

# Map your CSV column names to plotting names
if "SoC-L" not in df.columns and "SoC-Gen-L" in df.columns:
    df["SoC-L"] = df["SoC-Gen-L"]
if "SoC-Full" not in df.columns and "SoC-F" in df.columns:
    df["SoC-Full"] = df["SoC-F"]

# Strict plotting order to match first figure
models = ["SoC-L", "SoC-Rel", "SoC-Gen", "SoC-Full"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# =========================
# Build score matrix
# score = -log-likelihood
# =========================
logL = df[models].values
score = -logL

n_children = score.shape[0]
n_models = score.shape[1]

# Map your CSV names to plotting names
if "SoC-L" not in df.columns and "SoC-Gen-L" in df.columns:
    df["SoC-L"] = df["SoC-Gen-L"]
if "SoC-Full" not in df.columns and "SoC-F" in df.columns:
    df["SoC-Full"] = df["SoC-F"]

# Strict order to match target figure
models = ["SoC-L", "SoC-Rel", "SoC-Gen", "SoC-Full"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# =========================
# Build manipulated score
# =========================
logL = df[models].values

# choose ONE:
score = -logL
# score = (-logL) / 1000.0

n_children = score.shape[0]
n_models = score.shape[1]

# --- 2. Population score + bootstrap CI ---
mean_score = score.mean(axis=0)

# Bootstrap
n_boot = 1000
boot_means = np.zeros((n_boot, n_models))

for b in range(n_boot):
    sample_idx = np.random.choice(n_children, n_children, replace=True)
    sampled_score = score[sample_idx]
    boot_means[b] = sampled_score.mean(axis=0)

ci_lower = np.percentile(boot_means, 2.5, axis=0)
ci_upper = np.percentile(boot_means, 97.5, axis=0)
yerr = np.vstack([mean_score - ci_lower, ci_upper - mean_score])

plt.figure(figsize=(5, 4))
plt.bar(models, mean_score, yerr=yerr, capsize=5, color=colors)
plt.ylabel("Score (-log-likelihood)", fontsize=14)
plt.title("Population model fit (mean ± 95% CI)", fontsize=16)
plt.xticks(rotation=20, fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()