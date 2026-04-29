import math
import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"training_results\smc_trace_sweeps_redo\full_model_processed_grid.csv"
df = pd.read_csv(csv_path)

# Sort for clean plotting
df = df.sort_values(["skill", "prob_gen", "trueprior"])

skills = sorted(df["skill"].unique())
n = len(skills)

# layout
ncols = 4
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
axes = axes.flatten()

# shared color scale across all heatmaps
vmin = df["score"].min()
vmax = df["score"].max()

mappable = None

for i, skill in enumerate(skills):
    ax = axes[i]
    sub = df[df["skill"] == skill]

    # rows = prob_gen, cols = trueprior
    pivot = sub.pivot(index="prob_gen", columns="trueprior", values="score")
    pivot = pivot.sort_index().sort_index(axis=1)

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",   # lower = blue, higher = red
        vmin=vmin,
        vmax=vmax
    )
    mappable = im

    ax.set_title(f"skill = {skill:.3f}", fontsize=12)
    ax.set_xlabel("trueprior", fontsize=10)
    ax.set_ylabel("prob_gen", fontsize=10)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45, ha="right", fontsize=8)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.1f}" for y in pivot.index], fontsize=8)

# hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

# one shared colorbar
cbar = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.9)
cbar.set_label("score", fontsize=11)

plt.suptitle("Score heatmaps across skill slices", fontsize=16)
plt.tight_layout()
plt.show()