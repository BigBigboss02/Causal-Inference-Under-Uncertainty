import json
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
KIDS_PATH = r"data\Dolly_KeyEviModel_7.3.24.json"

MODEL_PATHS = {
    "SoC-L":    r"neurips_drafts\data\theta_90_1__gen_0p01__trueprior_0p02__particles_30__runs_100__trials_70\trial_major_particles_and_ig.json",
    "SoC-Rel":  r"neurips_drafts\data\theta_1_1__gen_0p01__trueprior_0p02__particles_30__runs_100__trials_70\trial_major_particles_and_ig.json",
    "SoC-Gen":  r"neurips_drafts\data\theta_90_1__gen_0p85__trueprior_0p02__particles_30__runs_100__trials_70\trial_major_particles_and_ig.json",
    "SoC-Full": r"neurips_drafts\data\theta_2_1__gen_0p4__trueprior_0p02__particles_30__runs_100__trials_70\trial_major_particles_and_ig.json",
}

# =========================
# Kids: real attempts-to-solve
# =========================
def load_kids_attempts_to_success(kids_path):
    with open(kids_path, "r") as f:
        data = json.load(f)

    attempts = []

    for child_id, trials in data.items():
        opened_boxes = set()
        solved_at = None

        for t, trial in enumerate(trials, start=1):
            key_id, box_id, outcome, _ = trial
            if outcome == 1 or outcome is True:
                opened_boxes.add(box_id)
            if len(opened_boxes) >= 5:
                solved_at = t
                break

        if solved_at is None:
            solved_at = len(trials)

        attempts.append(solved_at)

    return np.array(attempts)


# =========================
# Models: real attempts-to-solve from trial_major_trace
# =========================
def load_model_attempts_to_success(path):
    with open(path, "r") as f:
        data = json.load(f)

    trace = data["trial_major_trace"]

    # collect trial ids in sorted order
    trial_ids = []
    for trial_key in trace.keys():
        if trial_key.startswith("trial_"):
            trial_ids.append(int(trial_key.split("_")[1]))
    trial_ids = sorted(trial_ids)

    # collect run ids from first trial
    first_trial_key = f"trial_{trial_ids[0]}"
    run_keys = [rk for rk in trace[first_trial_key].keys() if rk.startswith("run_")]
    run_keys = sorted(run_keys, key=lambda x: int(x.split("_")[1]))

    attempts = []

    for run_key in run_keys:
        solved_at = None
        max_trial = 0

        for t in trial_ids:
            trial_key = f"trial_{t}"
            max_trial = t

            if run_key not in trace[trial_key]:
                continue

            run_data = trace[trial_key][run_key]
            opened_before = run_data.get("opened_before", 0)
            outcome = run_data.get("outcome", False)

            opened_after = opened_before + (1 if (outcome is True or outcome == 1) else 0)

            if opened_after >= 5:
                solved_at = t
                break

        if solved_at is None:
            solved_at = max_trial

        attempts.append(solved_at)

    return np.array(attempts)


# =========================
# Load real top-row data
# =========================
attempts_to_success = load_kids_attempts_to_success(KIDS_PATH)

model_names = ['SoC-L', 'SoC-Rel', 'SoC-Gen', 'SoC-Full']
sim_data_list = [load_model_attempts_to_success(MODEL_PATHS[name]) for name in model_names]

# =========================
# Pseudo bottom-row data
# =========================
model_color = '#4C72B0'
child_color = '#444444'
common_bins = np.linspace(0, 65, 22)

bar_specs = [
    {'labels': ['number'],
     'heights': [1.0]},
    {'labels': ['number'],
     'heights': [1.0]},
    {'labels': ['number', 'gen3', 'gen9', 'gen15', 'gen22'],
     'heights': [0.22, 0.40, 0.22, 0.10, 0.06]},
    {'labels': ['number', 'gen2', 'gen7', 'gen13', 'gen19', 'gen24'],
     'heights': [0.29, 0.30, 0.19, 0.12, 0.07, 0.03]},
]

# =========================
# Plot
# =========================
fig, axes = plt.subplots(
    2, 4,
    figsize=(16, 8),
    sharey='row',   # 🔥 key fix
    gridspec_kw={'height_ratios': [2.8, 1.8], 'hspace': 0.55, 'wspace': 0.12}
)

# ── top row: REAL histograms ────────────────────────────────────────────────
for i, (name, sim_data) in enumerate(zip(model_names, sim_data_list)):
    ax = axes[0, i]
    ax.hist(sim_data, bins=common_bins, alpha=0.65, color=model_color,
            edgecolor='white', linewidth=0.6,
            label='Model simulation' if i == 0 else '_nolegend_')
    ax.hist(attempts_to_success, bins=common_bins, alpha=0.45,
            color=child_color, edgecolor='white', linewidth=0.6,
            label='Children' if i == 0 else '_nolegend_')
    ax.set_title(name, fontsize=17, fontweight='bold', pad=10)
    ax.set_xlim(0, 65)
    ax.tick_params(axis='both', labelsize=13)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('N trials to solve', fontsize=13, labelpad=6)

axes[0, 0].set_ylabel('Number of participants', fontsize=14, labelpad=6)
axes[0, 0].legend(frameon=False, fontsize=13, loc='upper right',
                  handlelength=1.4, handletextpad=0.5)

# for ax in axes[0, 1:]:
#     ax.tick_params(left=False, labelleft=False)

# ── bottom row: PSEUDO bars ────────────────────────────────────────────────
CHILDREN_LINE = 0.22

for i, spec in enumerate(bar_specs):
    ax = axes[1, i]
    x = np.arange(len(spec['labels']))

    ax.bar(x, spec['heights'], color=model_color, alpha=0.75, width=0.55,
           edgecolor='white', linewidth=0.6)
    ax.axhline(CHILDREN_LINE, color='crimson', linestyle='--', linewidth=2.2,
               zorder=5, label='Children' if i == 0 else '_nolegend_')

    ax.set_xticks(x)
    ax.set_xticklabels(spec['labels'], fontsize=12, rotation=40, ha='right')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Generalization', fontsize=13, labelpad=6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=12)

    if i > 0:
        ax.tick_params(left=False, labelleft=False)
        ax.spines['left'].set_visible(False)

axes[1, 0].set_ylabel('Proportion', fontsize=14, labelpad=6)
axes[1, 0].legend(frameon=False, fontsize=13, loc='upper right')

plt.show()