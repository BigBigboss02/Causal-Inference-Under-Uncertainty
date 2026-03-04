import json
from typing import Dict, List, Tuple

from trainer2 import BoxTaskTrainer  # adjust import path if needed


# -----------------------------------------------------------------------------
# Pseudo data (kept for sanity testing)
# Each trial: (key_id, box_id, outcome_int, match_flags)
# -----------------------------------------------------------------------------
PSEUDO_DATA: Dict[str, List[Tuple[str, str, int, Dict[str, int]]]] = {
    "D001": [
        ("red", "red", 0, {"color": 1, "number": 0, "shape": 0}),
        ("pink", "pink", 0, {"color": 1, "number": 0, "shape": 0}),
        ("grey2", "pink", 1, {"color": 0, "number": 0, "shape": 0}),
        ("purple", "purple", 0, {"color": 1, "number": 0, "shape": 0}),
        ("green3", "purple", 1, {"color": 0, "number": 0, "shape": 0}),
        ("blue", "blue", 0, {"color": 1, "number": 0, "shape": 0}),
        ("yellow5", "blue", 1, {"color": 0, "number": 0, "shape": 0}),
        ("white", "white", 0, {"color": 1, "number": 0, "shape": 0}),
        ("orange4", "white", 1, {"color": 0, "number": 0, "shape": 0}),
    ],
}


def to_trainer_D_from_flags(
    child_trials: List[Tuple[str, str, int, Dict[str, int]]]
) -> List[Tuple[str, str, int]]:
    """Strip match flags to match BoxTaskTrainer input: (key_id, box_id, outcome_int)."""
    return [(k, b, o) for (k, b, o, _flags) in child_trials]


def load_json_dataset(json_path: str) -> Dict[str, List[List]]:
    """
    Your JSON file stores:
        { "ID": [ [key, box, outcome, {flags}], ... ], ... }
    JSON turns tuples into lists automatically, so we read lists.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_trainer_D_from_json_rows(rows: List[List]) -> List[Tuple[str, str, int]]:
    """
    Convert JSON rows like:
        [key, box, outcome, {flags}]
    into trainer D:
        (key, box, outcome)
    """
    D = []
    for r in rows:
        # r = [key, box, outcome, flags]
        k = str(r[0]).strip().lower()
        b = str(r[1]).strip().lower()
        o = int(r[2])
        D.append((k, b, o))
    return D


# -----------------------------------------------------------------------------
# SETTINGS (choose which data source and which IDs to train)
# -----------------------------------------------------------------------------
USE_JSON_DATA = True  # set False to use PSEUDO_DATA
JSON_PATH = r"C:\Users\MSN\Documents\Python\smc-s\data\Dolly_KeyEviModel_7.3.24.json"
TRAIN_IDS: List[str] = ["D001"]


# -----------------------------------------------------------------------------
# Fixed inference configuration (not being trained here)
# -----------------------------------------------------------------------------
SEED = 0
NUM_PARTICLES = 50
ESS_THRESHOLD = 0.5
K_REJUVENATE = 1
HOLDOUT_FRAC = 0.2  # keep your trainer's current 80/20 behavior unless you changed it

# -----------------------------------------------------------------------------
# Hyperparameter candidate lists (length = 3 each)
# -----------------------------------------------------------------------------
ALPHA_LIST = [0.5, 1.0, 2.0]
BETA_LIST = [0.5, 1.0, 2.0]
PRIOR_COLOR_LIST = [0.5, 2.0, 5.0]
PRIOR_ORDER_LIST = [1]
PRIOR_SHAPE_LIST = [0.5, 2.0, 5.0]
PRIOR_NUMBER_LIST = [0.5, 2.0, 5.0]
PRIOR_SIM_COLOR_TOTAL_LIST = [1]

PROP_RANDOM_LIST = [0.0]  # fixed unless you want [0.0, 0.1, 0.3]


def train_one(child_id: str, D: List[Tuple[str, str, int]]) -> None:
    trainer = BoxTaskTrainer(
        D=D,
        num_particles=NUM_PARTICLES,
        seed=SEED,
        ess_threshold=ESS_THRESHOLD,
        k_rejuvenate=K_REJUVENATE,
        holdout_frac=HOLDOUT_FRAC,
    )

    best_params, best_ll = trainer.grid_search_fit(
        alpha_list=ALPHA_LIST,
        beta_list=BETA_LIST,
        prior_color_list=PRIOR_COLOR_LIST,
        prior_order_list=PRIOR_ORDER_LIST,
        prior_shape_list=PRIOR_SHAPE_LIST,
        prior_number_list=PRIOR_NUMBER_LIST,
        prior_sim_color_total_list=PRIOR_SIM_COLOR_TOTAL_LIST,
        prop_random_list=PROP_RANDOM_LIST,
    )

    print(f"[GRID SEARCH] child={child_id}")
    print("  best_params:", best_params)
    print(f"  best_ll : {best_ll:.6f}   (higher is better)")
    print()


def main():
    if USE_JSON_DATA:
        dataset = load_json_dataset(JSON_PATH)
        all_ids = sorted(dataset.keys())

        if not TRAIN_IDS:
            print("[INFO] TRAIN_IDS is empty. Available IDs (first 30):", all_ids[:30])
            print("[INFO] Set TRAIN_IDS to one or more IDs from the list.")
            return

        for cid in TRAIN_IDS:
            if cid not in dataset:
                raise KeyError(f"ID '{cid}' not found in JSON. Example IDs: {all_ids[:10]}")
            D = to_trainer_D_from_json_rows(dataset[cid])
            train_one(cid, D)

    else:
        # Use pseudo data
        for cid, rows in PSEUDO_DATA.items():
            D = to_trainer_D_from_flags(rows)
            train_one(cid, D)


if __name__ == "__main__":
    main()