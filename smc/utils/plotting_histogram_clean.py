import json
import os
import matplotlib.pyplot as plt


KID_DATA_PATH = r"C:\Users\MSN\Documents\Python\smc-s\data\Dolly_KeyEviModel_7.3.24.json"
ROOT_EXPERIMENT_DIR = r"C:\Users\MSN\Documents\Python\smc-s\training_results\histogram_method"
OUTPUT_DIR = os.path.join(ROOT_EXPERIMENT_DIR, "overlay_plots_normalised")

X_MAX_DISPLAY = 70


def load_kid_trials_needed(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        kid_data = json.load(f)

    trials_needed_list = []
    total_kids = len(kid_data)
    success_kids = 0

    for _, trials in kid_data.items():
        opened_boxes = set()
        trial_count_to_5 = None

        for i, trial in enumerate(trials, start=1):
            box_id = trial[1]
            outcome = int(trial[2])

            if outcome == 1:
                opened_boxes.add(box_id)

            if len(opened_boxes) == 5 and trial_count_to_5 is None:
                trial_count_to_5 = i

        if trial_count_to_5 is not None:
            success_kids += 1
            trials_needed_list.append(trial_count_to_5)

    return trials_needed_list, success_kids, total_kids


def load_model_histogram_data(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    attempt_counts = model_data.get("attempt_counts", [])
    num_runs = model_data.get("num_runs", None)

    return attempt_counts, num_runs, model_data


def is_experiment_json(json_path: str):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and "attempt_counts" in data and "num_runs" in data
    except Exception:
        return False


def should_process_json(json_path: str):
    """
    Only process json files that are inside a real experiment subfolder:
    .../<batch_folder>/<experiment_folder>/<experiment_file>.json

    This avoids duplicate processing of any loose json files directly under batch folders.
    """
    rel_path = os.path.relpath(json_path, ROOT_EXPERIMENT_DIR)
    parts = rel_path.split(os.sep)

    # Need at least:
    # batch_folder / experiment_folder / file.json
    if len(parts) < 3:
        return False

    return True


def make_output_path(model_json_path: str, root_input_dir: str, root_output_dir: str):
    rel_path = os.path.relpath(model_json_path, root_input_dir)
    rel_dir = os.path.dirname(rel_path)
    base_name = os.path.splitext(os.path.basename(model_json_path))[0]

    out_dir = os.path.join(root_output_dir, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    return os.path.join(out_dir, f"{base_name}_overlay_normalised.png")


def plot_one_overlay(
    kid_trials_needed,
    success_kids,
    total_kids,
    model_attempt_counts,
    num_runs,
    save_path
):
    if len(kid_trials_needed) == 0:
        print(f"Skipped {save_path}: no kids opened all 5 boxes.")
        return

    if len(model_attempt_counts) == 0:
        print(f"Skipped {save_path}: model file has no attempt_counts.")
        return

    if not num_runs or num_runs <= 0:
        print(f"Skipped {save_path}: invalid num_runs = {num_runs}")
        return

    # Clip both datasets to display range
    kid_plot_data = [x for x in kid_trials_needed if x <= X_MAX_DISPLAY]
    model_plot_data = [x for x in model_attempt_counts if x <= X_MAX_DISPLAY]

    if len(kid_plot_data) == 0 or len(model_plot_data) == 0:
        print(f"Skipped {save_path}: no data left after x-axis clipping.")
        return

    x_min = min(min(kid_plot_data), min(model_plot_data))
    bins = range(x_min, X_MAX_DISPLAY + 2)

    # normalise model histogram from num_runs scale to total_kids scale
    scale_factor = total_kids / num_runs
    model_weights = [scale_factor] * len(model_plot_data)

    plt.figure(figsize=(4.2, 3.6))

    # blue filled histogram: model
    plt.hist(
        model_plot_data,
        bins=bins,
        weights=model_weights,
        color="#c6e3f1",
        edgecolor="gray",
        alpha=0.9
    )

    # red step histogram: kids
    plt.hist(
        kid_plot_data,
        bins=bins,
        histtype="step",
        color="red",
        linewidth=1.2
    )

    plt.xlabel("Trials needed to open all 5 boxes")
    plt.ylabel("Count")
    plt.title(f"Kids opened 5/5 boxes: {success_kids}/{total_kids}")
    plt.xlim(x_min, X_MAX_DISPLAY)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_experiment_files():
    kid_trials_needed, success_kids, total_kids = load_kid_trials_needed(KID_DATA_PATH)

    total_found = 0
    total_saved = 0

    for dirpath, _, filenames in os.walk(ROOT_EXPERIMENT_DIR):
        if os.path.abspath(dirpath).startswith(os.path.abspath(OUTPUT_DIR)):
            continue

        for filename in filenames:
            if not filename.lower().endswith(".json"):
                continue

            json_path = os.path.join(dirpath, filename)

            if not should_process_json(json_path):
                continue

            if not is_experiment_json(json_path):
                continue

            total_found += 1

            try:
                model_attempt_counts, num_runs, _ = load_model_histogram_data(json_path)
                save_path = make_output_path(json_path, ROOT_EXPERIMENT_DIR, OUTPUT_DIR)

                plot_one_overlay(
                    kid_trials_needed=kid_trials_needed,
                    success_kids=success_kids,
                    total_kids=total_kids,
                    model_attempt_counts=model_attempt_counts,
                    num_runs=num_runs,
                    save_path=save_path
                )

                total_saved += 1
                print(f"Saved: {save_path}")

            except Exception as e:
                print(f"Failed on {json_path}: {e}")

    print(f"\nDone. Found {total_found} experiment json files, saved {total_saved} plots.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    plot_all_experiment_files()