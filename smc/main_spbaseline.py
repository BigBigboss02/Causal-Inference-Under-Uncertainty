from collections import Counter
import json
import os

from environment import Environment
from sp_baseline import SPBaseline
from utils.logger import Logger
from utils.plotter import SPBaselineHistogramPlotter


if __name__ == '__main__':
    num_runs = 40
    max_trials = 70
    output_dir = r"training_results\LLM_results\qwen_plus"
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "spbaseline_histogram_results.json")
    plot_path = os.path.join(output_dir, "spbaseline_trials_histogram.png")
    log_path = os.path.join(output_dir, "spbaseline_progress.log")

    solved_trials = []
    all_results = []
    start_run_idx = 0

    # Resume from existing checkpoint if available
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)

            all_results = existing_summary.get("runs", [])
            start_run_idx = len(all_results)

            solved_trials = [
                run["trials"]
                for run in all_results
                if run.get("solved", False)
            ]

            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"[RESUME] Loaded checkpoint with {start_run_idx} completed runs.\n"
                )

            print(f"Resuming from run_idx={start_run_idx}")

        except Exception as e:
            print(f"Failed to load existing checkpoint. Starting fresh. Error: {e}")
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"[WARNING] Failed to load checkpoint. Starting fresh. Error: {e}\n"
                )
            solved_trials = []
            all_results = []
            start_run_idx = 0

    # Run remaining experiments
    for run_idx in range(start_run_idx, num_runs):
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[START] run_idx={run_idx}\n")

        env = Environment()

        try:
            logger = Logger(False)
        except TypeError:
            logger = Logger()

        model = SPBaseline(env, logger)
        result = model.run(max_trials=max_trials)

        run_record = {
            "run_idx": run_idx,
            "solved": result["solved"],
            "trials": result["trials"],
            "opened": result["opened"],
            "success_pairs": result["success_pairs"],
        }
        all_results.append(run_record)

        if result["solved"]:
            solved_trials.append(result["trials"])

        # Save checkpoint after every completed run
        hist = Counter(solved_trials)
        partial_summary = {
            "model": "SPBaseline",
            "num_runs": num_runs,
            "max_trials": max_trials,
            "num_completed_runs": len(all_results),
            "num_solved": len(solved_trials),
            "num_unsolved_so_far": len(all_results) - len(solved_trials),
            "num_remaining_runs": num_runs - len(all_results),
            "histogram": {str(k): v for k, v in sorted(hist.items())},
            "runs": all_results,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(partial_summary, f, indent=4)

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"[DONE] run_idx={run_idx}, solved={result['solved']}, "
                f"trials={result['trials']}, checkpoint_saved={json_path}\n"
            )

        print(f"Completed run {run_idx + 1}/{num_runs}")

    # Final summary
    hist = Counter(solved_trials)

    summary = {
        "model": "SPBaseline",
        "num_runs": num_runs,
        "max_trials": max_trials,
        "num_completed_runs": len(all_results),
        "num_solved": len(solved_trials),
        "num_unsolved": len(all_results) - len(solved_trials),
        "histogram": {str(k): v for k, v in sorted(hist.items())},
        "runs": all_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Plot at the end
    plotter = SPBaselineHistogramPlotter(
        histogram=summary["histogram"],
        num_runs=summary["num_completed_runs"],
        max_trials=max_trials,
    )
    plotter.plot_trials_histogram(
        title="Trials to solve across runs",
        save_path=plot_path,
        show=True,
    )

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"[FINISH] All available runs completed. Final JSON: {json_path}, Plot: {plot_path}\n"
        )

    print(f"Saved JSON to: {json_path}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved log to: {log_path}")