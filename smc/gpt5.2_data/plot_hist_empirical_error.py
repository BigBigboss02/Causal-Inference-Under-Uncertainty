"""
plot histogram for the empirical error rate of
supposed-to-open key-box trials due to fussy oracle

x-axis: error fraction of supposed-to-open actions
y-axis: # of experiment runs 
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_LOG_DIR = (
    Path(__file__).resolve().parent.parent
    / "gpt5.2_data"
    / "full_run_logs"
    / "gpt5.2_stochastic_0.6_oracle"
)
RUN_LOG_PREFIX = "llm_ps_stochastic_run_"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "hist_empirical_success.png"


def _load_runs_from_json_file(path: Path) -> list[dict]:
    """Return run dicts from one log file (aggregated ``runs`` or a single-run log)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        runs = data.get("runs")
        if isinstance(runs, list):
            return runs
        if "history" in data:
            return [data]
    raise ValueError(
        f"{path}: expected JSON object with 'runs' list or a single run with 'history'"
    )


def load_runs(path: Path, *, json_name_prefix: str | None = None) -> list[dict]:
    """
    Load all runs for an experiment path.

    If ``path`` is a file, load runs from that file only (``json_name_prefix`` ignored).
    If ``path`` is a directory, load each matching ``*.json`` (non-recursive), sorted by
    name. When ``json_name_prefix`` is non-empty, only files matching
    ``{json_name_prefix}*.json`` are included (e.g. prefix ``llm_ps_partially_obs_run_``
    picks ``..._run_1.json``, ``..._run_2.json``, ...). Empty string or ``None`` means
    all ``*.json`` in the directory.
    """
    path = Path(path)
    if path.is_file():
        return _load_runs_from_json_file(path)
    if path.is_dir():
        pattern = f"{json_name_prefix}*.json" if json_name_prefix else "*.json"
        all_runs: list[dict] = []
        for child in sorted(path.glob(pattern)):
            if not child.is_file():
                continue
            all_runs.extend(_load_runs_from_json_file(child))
        return all_runs
    raise FileNotFoundError(f"not a file or directory: {path}")


def empirical_success_rate(run: dict) -> float | None:
    history = run.get("history") or []
    supposed = sum(int(step.get("supposed_to_open", 0) or 0) for step in history)
    if supposed == 0:
        return None
    successes = sum(
        int(step.get("success", 0) or 0)
        for step in history
        if int(step.get("supposed_to_open", 0) or 0)
    )
    return successes / supposed


def plot_empirical_success_histogram(
    runs: list[dict],
    out_path: Path,
    *,
    source_label: str,
) -> bool:
    """
    Histogram of per-run empirical success rate (1 - empirical error on supposed-to-open).

    Each ``run`` should provide ``history`` with ``success`` and ``supposed_to_open`` on
    attempt steps (as produced by ``driver_sp_baseline.py`` enrichment or equivalent).
    """
    rates = []
    for run in runs:
        r = empirical_success_rate(run)
        if r is not None:
            rates.append(r)
        else:
            idx = run.get("run_idx", run.get("meta", {}).get("run_number", "?"))
            print(
                f"warning: run {idx} has zero supposed_to_open steps; skipped",
                file=sys.stderr,
            )

    if not rates:
        return False

    rates = np.asarray(rates, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, 11)

    fig, ax = plt.subplots(figsize=(9, 6))
    counts, _, _ = ax.hist(rates, bins=bin_edges, range=(0.0, 1.0), edgecolor="black", alpha=0.75)
    ax.set_xlabel("Empirical success rate (per run)")
    ax.set_ylabel("Number of runs")
    ax.set_title(f"empirical success rate\n({len(rates)} runs, {source_label})")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0, 1, 11))

    ymax = max(float(np.max(counts)), 1.0)
    ax.set_ylim(0.0, ymax * 1.05)

    rates_sorted = np.sort(rates)
    rates_str = np.array2string(
        rates_sorted,
        precision=5,
        separator=", ",
        max_line_width=96,
    )
    fig.text(
        0.5,
        0.02,
        f"Empirical success rates (sorted ascending):\n{rates_str}",
        ha="center",
        va="bottom",
        fontsize=8,
        family="monospace",
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def main() -> None:
    if len(sys.argv) > 1:
        history_path = Path(sys.argv[1])
        json_prefix = RUN_LOG_PREFIX if history_path.is_dir() else None
    else:
        history_path = EXPERIMENT_LOG_DIR
        json_prefix = RUN_LOG_PREFIX

    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    runs = load_runs(history_path, json_name_prefix=json_prefix)
    if history_path.is_dir():
        pattern = f"{json_prefix}*.json" if json_prefix else "*.json"
        n_json = sum(1 for p in history_path.glob(pattern) if p.is_file())
        prefix_note = f", prefix {json_prefix!r}" if json_prefix else ""
        source_label = (
            f"{history_path.name}/ ({n_json} json files, {len(runs)} runs{prefix_note})"
        )
    else:
        source_label = history_path.name
    if not plot_empirical_success_histogram(runs, out_path, source_label=source_label):
        print("no valid runs to plot", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
