"""
Histogram of per-run empirical success rate from sp_baseline_history.json.

Empirical success rate for one run:
    (# of steps with supposed_to_open and success) / (total supposed_to_open)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY = SCRIPT_DIR / "sp_baseline_history.json"
DEFAULT_OUT = SCRIPT_DIR / "hist_empirical_success.png"


def load_runs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["runs"]


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
    history_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HISTORY
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    runs = load_runs(history_path)
    if not plot_empirical_success_histogram(runs, out_path, source_label=history_path.name):
        print("no valid runs to plot", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
