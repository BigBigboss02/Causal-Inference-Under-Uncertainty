import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter


def _compose_title_with_run_config(
    base: str,
    alpha0: Optional[float] = None,
    beta0: Optional[float] = None,
    prop_random: Optional[float] = None,
    true_prior: Optional[float] = None,
) -> str:
    """Append alpha0/beta0, prop_random, and true_prior to the plot title when provided."""
    meta: List[str] = []
    if alpha0 is not None and beta0 is not None:
        meta.append(f"alpha0={alpha0}, beta0={beta0}")
    if prop_random is not None:
        meta.append(f"prop_random={prop_random}")
    if true_prior is not None:
        meta.append(f"true_prior={true_prior}")
    if not meta:
        return base
    return base + "\n" + " · ".join(meta)


def _apply_small_xaxis_labels(ax, fontsize: float = 6.0) -> None:
    ax.tick_params(axis="x", labelsize=fontsize)


class Plotter2:
    def __init__(self, history: List[Dict[str, Any]]):
        if not history:
            raise ValueError("history is empty")
        self.history = history

    def _get_ts(self) -> List[float]:
        ts = []
        for i, h in enumerate(self.history):
            ts.append(h.get("t", i + 1))
        return ts

    def plot_boxes_opened_over_trials(
        self,
        title: str = "Boxes opened vs trials",
        save_path: Optional[str] = None,
        show: bool = True,
        alpha0: Optional[float] = None,
        beta0: Optional[float] = None,
        prop_random: Optional[float] = None,
        true_prior: Optional[float] = None,
        x_tick_labelsize: float = 6.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        ts = self._get_ts()
        opened = [h.get("opened", 0) for h in self.history]

        fig, ax = plt.subplots()
        ax.plot(ts, opened, marker="o")
        ax.set_title(
            _compose_title_with_run_config(title, alpha0, beta0, prop_random, true_prior)
        )
        ax.set_xlabel("Trial")
        ax.set_ylabel("Opened boxes")
        _apply_small_xaxis_labels(ax, x_tick_labelsize)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    def plot_theta_over_trials(
        self,
        title: str = "E[theta] over trials",
        save_path: Optional[str] = None,
        show: bool = True,
        alpha0: Optional[float] = None,
        beta0: Optional[float] = None,
        prop_random: Optional[float] = None,
        true_prior: Optional[float] = None,
        x_tick_labelsize: float = 6.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        ts = self._get_ts()
        thetas = [h.get("theta", 0.0) for h in self.history]

        fig, ax = plt.subplots()
        ax.plot(ts, thetas, linestyle="--", marker="x", label="theta")
        ax.set_title(
            _compose_title_with_run_config(title, alpha0, beta0, prop_random, true_prior)
        )
        ax.set_xlabel("Trial")
        ax.set_ylabel("E[theta]")
        _apply_small_xaxis_labels(ax, x_tick_labelsize)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best", fontsize="small")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    def plot_particle_weights_over_trials(
        self,
        title: str = "Particle weights over trials (by particle index)",
        save_path: Optional[str] = None,
        show: bool = True,
        use_hypothesis_legend: bool = False,
        alpha0: Optional[float] = None,
        beta0: Optional[float] = None,
        prop_random: Optional[float] = None,
        true_prior: Optional[float] = None,
        x_tick_labelsize: float = 6.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot each particle's weight vs trial index. Requires history entries with
        ``particle_weights`` (list of floats, one per particle slot).
        """
        ts = self._get_ts()
        first = self.history[0].get("particle_weights")
        if not first:
            raise ValueError(
                'history entries must include "particle_weights" (update Engine.run logging).'
            )
        n = len(first)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(n):
            ys = []
            for h in self.history:
                pw = h.get("particle_weights", [0.0] * n)
                ys.append(pw[i] if i < len(pw) else 0.0)
            label = None
            if use_hypothesis_legend and self.history:
                names = self.history[-1].get("particle_names") or []
                if i < len(names):
                    label = f"{i}: {names[i]}"
            if label is None:
                label = f"p{i}"
            ax.plot(ts, ys, label=label, alpha=0.75, linewidth=1.0)

        ax.set_title(
            _compose_title_with_run_config(title, alpha0, beta0, prop_random, true_prior)
        )
        ax.set_xlabel("Trial")
        ax.set_ylabel("Weight")
        _apply_small_xaxis_labels(ax, x_tick_labelsize)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        if n <= 15:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="xx-small", ncol=1)
        else:
            ax.text(
                0.02,
                0.98,
                f"One line per particle slot 0…{n - 1} (legend omitted for readability)",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="gray",
            )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    def plot_weights_by_name_over_trials(
        self,
        title: str = (
            "Total weight per hypothesis name (sum of particles; includes t=0 before first action)"
        ),
        save_path: Optional[str] = None,
        show: bool = True,
        top_k: Optional[int] = None,
        alpha0: Optional[float] = None,
        beta0: Optional[float] = None,
        prop_random: Optional[float] = None,
        true_prior: Optional[float] = None,
        x_tick_labelsize: float = 6.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot aggregated particle mass per hypothesis name vs trial.
        Uses ``weights_by_name`` from history if present, else ``probs`` (same content).
        The first point is t=0 (initial weights after particle init, before any trial update).
        """
        ts = self._get_ts()

        weights_list: List[Dict[str, float]] = []
        for i, h in enumerate(self.history):
            wmap = h.get("weights_by_name")
            if wmap is None:
                wmap = h.get("probs", {})
            if not isinstance(wmap, dict):
                raise TypeError(
                    f'history[{i}] must include "weights_by_name" or "probs" as dict[str,float]'
                )
            weights_list.append(wmap)

        all_names = set()
        for wmap in weights_list:
            all_names.update(wmap.keys())
        names = sorted(all_names)

        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            ranked = []
            for name in names:
                peak = max(w.get(name, 0.0) for w in weights_list)
                ranked.append((peak, name))
            ranked.sort(reverse=True)
            names = [name for _, name in ranked[:top_k]]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        for j, name in enumerate(names):
            ys = [w.get(name, 0.0) for w in weights_list]
            color = plt.cm.tab20((j % 20) / 20.0)
            ax.plot(ts, ys, label=name, linewidth=1.25, color=color, alpha=0.9)

        ax.set_title(
            _compose_title_with_run_config(title, alpha0, beta0, prop_random, true_prior)
        )
        ax.set_xlabel("Trial (0 = initial, before first action)")
        ax.set_ylabel("Mass per hypothesis (sum of particle weights by name)")
        _apply_small_xaxis_labels(ax, x_tick_labelsize)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.margins(x=0.02)
        if names:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize="xx-small",
                ncol=1 if len(names) <= 12 else 2,
                framealpha=0.92,
            )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    def plot_hypothesis_probs_over_trials(
        self,
        title: str = "Hypothesis probabilities over trials",
        save_path: Optional[str] = None,
        show: bool = True,
        top_k: Optional[int] = None,
        alpha0: Optional[float] = None,
        beta0: Optional[float] = None,
        prop_random: Optional[float] = None,
        true_prior: Optional[float] = None,
        x_tick_labelsize: float = 6.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        ts = self._get_ts()

        probs_list: List[Dict[str, float]] = []
        for i, h in enumerate(self.history):
            probs = h.get("weights_by_name") or h.get("probs", {})
            if not isinstance(probs, dict):
                raise TypeError(
                    f'history[{i}]["weights_by_name"] or ["probs"] must be a dict[str,float]'
                )
            probs_list.append(probs)

        # Collect hypothesis names
        all_names = set()
        for probs in probs_list:
            all_names.update(probs.keys())
        names = sorted(all_names)

        # Optionally reduce to top_k by peak probability
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            ranked = []
            for name in names:
                peak = max(p.get(name, 0.0) for p in probs_list)
                ranked.append((peak, name))
            ranked.sort(reverse=True)
            names = [name for _, name in ranked[:top_k]]

        fig, ax = plt.subplots()
        for name in names:
            ys = [p.get(name, 0.0) for p in probs_list]
            ax.plot(ts, ys, label=name)

        ax.set_title(
            _compose_title_with_run_config(title, alpha0, beta0, prop_random, true_prior)
        )
        ax.set_xlabel("Trial")
        ax.set_ylabel("Hypothesis probability")
        _apply_small_xaxis_labels(ax, x_tick_labelsize)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        if names:
            ax.legend(loc="best", fontsize="small")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    # Optional convenience wrapper to make both plots in one call
    def plot_probs_and_theta_separately(
        self,
        probs_title: str = "Hypothesis probabilities over trials",
        theta_title: str = "E[theta] over trials",
        probs_save_path: Optional[str] = None,
        theta_save_path: Optional[str] = None,
        show: bool = True,
        top_k: Optional[int] = None,
        alpha0: Optional[float] = None,
        beta0: Optional[float] = None,
        prop_random: Optional[float] = None,
        true_prior: Optional[float] = None,
        x_tick_labelsize: float = 6.0,
    ):
        fig_probs, ax_probs = self.plot_hypothesis_probs_over_trials(
            title=probs_title,
            save_path=probs_save_path,
            show=show,
            top_k=top_k,
            alpha0=alpha0,
            beta0=beta0,
            prop_random=prop_random,
            true_prior=true_prior,
            x_tick_labelsize=x_tick_labelsize,
        )
        fig_theta, ax_theta = self.plot_theta_over_trials(
            title=theta_title,
            save_path=theta_save_path,
            show=show,
            alpha0=alpha0,
            beta0=beta0,
            prop_random=prop_random,
            true_prior=true_prior,
            x_tick_labelsize=x_tick_labelsize,
        )
        return (fig_probs, ax_probs), (fig_theta, ax_theta)

    @staticmethod
    def plot_trials_to_solve_histogram(
        trial_counts: List[int],
        title: str = "Histogram: trials to solve (per run)",
        alpha0: Optional[float] = None,
        beta0: Optional[float] = None,
        prop_random: Optional[float] = None,
        true_prior: Optional[float] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        x_tick_labelsize: float = 5.0,
    ) -> Tuple[plt.Figure, plt.Axes, Counter]:
        if not trial_counts:
            raise ValueError("trial_counts is empty")
        if any((not isinstance(x, int)) or x < 0 for x in trial_counts):
            raise ValueError("trial_counts must be a list of non-negative ints")

        freq = Counter(trial_counts)
        xs = sorted(freq.keys())
        ys = [freq[x] for x in xs]

        fig, ax = plt.subplots()
        bars = ax.bar(xs, ys, width=0.9)
        full_title = _compose_title_with_run_config(title, alpha0, beta0, prop_random, true_prior)
        ax.set_title(full_title)
        ax.set_xlabel("Trials taken")
        ax.set_ylabel("Number of runs")
        ax.set_xticks(xs)
        # Tilt and use very small x-axis tick labels to reduce overlap.
        ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=x_tick_labelsize)

        # Add count labels on top of each non-zero bar.
        for x, y, bar in zip(xs, ys, bars):
            if y > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y,
                    str(y),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig.tight_layout()
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax, freq
    
class KidDataPlotter:
    """
    Plot kid real-data attempt behavior from JSON:
    {
        "D001": [
            ["red", "red", 0, {...}],
            ["red", "red", 1, {...}],
            ...
        ],
        ...
    }
    """

    def __init__(self, kid_data: dict):
        self.kid_data = kid_data

    def _count_opened_progress(self, trials, max_attempts=None):
        """
        Count cumulative number of successful openings over attempts.
        A success is outcome == 1.
        """
        if max_attempts is not None:
            trials = trials[:max_attempts]

        opened = 0
        y = []

        for trial in trials:
            outcome = int(trial[2])
            if outcome == 1:
                opened += 1
            y.append(opened)

        return y

    def get_ids_with_min_attempts(self, min_attempts=9, strictly_more=True):
        ids = []
        for kid_id, trials in self.kid_data.items():
            n = len(trials)
            if strictly_more:
                if n > min_attempts:
                    ids.append(kid_id)
            else:
                if n >= min_attempts:
                    ids.append(kid_id)
        return sorted(ids)

    def plot_first_n_attempts_for_selected_ids(
        self,
        selected_ids,
        first_n=10,
        show=True,
    ):
        """
        For each selected child ID, plot cumulative opened boxes over first N attempts.
        """
        if not selected_ids:
            raise ValueError("No selected IDs to plot.")

        plt.figure(figsize=(10, 6))

        for kid_id in selected_ids:
            trials = self.kid_data[kid_id]
            y = self._count_opened_progress(trials, max_attempts=first_n)
            x = list(range(1, len(y) + 1))
            plt.plot(x, y, marker="o", label=kid_id)

        plt.xlabel("Attempt")
        plt.ylabel("Cumulative opened boxes")
        plt.title(f"Kid data: first {first_n} attempts")
        plt.xticks(range(1, first_n + 1))
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=2)

        if show:
            plt.show()

    def plot_boxes_opened_histogram(self, show=True):
        """
        Histogram of how many unique boxes each kid opened successfully.

        x-axis: number of boxes opened
        y-axis: number of kids
        """
        opened_counts = []

        for kid_id, trials in self.kid_data.items():
            opened_boxes = set()

            for trial in trials:
                box_id = trial[1]
                outcome = int(trial[2])

                if outcome == 1:
                    opened_boxes.add(box_id)

            opened_counts.append(len(opened_boxes))

        freq = Counter(opened_counts)

        x_vals = sorted(freq.keys())
        y_vals = [freq[x] for x in x_vals]

        plt.figure(figsize=(8, 5))
        plt.bar(x_vals, y_vals, width=0.8)
        plt.xlabel("Number of boxes opened")
        plt.ylabel("Number of kids")
        plt.title("Histogram of boxes opened across kids")
        plt.xticks(x_vals)
        plt.grid(axis="y", alpha=0.3)

        if show:
            plt.show()

        return freq