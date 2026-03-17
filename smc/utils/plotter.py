import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter


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
    ) -> Tuple[plt.Figure, plt.Axes]:
        ts = self._get_ts()
        opened = [h.get("opened", 0) for h in self.history]

        fig, ax = plt.subplots()
        ax.plot(ts, opened, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Opened boxes")
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
    ) -> Tuple[plt.Figure, plt.Axes]:
        ts = self._get_ts()
        thetas = [h.get("theta", 0.0) for h in self.history]

        fig, ax = plt.subplots()
        ax.plot(ts, thetas, linestyle="--", marker="x", label="theta")
        ax.set_title(title)
        ax.set_xlabel("Trial")
        ax.set_ylabel("E[theta]")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best", fontsize="small")

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
    ) -> Tuple[plt.Figure, plt.Axes]:
        ts = self._get_ts()

        probs_list: List[Dict[str, float]] = []
        for i, h in enumerate(self.history):
            probs = h.get("probs", {})
            if not isinstance(probs, dict):
                raise TypeError(f'history[{i}]["probs"] must be a dict[str,float]')
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

        ax.set_title(title)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Hypothesis probability")
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
    ):
        fig_probs, ax_probs = self.plot_hypothesis_probs_over_trials(
            title=probs_title, save_path=probs_save_path, show=show, top_k=top_k
        )
        fig_theta, ax_theta = self.plot_theta_over_trials(
            title=theta_title, save_path=theta_save_path, show=show
        )
        return (fig_probs, ax_probs), (fig_theta, ax_theta)
    
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