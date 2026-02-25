import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

"""
We need a plot that shows how many boxes are opened as trials progress.
Also a plot showing how the probability of each hypotheses, and theta expectation, are evolving over time (a line graph, one line graph per one simulation run)
"""

"""
    Expects Engine-style history:
      history = [
        {"t": ..., "opened": ..., "theta": ..., "probs": {hyp_name: prob, ...}},
        ...
      ]
"""
class Plotter:

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

    def plot_hypothesis_probs_and_theta(
        self,
        title: str = "Hypothesis probabilities and E[theta] over trials",
        save_path: Optional[str] = None,
        show: bool = True,
        top_k: Optional[int] = None,
    ) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:

        ts = self._get_ts()
        thetas = [h.get("theta", 0.0) for h in self.history]

        probs_list: List[Dict[str, float]] = []
        for i, h in enumerate(self.history):
            probs = h.get("probs", {})
            if not isinstance(probs, dict):
                raise TypeError(f'history[{i}]["probs"] must be a dict[str,float]')
            probs_list.append(probs)

        # Collect all hypothesis names seen over time
        all_names = set()
        for probs in probs_list:
            all_names.update(probs.keys())
        names = sorted(all_names)

        # Optionally reduce to top_k
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            ranked = []
            for name in names:
                peak = max(p.get(name, 0.0) for p in probs_list)
                ranked.append((peak, name))
            ranked.sort(reverse=True)
            names = [name for _, name in ranked[:top_k]]

        fig, ax_prob = plt.subplots()

        # Hypothesis probability lines
        for name in names:
            ys = [p.get(name, 0.0) for p in probs_list]
            ax_prob.plot(ts, ys, label=name)

        ax_prob.set_title(title)
        ax_prob.set_xlabel("Trial")
        ax_prob.set_ylabel("Hypothesis probability")
        ax_prob.set_ylim(0.0, 1.0)
        ax_prob.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        # Theta on a twin axis
        ax_theta = ax_prob.twinx()
        ax_theta.plot(ts, thetas, linestyle="--", marker="x", label="theta")
        ax_theta.set_ylabel("theta")
        ax_theta.set_ylim(0.0, 1.0)

        # Combined legend
        h1, l1 = ax_prob.get_legend_handles_labels()
        h2, l2 = ax_theta.get_legend_handles_labels()
        if h1 or h2:
            ax_prob.legend(h1 + h2, l1 + l2, loc="best", fontsize="small")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax_prob, ax_theta