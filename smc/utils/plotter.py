import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple


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