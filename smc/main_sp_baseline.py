import hashlib

from environment import Environment
from sp_baseline import SPBaseline
from utils.plotter2 import Plotter2


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        if self.logging:
            print(log_str)


def _hyp_label(code: str) -> str:
    if code is None:
        return "hyp_none"
    h = hashlib.md5(code.encode("utf-8")).hexdigest()[:10]
    return f"hyp_{h}"


if __name__ == "__main__":
    num_runs = 5
    max_trials = 70

    trial_counts = []
    target_plot_history = None
    max_trials_taken = -1

    logger = Logger(logging=False)

    for _ in range(num_runs):
        env = Environment(opening_prob=,include_inspect=False)
        agent = SPBaseline(env, logger)
        result = agent.run(max_trials=max_trials)

        trials_taken = int(result.get("trials", 0))
        trial_counts.append(trials_taken)

        if trials_taken > max_trials_taken:
            max_trials_taken = trials_taken
            # Convert baseline per-step history into Plotter2-compatible history:
            # each step has a degenerate distribution over a single current hypothesis label.
            plot_history = []
            for step in result.get("history", []):
                label = _hyp_label(step.get("hypothesis"))
                plot_history.append(
                    {
                        "t": step.get("t"),
                        "opened": step.get("opened", 0),
                        "probs": {label: 1.0},
                    }
                )
            target_plot_history = plot_history

    Plotter2.plot_trials_to_solve_histogram(
        trial_counts,
        title=f"SP baseline: trials to solve across {num_runs} runs (max_trials={max_trials})",
        show=True,
    )

    if False:#target_plot_history:
        n_steps = len(target_plot_history)
        plotter = Plotter2(target_plot_history)
        plotter.plot_weights_by_name_over_trials(
            title=(
                "SP baseline: hypothesis identity over trials (degenerate mass = 1.0) "
                f"({n_steps} steps; max_trials={max_trials})"
            ),
            show=True,
        )
