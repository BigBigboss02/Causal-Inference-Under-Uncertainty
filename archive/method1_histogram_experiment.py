import os
import copy
import itertools
import json
from datetime import datetime
from method1_histogram_fit import HistogramFitter, gen_config, smc_config, hist_config

THETA_LIST = [
    (9, 1),
    (19, 1),
    (8, 1),
    (7, 1),
    (6, 1),
]

GENERATOR_PRIOR_LIST = [0.01, 0.1, 0.3, 0.5, 0.8]
TRUE_RULE_PRIOR_LIST = [0.01, 0.1, 0.2]

# THETA_LIST = [
#     (9, 1),
# ]

# GENERATOR_PRIOR_LIST = [0.01]
# TRUE_RULE_PRIOR_LIST = [0.01]
SAVING_DIR = r"training_results\histogram_method\experiments_batch_1000_run_08032026"


class HistogramExperimentRunner:
    """
    Sweep over:
      1. theta prior
      2. generator prior
      3. true rule prior (prior_order)

    Rule-prior constraint:
      prior_order = true_rule_prior
      all other 4 priors = (1 - prior_order) / 4
    """
    @staticmethod
    def make_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")


    @staticmethod
    def make_config_tag(theta, generator_prior, true_rule_prior):
        theta_str = f"{theta[0]}_{theta[1]}"
        gen_str = str(generator_prior).replace(".", "p")
        rule_str = str(true_rule_prior).replace(".", "p")

        return (
            f"theta_{theta_str}"
            f"__gen_{gen_str}"
            f"__trueprior_{rule_str}"
        )
    
    def __init__(
        self,
        base_gen_config: dict,
        base_smc_config: dict,
        base_hist_config: dict,
        theta_list=None,
        generator_prior_list=None,
        true_rule_prior_list=None,
    ):
        self.base_gen_config = copy.deepcopy(base_gen_config)
        self.base_smc_config = copy.deepcopy(base_smc_config)
        self.base_hist_config = copy.deepcopy(base_hist_config)

        self.theta_list = theta_list or THETA_LIST
        self.generator_prior_list = generator_prior_list or GENERATOR_PRIOR_LIST
        self.true_rule_prior_list = true_rule_prior_list or TRUE_RULE_PRIOR_LIST

        self.results = []

    @staticmethod
    def assign_rule_priors(gcfg: dict, prior_order: float):
        """
        Set the 5 rule priors so they sum to 1:
            prior_order = given value
            the other 4 priors = (1 - prior_order) / 4
        """
        if not (0.0 <= prior_order <= 1.0):
            raise ValueError(f"prior_order must be in [0, 1], got {prior_order}")

        other_prior = (1.0 - prior_order) / 4.0

        gcfg["prior_color"] = other_prior
        gcfg["prior_order"] = prior_order
        gcfg["prior_shape"] = other_prior
        gcfg["prior_number"] = other_prior
        gcfg["prior_sim_color_total"] = other_prior

        return gcfg
    
    def build_configs(
        self,
        theta,
        generator_prior,
        true_rule_prior,
        num_runs=None,
        trials_per_run=None,
    ):
        gcfg = copy.deepcopy(self.base_gen_config)
        scfg = copy.deepcopy(self.base_smc_config)
        hcfg = copy.deepcopy(self.base_hist_config)

        # 1) theta prior in SMC
        scfg["init_theta"] = theta

        # 2) generator prior
        gcfg["prop_random"] = generator_prior

        # 3) true rule prior = prior_order
        gcfg = self.assign_rule_priors(gcfg, prior_order=true_rule_prior)

        # 4) allow num_runs / trials_per_run to be modified per hyperparameter setting
        if num_runs is not None:
            hcfg["num_runs"] = num_runs
        if trials_per_run is not None:
            hcfg["trials_per_run"] = trials_per_run

        # 5) force saving behavior
        hcfg["save_plot"] = True
        hcfg["show_plot"] = False

        # 6) build timestamped config tag
        timestamp = self.make_timestamp()
        config_tag = self.make_config_tag(
            theta=theta,
            generator_prior=generator_prior,
            true_rule_prior=true_rule_prior,
        )

        # 7) create one folder per hyperparameter setting
        experiment_dir = os.path.join(SAVING_DIR, f"{config_tag}__{timestamp}")
        hcfg["save_dir"] = experiment_dir
        hcfg["file_name"] = f"{config_tag}__{timestamp}.png"

        return gcfg, scfg, hcfg, experiment_dir, config_tag, timestamp

    def run_one(
        self,
        theta,
        generator_prior,
        true_rule_prior,
        num_runs=None,
        trials_per_run=None,
    ):
        gcfg, scfg, hcfg, experiment_dir, config_tag, timestamp = self.build_configs(
            theta=theta,
            generator_prior=generator_prior,
            true_rule_prior=true_rule_prior,
            num_runs=num_runs,
            trials_per_run=trials_per_run,
        )

        fitter = HistogramFitter(
            gen_config=gcfg,
            smc_config=scfg,
            hist_config=hcfg,
        )

        attempt_counts, freq, failed_runs = fitter.fit()
        save_path = fitter.plot()

        result = {
            "timestamp": timestamp,
            "config_tag": config_tag,
            "theta": list(theta),
            "generator_prior": generator_prior,
            "true_rule_prior": true_rule_prior,
            "num_runs": hcfg["num_runs"],
            "trials_per_run": hcfg["trials_per_run"],
            "prior_color": gcfg["prior_color"],
            "prior_order": gcfg["prior_order"],
            "prior_shape": gcfg["prior_shape"],
            "prior_number": gcfg["prior_number"],
            "prior_sim_color_total": gcfg["prior_sim_color_total"],
            "num_successful_runs": len(attempt_counts),
            "num_failed_runs": failed_runs,
            "attempt_counts": attempt_counts,
            "frequency_table": dict(sorted(freq.items())),
            "save_path": save_path,
        }

        if attempt_counts:
            result["mean_attempts"] = sum(attempt_counts) / len(attempt_counts)
            result["min_attempts"] = min(attempt_counts)
            result["max_attempts"] = max(attempt_counts)
        else:
            result["mean_attempts"] = None
            result["min_attempts"] = None
            result["max_attempts"] = None

        os.makedirs(experiment_dir, exist_ok=True)

        result_json_path = os.path.join(
            experiment_dir,
            f"{config_tag}__{timestamp}.json"
        )

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        result["result_json_path"] = result_json_path

        self.results.append(result)
        return result
    def run_all(self, num_runs=None, trials_per_run=None):
        all_conditions = list(
            itertools.product(
                self.theta_list,
                self.generator_prior_list,
                self.true_rule_prior_list,
            )
        )

        print(f"Running {len(all_conditions)} experiment conditions...")

        for idx, (theta, generator_prior, true_rule_prior) in enumerate(all_conditions, start=1):
            print(
                f"[{idx}/{len(all_conditions)}] "
                f"theta={theta}, "
                f"generator_prior={generator_prior}, "
                f"prior_order={true_rule_prior}, "
                f"num_runs={num_runs if num_runs is not None else self.base_hist_config['num_runs']}, "
                f"trials_per_run={trials_per_run if trials_per_run is not None else self.base_hist_config['trials_per_run']}"
            )

            self.run_one(
                theta=theta,
                generator_prior=generator_prior,
                true_rule_prior=true_rule_prior,
                num_runs=num_runs,
                trials_per_run=trials_per_run,
            )

        return self.results

    def save_results_json(self, out_path=None):
        if out_path is None:
            out_path = os.path.join(
                self.base_hist_config.get("save_dir", SAVING_DIR),
                "experiments",
                "histogram_experiment_results.json",
            )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        return out_path

    def print_brief_table(self):
        print("\n=== Experiment Summary ===")
        for r in self.results:
            print(
                f"theta={tuple(r['theta'])}, "
                f"gen_prior={r['generator_prior']}, "
                f"prior_order={r['prior_order']:.3f}, "
                f"other_priors={r['prior_color']:.3f} | "
                f"success={r['num_successful_runs']}, "
                f"failed={r['num_failed_runs']}, "
                f"mean_attempts={r['mean_attempts']}"
            )


if __name__ == "__main__":
    runner = HistogramExperimentRunner(
        base_gen_config=gen_config,
        base_smc_config=smc_config,
        base_hist_config=hist_config,
    )
    runner.run_all(num_runs=1000, trials_per_run=100)
    runner.print_brief_table()
    json_path = runner.save_results_json()

    print(f"\nSaved sweep results to: {json_path}")