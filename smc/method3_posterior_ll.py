"""
===========================================
METHOD 2: POSTERIOR-OPENING-PROBABILITY FITTING
===========================================


Mathematical model

For one child i:

    p(D_i | φ) = ∏_{t=1}^{T_i} p(a_t | h_<t, φ)

where:
    D_i   = the full observed sequence for child i
    a_t   = the child's chosen action at trial t
    h_<t  = the history before trial t
    φ     = the model configuration

So the child-level log-likelihood is:

    LL_i(φ) = ∑_{t=1}^{T_i} log p(a_t | h_<t, φ)

Across all children:

    LL_total(φ) = ∑_{i=1}^{N} LL_i(φ)

Best model:

    φ* = argmax_φ LL_total(φ)

"""

import os
import copy
import json
import math
import itertools
from tqdm import tqdm

from smc_soc import Engine
from environment import Environment
from gen_soc import Generator


# =========================
# Base configs (same style)
# =========================

gen_config = {
    "omega": 2.0,
    "prop_random": 0.1,
    "train": False,
}

smc_config = {
    "num_particles": 30,
    "init_theta": (0.5, 0.5),
    "ess_threshold": 0.5,
    "skill": True,
    "mode": "soc",
    "prior": "uniform",
}

fit_config = {
    "data_path": r"C:\Users\MSN\Documents\Python\smc-s\data\Dolly_KeyEviModel_7.3.24.json",
    "save_dir": r"training_results\posterior_opening_method",
    "show_progress": True,
    "eps": 1e-12,
}


# THETA_LIST = [
#     (9, 1),
#     (19, 1),
#     (8, 1),
#     (7, 1),
#     (6, 1),
# ]

# GENERATOR_PRIOR_LIST = [0.01, 0.1, 0.3, 0.5, 0.8]
# TRUE_RULE_PRIOR_LIST = [0.01, 0.1, 0.2]

THETA_LIST = [
    (9, 1),
]

GENERATOR_PRIOR_LIST = [0.01]
TRUE_RULE_PRIOR_LIST = [0.01]


# =========================
# Logger
# =========================

class Logger:
    def __init__(self, logging=False):
        self.logging = logging

    def log(self, s):
        if self.logging:
            print(s)


# =========================
# ADD THIS TO YOUR ENGINE
# =========================

def add_posterior_opening_to_engine():
    """
    Monkey-patch helper methods into Engine for Method 2.
    """

    def get_action_open_probs(self):
        """
        Compute q(a) = posterior probability that action a opens the box.

        For each candidate action a = (key, box):

            q(a) = ∑_j w_j * P(open=True | a, H_j)

        where:
            - w_j is particle weight
            - H_j is particle j's hypothesis
            - P(open=True | a, H_j) is computed using the engine's
              existing likelihood model for outcome=True
        """
        actions = self.env.actions
        action_open_probs = {}

        for (key, box) in actions:
            if key == "inspect":
                continue

            if (key.id, box.id) in self.env.success_pairs:
                continue

            q_a = 0.0
            for particle in self.particles:
                pred_open = particle.evaluate(key, box)
                p_open_given_h = self._compute_likelihood(pred_open, True)
                q_a += particle.weight * p_open_given_h

            action_open_probs[(key.id, box.id)] = q_a

        return action_open_probs

    def get_action_probs_posterior_opening(self, eps=1e-12):
        """
        Normalize q(a) across actions:

            p(a) = q(a) / ∑_{a'} q(a')

        If the total mass is numerically zero, fall back to uniform.
        """
        action_open_probs = self.get_action_open_probs()

        if not action_open_probs:
            return {}

        total = sum(action_open_probs.values())

        if total <= 0:
            n = len(action_open_probs)
            return {a: 1.0 / n for a in action_open_probs}

        probs = {
            a: max(eps, q_a / total)
            for a, q_a in action_open_probs.items()
        }

        # re-normalize after eps floor
        z = sum(probs.values())
        probs = {a: p / z for a, p in probs.items()}

        return probs

    def observe_child_trial(self, key, box, outcome):
        """
        Update the engine with the child's actual observed trial.
        This mirrors the Bayesian update logic used in run(),
        except the action is supplied by the child data.
        """
        self.evidence.append((key, box, outcome))

        if outcome:
            self.proposal.prune_proposal_dist(key, box)
            self.succ_count[(key.id, box.id)] += 1
        else:
            self.fail_count[(key.id, box.id)] += 1

        if self.skill:
            self._compute_theta()

        self._update_particle_weights(key, box, outcome)

    Engine.get_action_open_probs = get_action_open_probs
    Engine.get_action_probs_posterior_opening = get_action_probs_posterior_opening
    Engine.observe_child_trial = observe_child_trial


# =========================
# FITTER
# =========================

class PosteriorOpeningFitter:

    def __init__(self, gen_config, smc_config, fit_config):
        self.gen_config = gen_config
        self.smc_config = smc_config
        self.fit_config = fit_config
        self.data_path = fit_config["data_path"]
        self.eps = fit_config.get("eps", 1e-12)

    def load_children(self):
        """
        Child data format:
        {
            "D001": [
                ["red", "red", 0, {...}],
                ...
            ],
            "D002": [...],
            ...
        }
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.children = [
            {"child_id": k, "trials": v}
            for k, v in data.items()
        ]

    def fit_one_child(self, trials):
        env = Environment(include_inspect=False)
        gen = Generator(self.gen_config, env)
        logger = Logger(False)
        smc = Engine(self.smc_config, env, gen, logger)

        loglik = 0.0

        for trial in trials:
            key_id, box_id, outcome = trial[0], trial[1], trial[2]

            key = env.id_to_key[key_id]
            box = env.id_to_box[box_id]

            probs = smc.get_action_probs_posterior_opening(eps=self.eps)

            p = probs.get((key_id, box_id), self.eps)
            p = max(self.eps, p)
            loglik += math.log(p)

            smc.observe_child_trial(key, box, bool(outcome))

        return loglik

    def fit(self):
        self.load_children()

        total_ll = 0.0
        child_results = []

        iterator = self.children
        if self.fit_config["show_progress"]:
            iterator = tqdm(iterator)

        for child in iterator:
            ll = self.fit_one_child(child["trials"])
            total_ll += ll
            child_results.append({
                "child_id": child["child_id"],
                "loglik": ll,
            })

        return {
            "total_loglik": total_ll,
            "child_results": child_results,
        }


# =========================
# EXPERIMENT RUNNER
# =========================

class ExperimentRunner:

    def __init__(self):
        self.results = []

    @staticmethod
    def assign_rule_priors(gcfg, prior_order):
        other = (1 - prior_order) / 4
        gcfg["prior_color"] = other
        gcfg["prior_order"] = prior_order
        gcfg["prior_shape"] = other
        gcfg["prior_number"] = other
        gcfg["prior_sim_color_total"] = other
        return gcfg

    def run_all(self):
        add_posterior_opening_to_engine()

        all_conditions = list(itertools.product(
            THETA_LIST,
            GENERATOR_PRIOR_LIST,
            TRUE_RULE_PRIOR_LIST
        ))

        for theta, gen_prior, rule_prior in all_conditions:
            gcfg = copy.deepcopy(gen_config)
            scfg = copy.deepcopy(smc_config)

            scfg["init_theta"] = theta
            gcfg["prop_random"] = gen_prior
            gcfg = self.assign_rule_priors(gcfg, rule_prior)

            fitter = PosteriorOpeningFitter(gcfg, scfg, fit_config)
            fit_result = fitter.fit()

            result = {
                "theta": theta,
                "gen_prior": gen_prior,
                "rule_prior": rule_prior,
                "loglik": fit_result["total_loglik"],
                #"child_results": fit_result["child_results"],
            }

            print(result)
            self.results.append(result)

        return self.results


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    runner = ExperimentRunner()
    results = runner.run_all()

    print("\n=== FINAL RESULTS ===")
    for r in results:
        print(r)