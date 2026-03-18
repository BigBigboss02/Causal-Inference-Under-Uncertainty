"""
===========================================
METHOD 1: IG-SOFTMAX LIKELIHOOD FITTING
===========================================

Mathematical model:

For one child i:

    p(D_i | φ) = ∏_{t=1}^{T_i} p(a_t | h_<t, φ)

Log-likelihood:

    LL_i(φ) = ∑_{t=1}^{T_i} log p(a_t | h_<t, φ)

Across all children:

    LL_total(φ) = ∑_{i=1}^{N} LL_i(φ)

Best model:

    φ* = argmax_φ LL_total(φ)

-------------------------------------------
We FIX temperature to a LOW value:

    TEMPERATURE = 0.2

This approximates near-greedy information-seeking behavior.
"""

import os
import copy
import json
import math
import itertools
from datetime import datetime
from tqdm import tqdm

from smc_soc import Engine
from environment import Environment
from gen_soc import Generator


# =========================
# FIXED TEMPERATURE
# =========================
TEMPERATURE = 0.2


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
    "save_dir": r"training_results\ig_softmax_method",
    "show_progress": True,
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

def add_ig_softmax_to_engine():

    def get_action_probs_ig_softmax(self, temperature=TEMPERATURE):
        actions = self.env.actions

        ig_values = []
        action_list = []

        for (key, box) in actions:
            if (key.id, box.id) in self.env.success_pairs:
                continue

            ig = self._compute_info_gain(key, box)
            ig_values.append(ig)
            action_list.append((key.id, box.id))

        if len(ig_values) == 0:
            return {}

        scaled = [ig / temperature for ig in ig_values]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        total = sum(exps)

        probs = [e / total for e in exps]

        return {a: p for a, p in zip(action_list, probs)}

    def observe_child_trial(self, key, box, outcome):
        self.evidence.append((key, box, outcome))

        if outcome:
            self.proposal.prune_proposal_dist(key, box)
            self.succ_count[(key.id, box.id)] += 1
        else:
            self.fail_count[(key.id, box.id)] += 1

        if self.skill:
            self._compute_theta()

        self._update_particle_weights(key, box, outcome)

    Engine.get_action_probs_ig_softmax = get_action_probs_ig_softmax
    Engine.observe_child_trial = observe_child_trial


# =========================
# FITTER
# =========================

class IGSoftmaxFitter:

    def __init__(self, gen_config, smc_config, fit_config):
        self.gen_config = gen_config
        self.smc_config = smc_config
        self.fit_config = fit_config
        self.data_path = fit_config["data_path"]

    def load_children(self):
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

            probs = smc.get_action_probs_ig_softmax()

            p = probs.get((key_id, box_id), 1e-12)
            loglik += math.log(p)

            smc.observe_child_trial(key, box, bool(outcome))

        return loglik

    def fit(self):

        self.load_children()

        total_ll = 0.0

        iterator = self.children
        if self.fit_config["show_progress"]:
            iterator = tqdm(iterator)

        for child in iterator:
            ll = self.fit_one_child(child["trials"])
            total_ll += ll

        return total_ll


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

        add_ig_softmax_to_engine()

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

            fitter = IGSoftmaxFitter(gcfg, scfg, fit_config)

            ll = fitter.fit()

            result = {
                "theta": theta,
                "gen_prior": gen_prior,
                "rule_prior": rule_prior,
                "loglik": ll
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