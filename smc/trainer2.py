import math
import random
from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Tuple, Dict, Any, Optional

from environment import Environment
from generator import Generator
from smc import Engine


class BoxTaskTrainer:
    """
    Grid-search trainer for the Box Task using your existing Generator + SMC Engine.

    Matches the "fit parameters by maximizing cross-validated predictive likelihood"
    idea (80% fit / 20% evaluate) from the slides. :contentReference[oaicite:3]{index=3}

    IMPORTANT:
    - This trainer scores *outcome likelihood* p(o_t | history, params) under the SMC mixture.
    - It does NOT yet implement the paper's action-choice likelihood / softmax temperature τ.
      (Your current Engine does not take human actions as stochastic policy outputs.)
    """

    @dataclass(frozen=True)
    class FitParams:
        alpha0: float
        beta0: float

        # Keep prop_random available, but by default we keep it fixed unless you pass a list for it.
        prop_random: float

        # Unnormalised weights for rule families (Generator normalises internally)
        prior_color: float
        prior_order: float
        prior_shape: float
        prior_number: float
        prior_sim_color_total: float

        # SMC knobs (usually fixed)
        ess_threshold: float = 0.5
        k_rejuvenate: int = 1

    def __init__(
        self,
        D: Iterable[Tuple[str, str, int]],
        num_particles: int = 50,
        seed: int = 0,
        ess_threshold: float = 0.5,
        k_rejuvenate: int = 1,
        holdout_frac: float = 0.2,
    ):
        self.D = list(D)
        self.num_particles = num_particles
        self.seed = seed
        self.ess_threshold = ess_threshold
        self.k_rejuvenate = k_rejuvenate
        self.holdout_frac = holdout_frac

    # -----------------------------
    # Main scoring: predictive LL on held-out tail after fitting on prefix
    # -----------------------------
    def train_and_eval_map_on_full_data(self, params: "BoxTaskTrainer.FitParams") -> float:
        """
        Sanity-mode:
        - Use 100% of D to update the SMC posterior (fit).
        - Pick a single MAP hypothesis from the final particle weights.
        - Evaluate log-likelihood on the SAME D using ONLY that hypothesis (not mixture).
        """
        # validity checks (same as before)
        if params.alpha0 <= 0 or params.beta0 <= 0:
            return float("-inf")
        if not (0.0 <= params.prop_random <= 1.0):
            return float("-inf")
        for w in (
            params.prior_color,
            params.prior_order,
            params.prior_shape,
            params.prior_number,
            params.prior_sim_color_total,
        ):
            if w <= 0:
                return float("-inf")

        random.seed(self.seed)

        # 1) fit on 100% data (update posterior)
        env, generator, engine = self._build_engine(params)
        for (kid, bid, o_int) in self.D:
            if kid not in env.id_to_key or bid not in env.id_to_box:
                return float("-inf")
            key = env.id_to_key[kid]
            box = env.id_to_box[bid]
            outcome = bool(int(o_int))
            self._update_state(env, generator, engine, key, box, kid, bid, outcome)

        # 2) pick MAP hypothesis (highest final particle weight)
        map_particle = max(engine.particles, key=lambda p: p.weight)

        # 3) evaluate on same data using ONLY that hypothesis
        #    keep theta dynamics the same (alpha/beta update) but do NOT update particle weights
        env2, generator2, engine2 = self._build_engine(params)
        engine2.particles = [map_particle]
        engine2.particles[0].weight = 1.0

        ll = 0.0
        for (kid, bid, o_int) in self.D:
            key = env2.id_to_key[kid]
            box = env2.id_to_box[bid]
            outcome = bool(int(o_int))

            pred = engine2.particles[0].evaluate(key, box)
            p_out = engine2._compute_likelihood(pred, outcome)
            ll += math.log(max(p_out, 1e-12))

            # keep theta evolution consistent with your engine
            engine2._update_theta(box, outcome)
            engine2.trial_count = getattr(engine2, "trial_count", 0) + 1

        return ll

    # -----------------------------
    # Grid search over provided lists
    # -----------------------------
    def grid_search_fit(
        self,
        # 2 lists for skill prior hyperparameters
        alpha_list: List[float],
        beta_list: List[float],
        # 5 lists for rule-family weights (unnormalised)
        prior_color_list: List[float],
        prior_order_list: List[float],
        prior_shape_list: List[float],
        prior_number_list: List[float],
        prior_sim_color_total_list: List[float],
        # optional: keep prop_random fixed by default unless you provide a list
        prop_random_list: Optional[List[float]] = None,
    ) -> Tuple["BoxTaskTrainer.FitParams", float, float]:
        """
        Returns: (best_params, best_nll, best_ll)

        - best_nll is minimized (lower is better)
        - best_ll is the corresponding predictive log-likelihood (higher is better)
        """

        if prop_random_list is None:
            prop_random_list = [0.0]  # default: no generator-mass, unless you explicitly vary it

        best_params: Optional[BoxTaskTrainer.FitParams] = None
        best_nll = float("inf")
        best_ll = float("-inf")

        # NOTE: list length 1 means fixed; product() naturally handles that.
        for (
            a0, b0,
            pgen,
            wc, wo, ws, wn, wsim,
        ) in product(
            alpha_list, beta_list,
            prop_random_list,
            prior_color_list, prior_order_list, prior_shape_list, prior_number_list, prior_sim_color_total_list
        ):
            params = BoxTaskTrainer.FitParams(
                alpha0=a0,
                beta0=b0,
                prop_random=pgen,
                prior_color=wc,
                prior_order=wo,
                prior_shape=ws,
                prior_number=wn,
                prior_sim_color_total=wsim,
                ess_threshold=self.ess_threshold,
                k_rejuvenate=self.k_rejuvenate,
            )

            ll = self.train_and_eval_map_on_full_data(params)
            if ll == float("-inf"):
                continue

            nll = -ll
            if nll < best_nll:
                best_nll = nll
                best_ll = ll
                best_params = params

        if best_params is None:
            raise RuntimeError("No valid parameter combination produced a finite likelihood.")

        return best_params, best_ll

    # -----------------------------
    # internals: build + likelihood + update (mirrors Engine.run loop)
    # -----------------------------
    def _build_engine(self, params: "BoxTaskTrainer.FitParams"):
        env = Environment(include_inspect=False)

        # This assumes you modified Generator to support train=True without requiring omega.
        gen_config: Dict[str, Any] = {
            "train": True,

            # generator-mass (optional hyperparameter)
            "prop_random": params.prop_random,

            # rule-family weights (unnormalised)
            "prior_color": params.prior_color,
            "prior_order": params.prior_order,
            "prior_shape": params.prior_shape,
            "prior_number": params.prior_number,
            "prior_sim_color_total": params.prior_sim_color_total,
        }
        generator = Generator(gen_config, env)

        smc_config: Dict[str, Any] = {
            "num_particles": self.num_particles,
            "theta_distribution": (params.alpha0, params.beta0),
            "ess_threshold": params.ess_threshold,
            "k_rejuvenate": params.k_rejuvenate,
            "mode": "soc",
        }
        engine = Engine(smc_config, env, generator)
        return env, generator, engine

    def _marginal_outcome_prob(self, engine: Engine, key, box, outcome: bool) -> float:
        # p(o_t | mixture) = Σ_i w_i * p(o_t | h_i, theta)
        p_out = 0.0
        for particle in engine.particles:
            pred = particle.evaluate(key, box)
            p_out += particle.weight * engine._compute_likelihood(pred, outcome)
        return p_out

    def _update_state(
        self,
        env: Environment,
        generator: Generator,
        engine: Engine,
        key,
        box,
        kid: str,
        bid: str,
        outcome: bool,
    ) -> None:
        # evidence for MH rejuvenation
        engine.evidence.append((key, box, outcome))

        # prune proposals after a success (same as Engine.run)
        if outcome is True:
            env.success_pairs.add((kid, bid))
            generator.prune_proposal_dist(key, box)

        engine._update_theta(box, outcome)
        engine._update_particle_weights(key, box, outcome)
        engine.trial_count = getattr(engine, "trial_count", 0) + 1