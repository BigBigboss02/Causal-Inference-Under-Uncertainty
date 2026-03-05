import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict, Any
from smc import Engine
from environment import Environment
from smc.gen_soc import Generator


class BoxTaskTrainer:

    def __init__(
        self,
        D: Iterable[Tuple[str, str, int]],
        num_particles: int = 20,
        seed: int = 0,
    ):
        self.D = list(D)
        self.num_particles = num_particles
        self.seed = seed

    def log_likelihood(
        self,
        skill: SkillOnlyParams,
        strategy: Optional[StrategyParams] = None,
    ) -> float:
        if skill.alpha0 <= 0 or skill.beta0 <= 0:
            return float("-inf")

        random.seed(self.seed) # ensure reproducibility

        env, generator, engine = self.training_engine(skill=skill, strategy=strategy)

        ll = 0.0
        for (kid, bid, o_int) in self.D:
            key = env.id_to_key[kid]
            box = env.id_to_box[bid]
            outcome = bool(o_int)

            # 1) marginal probability of observed outcome under current particle mixture
            p_out = self._marginal_outcome_prob(engine, key, box, outcome)
            ll += math.log(max(p_out, 1e-12))

            # 2) update internal state using the OBSERVED outcome
            self._update_state(env, generator, engine, key, box, kid, bid, outcome)

        return ll

    def training_engine(
        self):

        env = Environment(include_inspect=False)

        gen_config = {
            "omega": 2.0,
            "prop_random": 0.0,  # disable generator proposals
        }
        generator = Generator(gen_config, env)

        smc_config = {
            "num_particles": self.num_particles,
            "theta_distribution": (skill.alpha0, skill.beta0),
            "ess_threshold": 0.0,   # disables resample
            "k_rejuvenate": 0,
            "mode": "soc",
            "prior": "uniform",
        }

        engine = Engine(smc_config, env, generator)
        return env, generator, engine

    # Likelihood computation
    def _marginal_outcome_prob(self, engine, key, box, outcome: bool) -> float:
        """
        p(o_t | mixture over particles) = Σ_i w_i * p(o_t | h_i, theta)
        """
        p_out = 0.0
        for particle in engine.particles:
            pred = particle.evaluate(key, box)  # predicted open/close under that hypothesis
            p_out += particle.weight * engine._compute_likelihood(pred, outcome)
        return p_out

    # State update
    def _update_state(
        self,
        env,
        generator,
        engine,
        key,
        box,
        kid: str,
        bid: str,
        outcome: bool,
    ) -> None:

        engine.evidence.append((key, box, outcome))

        if outcome:
            # Keep internal env state consistent with "opened" evidence
            env.success_pairs.add((kid, bid))
            generator.prune_proposal_dist(key, box)

        # Update theta + particle weights
        engine._update_theta(box, outcome)
        engine._update_particle_weights(key, box, outcome)

        engine.trial_count = getattr(engine, "trial_count", 0) + 1