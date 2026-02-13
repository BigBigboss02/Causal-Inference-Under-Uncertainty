from this import d
from environment import Environment, Key, Box
from typing import List, Optional, Dict, Tuple
from generator import Generator
from default_data import default_hypothesis_soc, default_hypothesis_id, default_hypothesis_code
from code_utils import execute_hypothesis_code
import math, random


class Particle:

    def __init__(self, id, hypothesis, weight: float = 1.0):

        self.mode = 'soc' if isinstance(hypothesis, dict) else 'code'

        self.id = id
        self.hypothesis = hypothesis
        self.weight = weight
    
    def evaluate(self, key: Key, box: Box) -> bool:
        if self.mode == 'soc':
            result = (self.hypothesis.get(key.id) == box.id)
        elif self.mode == 'code':
            result = execute_hypothesis_code(self.hypothesis, key, box)
        return result


class Engine:

    def __init__(self, config: Dict, environment: Environment, llm: Generator = None, logger = None):
        
        self.num_particles: int = config['num_particles']
        self.theta: float = config['theta']
        self.ess_threshold: float = config['ess_threshold']

        self.environment: Environment = environment
        self.llm: Generator = llm
        self.logger = logger

        self.mode: str = config['mode']
        self.prior: str = config['prior'] # uniform or random

        self.particles = self._initialize_particles()

        
    def _initialize_particles(self) -> List[Particle]:

        if self.mode == "soc":
            hypotheses, ids = default_hypothesis_soc[:self.num_particles], default_hypothesis_id[:self.num_particles]
        elif self.mode == "code":
            hypotheses, ids= self.llm.generate_hypotheses(self.num_particles)
        
        if self.prior == "uniform":
            priors = [(1.0 / self.num_particles) for _ in range(self.num_particles)]
        elif self.prior == "random":
            priors = [random.random() for _ in range(self.num_particles)]
            total = sum(priors)
            priors = [p / total for p in priors]

        particles = [Particle(id=ids[i], hypothesis=hypotheses[i], weight=priors[i]) for i in range(self.num_particles)]
        return particles


    def _resample(self):
        weights = [p.weight for p in self.particles]

        if sum(weights) == 0:
            indices = random.choices(range(self.num_particles), k=self.num_particles)
        else:
            indices = random.choices(range(self.num_particles), k=self.num_particles, weights=weights)
        
        resampled = list()
        for i in indices:
            new_particle = Particle(id=self.particles[i].id, hypothesis=self.particles[i].hypothesis, weight=(1.0 / self.num_particles))
            resampled.append(new_particle)
        self.particles = resampled
            

    def _compute_ess(self) -> float:
        """
        compute effective sample size
        """
        weights = [p.weight for p in self.particles]
        if sum(weights) == 0:
            return 0.0
        else:
            return 1.0 / sum(w**2 for w in weights)


    def _compute_entropy(self, particle_weights: List[float]) -> float:
        weights = [w for w in particle_weights if w > 0]
        if len(weights) <= 1:
            return 0.0
        else:
            return -1.0 * sum(w * math.log2(w) for w in weights)


    def _compute_inspect_info_gain(self, box: Box) -> float:
        """
        calculate information gain by inspect box action
        """
        pass
        
    
    def _compute_likelihood(self, predict: bool, outcome: bool) -> float:
        if predict and outcome:
            return self.theta
        elif predict and not outcome:
            return 1 - self.theta
        elif not predict and outcome:
            return 0.0
        else:
            return 1.0

    def _compute_info_gain(self, key: Key, box: Box) -> float:
        """
        calculate information gain as a difference of entropy by taking trial action
        """
        current_entropy = self._compute_entropy([p.weight for p in self.particles])
        expected_entropy = 0.0 # expected entropy after observing outcome
        
        for outcome in [True, False]:
            # predicted probability of outcome based on current weight distribution
            outcome_prob = 0.0
            # simulate weight updates to calculate new entropy if outcome is observed
            updated_weights = list() 

            for particle in self.particles:
                pred_outcome = particle.evaluate(key, box)
                if pred_outcome == outcome:
                    outcome_prob += particle.weight
                likelihood = self._compute_likelihood(pred_outcome, outcome)
                updated_weights.append(particle.weight * likelihood)

            # normalize updated weights
            total = sum(updated_weights)
            if total <= 0:
                continue
            updated_weights = [w / total for w in updated_weights if w > 0]
            new_entropy = self._compute_entropy(updated_weights)
            
            expected_entropy += outcome_prob * new_entropy
        
        return current_entropy - expected_entropy
    
    
    def _update_particle_weights(self, key: Key, box: Box, outcome: bool):
        """
        update weight of each particle based on whether outcome matches with hypothesis
        """
        for particle in self.particles:
            pred_outcome = particle.evaluate(key, box)
            likelihood = self._compute_likelihood(pred_outcome, outcome)
            particle.weight = particle.weight * likelihood

        # normalize weights
        total_weight = sum([p.weight for p in self.particles])
        for particle in self.particles:
            particle.weight = (particle.weight / total_weight) if total_weight > 0 else (1.0 / self.num_particles)

        # get effective sample size
        ess = self._compute_ess()
        if ess < self.num_particles * self.ess_threshold:
            self._resample()

    
    def _select_action(self):

        actions = self.environment.actions
        
        # determine action that maximizes information gain
        max_info_gain = float('-inf')
        best_action = None
        for (key, box) in actions:
            if key == 'inspect':
                info_gain = self._compute_inspect_info_gain(box)
            else:
                if (key.id, box.id) in self.environment.success_pairs: # box already opened with key
                    continue
                info_gain = self._compute_info_gain(key, box)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_action = (key, box)

        return best_action

        
    def run(self, max_trials: int) -> bool:
        
        self.trial_count = 0
        while not self.environment.is_solved() and self.trial_count < max_trials:
            
            self.logger.log(f"TRIAL {self.trial_count + 1}")
            self.logger.log(f"partcle ids: {[p.id for p in self.particles]}")
            self.logger.log(f"particle weights: {[p.weight for p in self.particles]}")

            #if self.trial_count == 0:
            #    key, box = self.environment.id_to_key['red'], self.environment.id_to_box['red']

            (key, box) = self._select_action()
            outcome = self.environment.test_action(key, box)

            self.logger.log(f"Action chosen: ({key.id}, {box.id})")
            self.logger.log(f"Outcome: {outcome}")

            self._update_particle_weights(key, box, outcome)

            self.logger.log(f"partcle ids: {[p.id for p in self.particles]}")
            self.logger.log(f"particle weights: {[p.weight for p in self.particles]}")

            self.trial_count += 1

        return self.environment.is_solved()
