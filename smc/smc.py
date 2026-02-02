from this import d
from environment import Environment, Key, Box
from typing import List, Optional, Dict, Tuple
from generator import Generator, default_hypothesis_codes
from code_utils import execute_hypothesis_code
import math, random


class Particle:

    def __init__(self, code: str, weight: float = 1.0):
        self.code = code
        self.weight = weight
    
    def evaluate(self, key: Key, box: Box) -> bool:
        result = execute_hypothesis_code(self.code, key, box)
        return result



class Engine:

    def __init__(self, config: Dict, environment: Environment, llm: Generator = None):
        
        self.num_particles: int = config.num_particles
        self.theta: float = config.theta
        self.ess_threshold: float = config.ess_threshold

        self.environment: Environment = environment
        self.llm: Generator = llm

        self.particles = self._initialize_particles()

        
    def _initialize_particles(self) -> List[Particle]:

        if self.llm:
            hypothesis_codes = self.llm.generate_hypotheses(self.num_particles)
        else:
            hypothesis_codes = default_hypothesis_codes[:self.num_particles]

        particles = [Particle(code=code, weight=(1.0 / self.num_particles)) for code in hypothesis_codes]
        return particles


    def _resample(self):
        weights = [p.weight for p in self.particles]

        if sum(weights) == 0:
            indices = list(range(self.num_particles))
        else:
            indices = list()
            for _ in range(self.num_particles):
                indices.append(random.choice(range(self.num_particles), weights=weights)[0])
        
        resampled = list()
        for i in indices:
            new_particle = Particle(code=self.particles[i].code, weight=(1.0 / self.num_particles))
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


    def _compute_inspect_information_gain(self, box: Box) -> float:
        """
        calculate information gain by inspect box action
        """
        pass
        

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
                likelihood = self.theta if pred_outcome == outcome else (1 - self.theta)
                updated_weights.append(particle.weight * likelihood)

            # normalize updated weights
            total = sum(updated_weights)
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
            likelihood = self.theta if pred_outcome == outcome else (1 - self.theta)
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
        max_info_gain = float('inf')
        best_action = None
        for (key, box) in actions:
            if (key.id, box.id) in self.environment.success_pairs: # box already opened with key
                continue
            else:
                info_gain = self._compute_info_gain(key, box)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_action = (key, box)
        return best_action

        

    def run(self, max_trials: int = None) -> bool:
        
        self.trial_count = 0
        while not self.environment.is_solved() and self.trial_count < max_trials:
            (key, box) = self._select_action()
            outcome = self.environment.test_action(key, box)

            self.update_particle_weights(key, box, outcome)

            self.trial_count += 1

        return self.environment.is_solved()
