from sre_parse import SUCCESS
from environment import Environment
from llm.llm import LLM
from llm.code import execute_hypothesis_code
import random

class SPBaseline:

    def __init__(self, env: Environment, logger):

        self.env = env
        self.llm = LLM()
        self.logger = logger

        self.hypothesis: str = None
        self.evidence = list()

    def _select_action(self):

        opened = set([pair[1] for pair in self.env.success_pairs])
        candidate_actions = list()
        fallback_actions = list()
        for (key, box) in self.env.actions:
            if box.id not in opened:
                if execute_hypothesis_code(self.hypothesis, key, box) is True:
                    candidate_actions.append((key, box))
                else:
                    fallback_actions.append((key, box))
        
        if len(candidate_actions) > 0:
            return random.choice(candidate_actions)
        else:
            return random.choice(fallback_actions)

    def _accept_h(self) -> bool:
        # check consistency with opened boxes
        for (key_id, box_id) in self.env.success_pairs:
            key, box = self.env.id_to_key[key_id], self.env.id_to_box[box_id]
            if execute_hypothesis_code(self.hypothesis, key, box) is False:
                return False
            
        # check consistency with failure evidence
        return True


    def run(self, max_trials: int) -> dict:
        self.trial_count = 0
        self.history = []

        while not self.env.is_solved() and self.trial_count < max_trials:
            self.logger.log(f"TRIAL {self.trial_count}")

            if self.trial_count == 0:
                self.hypothesis, h_name = self.llm.generate([])
            else:
                while True:
                    self.hypothesis, new_name = self.llm.refine(self.evidence, self.hypothesis)
                    if self._accept_h():
                        break

            self.logger.log(f"{self.hypothesis}")

            key, box = self._select_action()
            outcome = self.env.test_action(key, box)

            self.evidence.append((key, box, outcome))

            self.logger.log(f"Action chosen: ({key.id}, {box.id})")
            self.logger.log(f"Outcome: {outcome}")
            self.logger.log(f"Boxes opened: {self.env.success_pairs}")

            self.trial_count += 1

            self.history.append({
                "t": self.trial_count,
                "opened": len(self.env.success_pairs),
                "hypothesis": self.hypothesis,
                "action": [key.id, box.id],
                "outcome": bool(outcome),
            })

        return {
            "solved": self.env.is_solved(),
            "trials": self.trial_count,
            "opened": len(self.env.success_pairs),
            "success_pairs": list(self.env.success_pairs),
            "history": self.history,
        }