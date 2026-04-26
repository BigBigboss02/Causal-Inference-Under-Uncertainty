import random
from typing import List, Optional

from environment import Environment
from llm.code import execute_hypothesis_code
from llm.llm import LLM


class SPBaseline:

    def __init__(
        self,
        env,
        logger,
        model_name="gpt-5.2",
        temperature=0.2,
        max_tokens=200,
    ):
        self.env: Environment = env
        self.logger = logger

        self.llm = LLM(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.hypothesis: Optional[str] = None
        self.evidence = list()
        self.evidence_lines: List[str] = []

    def _log(self, msg: str) -> None:
        try:
            self.logger.log(msg)
        except Exception:
            pass
        print(msg)

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
        self._interaction_seq = 0
        self.history = []

        while not self.env.is_solved() and self.trial_count < max_trials:
            self._log(f"TRIAL {self.trial_count}")

            if self.evidence_lines:
                self._log("Evidence lines (included in prompt):")
                for i, line in enumerate(self.evidence_lines, start=1):
                    self._log(f"{i}. {line}")
            else:
                self._log("Evidence lines (included in prompt): (none)")

            if self.trial_count == 0:
                self.hypothesis, h_name = self.llm.generate([])
            else:
                while True:
                    self.hypothesis, new_name = self.llm.refine(
                        self.evidence, self.hypothesis
                    )
                    if self._accept_h():
                        break

            self._log(f"LLM response:\n{self.hypothesis}")
            self._log(f"Hypothesis:\n{self.hypothesis}")

            key, box = self._select_action()
            outcome = self.env.test_action(key, box)

            self.evidence.append((key, box, outcome))
            status = "success" if outcome is True else "failure"
            self.evidence_lines.append(
                f"Open box {box.id} with key {key.id}: {status}"
            )

            self._log(f"Action chosen: ({key.id}, {box.id})")
            self._log(f"Outcome: {outcome}")
            self._log(f"Boxes opened: {self.env.success_pairs}")

            self.trial_count += 1
            self._interaction_seq += 1
            self.history.append(
                {
                    "t": self._interaction_seq,
                    "opened": len(self.env.success_pairs),
                    "hypothesis": self.hypothesis,
                    "action": [key.id, box.id],
                    "outcome": bool(outcome),
                    "llm_response": self.hypothesis,
                    "evidence_lines": list(self.evidence_lines),
                }
            )

        return {
            "solved": self.env.is_solved(),
            "trials": self.trial_count,
            "opened": len(self.env.success_pairs),
            "success_pairs": list(self.env.success_pairs),
            "history": self.history,
        }
