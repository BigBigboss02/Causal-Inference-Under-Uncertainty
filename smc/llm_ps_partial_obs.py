import random
import re
from typing import List, Optional

from environment import Environment
from llm.code import check_valid_program, execute_hypothesis_code
from llm.llm import LLM
from llm.prompt_partially_observed import action_prompt, env_prompt, starter_code_prompt, sys_prompt


class LlmPSP:

    def __init__(self, env, logger, llm: LLM):
        self.env: Environment = env
        self.llm: LLM = llm

        self.logger = logger

        self.hypothesis: Optional[str] = None
        self.evidence_lines: List[str] = list()

    def _parse_pick_up(self, text: str) -> Optional[str]:
        """
        search LLM response for pick up action
        """
        m = re.match(r"^\s*PICK\s+UP\s+([A-Za-z0-9_\-]+)\s*$", text, flags=re.IGNORECASE)
        if not m:
            return None
        return m.group(1)

    def _build_user_prompt(self) -> str:
        """
        build prompt for next LLM action
        """
        history = "\n".join(self.evidence_lines).strip()
        user_prompt = env_prompt + action_prompt[0] + starter_code_prompt + action_prompt[1]
        user_prompt += (history + "\n") if history else "(no prior actions/evidence)\n"
        return user_prompt

    def _select_action(self):
        """
        select opening action based on currently active hypothesis
        randomly select a key-box opening action consistent with hypothesis
        if none exists, perform random action
        """
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

    def run(self, max_trials: int) -> dict:
        self.trial_count = 0
        self._interaction_seq = 0
        self.history = []

        while not self.env.is_solved() and self.trial_count < max_trials:
            self.logger.log(f"Trial {self.trial_count}")

            # log observed evidence from opening/pick up actions
            if self.evidence_lines:
                self.logger.log("Evidence lines (included in prompt):")
                for i, line in enumerate(self.evidence_lines, start=1):
                    self.logger.log(f"{i}. {line}")
            else:
                self.logger.log("Evidence lines (included in prompt): (none)")

            # get next action from LLM
            user_prompt = self._build_user_prompt()
            response = self.llm.get_openai_completion(sys_prompt, user_prompt)

            self.logger.log(f"LLM response:\n{response}")

            box_to_examine = self._parse_pick_up(response)
            if box_to_examine is not None:
                # if PICK UP action is chosen
                if box_to_examine not in self.env.id_to_box:
                    self.logger.log(f"Invalid PICK UP: {box_to_examine}")
                else:
                    box = self.env.id_to_box[box_to_examine]
                    box.number = set([box.count])
                    line = f"Examine {box_to_examine}: {box.count} faces have shape on them."
                    self.evidence_lines.append(line)

                self._interaction_seq += 1
                self.history.append(
                    {
                        "t": self._interaction_seq,
                        "opened": len(self.env.success_pairs),
                        "action": ["examine", box_to_examine],
                        "llm_response": response,
                        "evidence_lines": list(self.evidence_lines),
                    }
                )
                continue
            
            # if LLM prefers to generate a new hypothesis
            hypothesis = response
            for _ in range(3):
                if check_valid_program(hypothesis):
                    break
                hypothesis = self.llm.get_openai_completion(sys_prompt, user_prompt)

            if not check_valid_program(hypothesis):
                self.logger.log("Failed to get a valid hypothesis program; skipping trial.")
                self.trial_count += 1
                continue

            self.hypothesis = hypothesis
            self.logger.log(f"Hypothesis:\n{self.hypothesis}")

            # select and test new trial
            key, box = self._select_action()
            outcome = self.env.test_action(key, box)
            
            # update evidence
            status = "success" if outcome is True else "failure"
            self.evidence_lines.append(f"Open box {box.id} with key {key.id}: {status}")

            self.logger.log(f"Action chosen: ({key.id}, {box.id})")
            self.logger.log(f"Outcome: {outcome}")
            self.logger.log(f"Boxes opened: {self.env.success_pairs}")

            self.trial_count += 1
            self._interaction_seq += 1
            self.history.append(
                {
                    "t": self._interaction_seq,
                    "opened": len(self.env.success_pairs),
                    "hypothesis": self.hypothesis,
                    "action": [key.id, box.id],
                    "outcome": bool(outcome),
                    "llm_response": response,
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
