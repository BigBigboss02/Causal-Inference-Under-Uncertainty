import random
import re
from typing import List, Optional

from environment import Environment
from llm.code import check_valid_program, execute_hypothesis_code
from llm.llm import LLM
from llm.prompt_partially_observed import action_prompt, env_prompt, starter_code_prompt, sys_prompt


class SPBaselinePartiallyObserved:
    """
    Same idea as `smc/sp_baseline.py`, but the LLM can also request an Observe action:

    - If LLM outputs: "PICK UP <box_id>"
      we reveal the true number of faces from `environment.py` and append a single
      evidence line:
        "Examined box <id>, <n> faces have <shape> shape"

    - Otherwise, the LLM output is treated as a hypothesis program for `predict(key, box)`
      and the rest of the loop is the same as `SPBaseline`.

    This class only updates in-memory state and appends each step to ``history``.
    Persisting JSON/CSV is the caller's responsibility (see ``driver_sp_baseline_partially_observed.py``).
    """

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
        self.evidence_lines: List[str] = []

    def _log(self, msg: str) -> None:
        try:
            self.logger.log(msg)
        except Exception:
            pass
        print(msg)

    def _parse_pick_up(self, text: str) -> Optional[str]:
        m = re.match(r"^\s*PICK\s+UP\s+([A-Za-z0-9_\-]+)\s*$", text, flags=re.IGNORECASE)
        if not m:
            return None
        return m.group(1)

    def _build_user_prompt(self) -> str:
        history = "\n".join(self.evidence_lines).strip()
        user_prompt = env_prompt + action_prompt[0] + starter_code_prompt + action_prompt[1]
        user_prompt += (history + "\n") if history else "(no prior actions/evidence)\n"
        return user_prompt

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

            user_prompt = self._build_user_prompt()
            response = self.llm.get_openai_completion(sys_prompt, user_prompt)
            self._log(f"LLM response:\n{response}")

            box_to_examine = self._parse_pick_up(response)
            if box_to_examine is not None:
                if box_to_examine not in self.env.id_to_box:
                    self._log(f"Invalid PICK UP target: {box_to_examine}")
                else:
                    box = self.env.id_to_box[box_to_examine]
                    box.number = set([box.count])
                    line = f"Examine {box_to_examine}: {box.count} faces have shape on them."
                    self.evidence_lines.append(line)
                    self._log(line)

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

            hypothesis = response
            for _ in range(3):
                if check_valid_program(hypothesis):
                    break
                hypothesis = self.llm.get_openai_completion(sys_prompt, user_prompt)

            if not check_valid_program(hypothesis):
                self._log("Failed to get a valid hypothesis program; skipping trial.")
                self.trial_count += 1
                continue

            self.hypothesis = hypothesis
            self._log(f"Hypothesis:\n{self.hypothesis}")

            key, box = self._select_action()
            outcome = self.env.test_action(key, box)

            status = "success" if outcome is True else "failure"
            self.evidence_lines.append(f"Open box {box.id} with key {key.id}: {status}")

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
