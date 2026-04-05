from llm.llm_vs2 import LLM


class SPBaseline:
    def __init__(self, env, logger, model_name="qwen-plus", temperature=0.2, max_tokens=20):
        self.env = env
        self.logger = logger
        self.llm = LLM(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_env_description(self) -> str:
        key_ids = [str(k.id) for k in self.env.keys]
        box_ids = [str(b.id) for b in self.env.boxes]

        return (
            "Environment description:\n"
            f"Available keys: {key_ids}\n"
            f"Available boxes: {box_ids}\n"
            "Goal: discover which keys open which boxes.\n"
            "You may try one key on one box per trial.\n"
        )

    def _get_key_by_id(self, key_id: str):
        for key in self.env.keys:
            if str(key.id) == str(key_id):
                return key
        return None

    def _get_box_by_id(self, box_id: str):
        for box in self.env.boxes:
            if str(box.id) == str(box_id):
                return box
        return None

    def _try_open(self, key, box) -> bool:
        """
        Replace this function body with your actual environment call.

        Common possibilities:
            return self.env.open(key, box)
            return self.env.try_key(key, box)
            return self.env.step(key, box)
            return box.can_open(key)
        """
        return self.env.open(key, box)

    def run(self, max_trials=70):
        opened_boxes = set()
        success_pairs = []
        tried_pairs = []

        env_description = self._build_env_description()
        self.llm.start_session(env_description=env_description)

        valid_key_ids = [str(k.id) for k in self.env.keys]
        valid_box_ids = [str(b.id) for b in self.env.boxes]

        for trial_idx in range(1, max_trials + 1):
            try:
                proposed_key_id, proposed_box_id = self.llm.propose_action()
            except Exception as e:
                print(f"[Trial {trial_idx}] Failed to parse LLM output: {e}")
                break

            key = self._get_key_by_id(proposed_key_id)
            box = self._get_box_by_id(proposed_box_id)

            if key is None or box is None:
                self.llm.report_invalid_action(
                    bad_key_id=proposed_key_id,
                    bad_box_id=proposed_box_id,
                    valid_key_ids=valid_key_ids,
                    valid_box_ids=valid_box_ids,
                )
                continue

            opened = self._try_open(key, box)

            tried_pairs.append((str(key.id), str(box.id), bool(opened)))

            if opened:
                opened_boxes.add(str(box.id))
                success_pairs.append((str(key.id), str(box.id)))

            self.llm.report_outcome(
                key_id=str(key.id),
                box_id=str(box.id),
                opened=bool(opened),
                opened_boxes=sorted(opened_boxes),
                tried_pairs=tried_pairs,
            )

            # Optional: compress every 10 trials to reduce context growth
            if trial_idx % 10 == 0:
                self.llm.compress_history(
                    opened_boxes=sorted(opened_boxes),
                    tried_pairs=tried_pairs,
                )

            if len(opened_boxes) == len(self.env.boxes):
                return {
                    "solved": True,
                    "trials": trial_idx,
                    "opened": len(opened_boxes),
                    "success_pairs": success_pairs,
                }

        return {
            "solved": False,
            "trials": max_trials,
            "opened": len(opened_boxes),
            "success_pairs": success_pairs,
        }