import os
import re
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

from llm.prompt import env_prompt, sys_prompt

load_dotenv()


class LLM:
    def __init__(
        self,
        model: str = "qwen-plus",
        temperature: float = 0.2,
        max_tokens: int = 20,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []

        if "gpt" in model:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif "deepseek" in model:
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
        elif "qwen" in model:
            self.client = OpenAI(
                api_key=os.getenv("QWEN_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def reset_session(self):
        self.messages = []

    def start_session(self, env_description: Optional[str] = None):
        """
        Start one run as one chat session.
        system/environment prompt is inserted only once.
        """
        self.reset_session()

        if env_description is None:
            env_description = env_prompt

        init_user_prompt = (
            f"{env_description}\n\n"
            "You are solving the box task trial by trial.\n"
            "At each step, propose exactly one key-box attempt.\n"
            "Return exactly one line in this format:\n"
            "KEY=<key_id>, BOX=<box_id>\n"
            "Do not explain.\n"
            "Do not output code.\n"
            "Do not output anything else."
        )

        self.messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": init_user_prompt},
        ]

    def _chat(self) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _parse_action(self, text: str) -> Tuple[str, str]:
        """
        Expected:
        KEY=A, BOX=3
        """
        text = text.strip()

        match = re.search(
            r"KEY\s*=\s*([A-Za-z0-9_\-]+)\s*,\s*BOX\s*=\s*([A-Za-z0-9_\-]+)",
            text
        )
        if match:
            return match.group(1), match.group(2)

        key_match = re.search(r"KEY\s*[:=]?\s*([A-Za-z0-9_\-]+)", text)
        box_match = re.search(r"BOX\s*[:=]?\s*([A-Za-z0-9_\-]+)", text)

        if key_match and box_match:
            return key_match.group(1), box_match.group(1)

        raise ValueError(f"Could not parse action from model output: {text}")

    def propose_action(self) -> Tuple[str, str]:
        """
        Ask for the next key-box action.
        """
        raw = self._chat()
        key_id, box_id = self._parse_action(raw)

        self.messages.append({"role": "assistant", "content": raw})
        return key_id, box_id

    def report_outcome(
        self,
        key_id: str,
        box_id: str,
        opened: bool,
        opened_boxes: Optional[List[str]] = None,
        tried_pairs: Optional[List[Tuple[str, str, bool]]] = None,
    ):
        """
        Feed the environment result back to the model.
        """
        result_line = (
            f"Result: KEY={key_id}, BOX={box_id} -> OPENED"
            if opened else
            f"Result: KEY={key_id}, BOX={box_id} -> FAILED"
        )

        extra_lines = []
        if opened_boxes is not None:
            extra_lines.append(f"Opened boxes so far: {opened_boxes}")
        if tried_pairs is not None:
            extra_lines.append(f"Trial history: {tried_pairs}")

        feedback = result_line
        if extra_lines:
            feedback += "\n" + "\n".join(extra_lines)

        feedback += (
            "\nNow propose the next action."
            "\nReturn exactly one line in the format: KEY=<key_id>, BOX=<box_id>"
        )

        self.messages.append({"role": "user", "content": feedback})

    def report_invalid_action(self, bad_key_id: str, bad_box_id: str, valid_key_ids: List[str], valid_box_ids: List[str]):
        msg = (
            f"Invalid action: KEY={bad_key_id}, BOX={bad_box_id}.\n"
            f"Valid keys: {valid_key_ids}\n"
            f"Valid boxes: {valid_box_ids}\n"
            "Try again.\n"
            "Return exactly one line in the format: KEY=<key_id>, BOX=<box_id>"
        )
        self.messages.append({"role": "user", "content": msg})

    def compress_history(
        self,
        opened_boxes: List[str],
        tried_pairs: List[Tuple[str, str, bool]]
    ):
        """
        Optional: keep the conversation from growing forever.
        Keeps only the initial system/user messages + a summary.
        """
        if len(self.messages) <= 2:
            return

        summary = (
            f"Summary so far:\n"
            f"Opened boxes: {opened_boxes}\n"
            f"Trials: {tried_pairs}\n"
            "Continue solving.\n"
            "Return exactly one line in the format: KEY=<key_id>, BOX=<box_id>"
        )

        self.messages = self.messages[:2] + [
            {"role": "user", "content": summary}
        ]