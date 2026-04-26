import pathlib
import sys


def _ensure_imports_work() -> None:
    """
    This repo's scripts commonly run with cwd=smc/, which makes `llm.*` imports work.
    If this file is run from the repo root, add `smc/` to sys.path so `llm` resolves.
    """
    here = pathlib.Path(__file__).resolve()
    smc_dir = here.parent
    if (smc_dir / "llm").is_dir() and str(smc_dir) not in sys.path:
        sys.path.insert(0, str(smc_dir))


if __name__ == "__main__":
    _ensure_imports_work()

    from llm.llm import LLM
    from llm.prompt_partially_observed import (
        sys_prompt,
        env_prompt,
        action_prompt,
        starter_code_prompt,
    )

    # Optional: paste previous actions/evidence here (or leave empty).
    history = ""

    user_prompt = (
        env_prompt
        + action_prompt[0]
        + starter_code_prompt
        + action_prompt[1]
        + (history.strip() + "\n" if history.strip() else "(no prior actions/evidence)\n")
    )

    llm = LLM(model="gpt-5.2")
    print(llm.get_completion(sys_prompt, user_prompt))

