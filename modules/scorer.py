import os
import re
from modules.llm import call_llm


def _load_prompt() -> str:
    path = os.path.join("prompts", "scorer.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def score_article(text: str) -> int:
    prompt = _load_prompt().replace("{text}", text)
    result = call_llm(prompt, fast=True)

    try:
        m = re.search(r"得分：\s*(\d+)", result)
        score = int(m.group(1)) if m else 60
    except Exception:
        score = 60

    return score
