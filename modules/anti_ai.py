import os
from modules.llm import call_llm


def _load(name: str) -> str:
    path = os.path.join("prompts", "anti_ai", name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def anti_ai_pipeline(text: str) -> str:
    """单次调用完成全部去AI化（使用快速模型）"""
    prompt = _load("combined.txt").replace("{text}", text)
    return call_llm(prompt, fast=True)
