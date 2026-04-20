import os
import random
from modules.llm import call_llm

_VARIATION_STYLES = [
    "从个人经历角度重新表达",
    "从旁观者/帮朋友查资料的角度表达",
    "用更平淡、不确定的语气表达",
    "用更口语化、随意的方式表达",
]


def _load(name: str) -> str:
    path = os.path.join("prompts", "rewriter", name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def rewrite_text(text: str) -> str:
    """句式级改写：完全改变句式，保留核心意思"""
    prompt = _load("rewrite.txt").replace("{text}", text)
    return call_llm(prompt)


def restructure_text(text: str) -> str:
    """结构级优化：轻度重排段落，增加过渡"""
    prompt = _load("restructure.txt").replace("{text}", text)
    return call_llm(prompt)


def semantic_variation(text: str) -> str:
    """语义路径变体：换一种表达角度"""
    style = random.choice(_VARIATION_STYLES)
    prompt = _load("variation.txt").replace("{text}", text).replace("{style}", style)
    return call_llm(prompt)


def apply_random_rewrite(text: str) -> str:
    """随机选择一种重写方式（低分重试时调用）"""
    fn = random.choice([rewrite_text, restructure_text, semantic_variation])
    return fn(text)
