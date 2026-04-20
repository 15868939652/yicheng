import os
import random
from datetime import datetime
from functools import lru_cache

from config import BRAND, MIN_SCORE, MAX_RETRY
from modules.llm import call_llm
from modules.randomizer import random_profile, random_style, random_trigger
from modules.anti_ai import anti_ai_pipeline
from modules.rewriter import apply_random_rewrite
from modules.scorer import score_article
from modules.progress import show_params, show_done, show_score, show_retry, step


# ========================= 文章模式配置 =========================
# 每种模式对应一个 prompt 文件 + 选取权重
_MODES = {
    "info":       ("prompts/article_base.txt",      0.30),  # 查资料型，信息密度高
    "light_exp":  ("prompts/article_base.txt",      0.20),  # 轻体验型，有一点亲身了解
    "other_exp":  ("prompts/article_base.txt",      0.15),  # 转述他人经历
    "exp":        ("prompts/article_exp.txt",       0.15),  # 完整就诊经历分享
    "hesitate":   ("prompts/article_hesitate.txt",  0.12),  # 纠结型，还没决定
    "short":      ("prompts/article_short.txt",     0.08),  # 简短随手记
}

_MODE_NAMES   = list(_MODES.keys())
_MODE_WEIGHTS = [v[1] for v in _MODES.values()]


def _load(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=None)
def _load_examples(platform: str) -> str:
    """
    读取平台范文文件，提取已填写的范文内容。
    若文件不存在或尚未填写，返回空字符串。结果按平台缓存，避免重复读盘。
    """
    path = os.path.join("prompts", "examples", f"{platform}.txt")
    if not os.path.exists(path):
        return ""

    raw = _load(path)
    examples = []

    for block in raw.split("===范文")[1:]:       # 按分隔符切块
        content = block.split("===")[0]          # 取分隔符之间的内容
        # 去掉序号行（如 "1===\n"）
        lines = content.split("\n")
        body  = "\n".join(l for l in lines if not l.strip().endswith("===")).strip()
        if body:
            examples.append(body)

    if not examples:
        return ""

    parts = [f"【范文{i+1}】\n{ex}" for i, ex in enumerate(examples)]
    return (
        "\n\n---\n"
        "【风格参考范文：请学习以下内容的语气、结构和立场，不要复制任何具体内容】\n\n"
        + "\n\n".join(parts)
    )


def _parse_output(raw: str) -> tuple:
    """从 LLM 输出中解析标题和正文"""
    if "【标题】" in raw and "【正文】" in raw:
        title   = raw.split("【标题】")[1].split("【正文】")[0].strip()
        article = raw.split("【正文】")[1].strip()
    else:
        # 降级：首行作标题，其余作正文
        lines   = raw.strip().split("\n")
        title   = lines[0].lstrip("#").strip()
        article = "\n".join(lines[1:]).strip()
    return title, article


def _build_prompt(keyword: str, mode: str, profile: str, style: str,
                  trigger: str, platform: str, platform_prompt: str) -> str:
    """组装最终发给 LLM 的 prompt"""

    current_month = datetime.now().month

    prompt_file = _MODES[mode][0]
    base = _load(prompt_file)

    # article_base.txt 内置了 {brand} 和 {关键词} 占位符
    if mode in ("info", "light_exp", "other_exp"):
        base = base.replace("{brand}", BRAND).replace("{关键词}", keyword)
    else:
        base += (
            f"\n\n关键词：{keyword}"
            f"\n如需自然提及医院，使用：{BRAND}（只出现1次，以'看到/听说'形式，不评价好坏）"
            f"\n基调：整体中立偏正向，不得出现明显推荐/宣传语气；如提及不足，一笔带过，不展开"
        )

    examples_block = _load_examples(platform)

    title_instruction = (
        "\n\n---\n"
        "【输出格式要求（必须严格遵守）】\n"
        "【标题】两段式，逗号分隔，共15-28字：\n"
        "  第一段：以『义乌义城医院』开头 + 关键词延伸（疑问/经历/纠结风格，例：义乌义城医院靠谱吗 / 义乌义城医院皮肤科怎么样）\n"
        "  第二段：第一人称个人视角短语（例：我的一点个人见解 / 说说我的真实感受 / 来聊聊我的经历 / 记录一下我的体验）\n"
        "  不要营销感，不要感叹号\n"
        "【正文】\n"
        "（在此写正文内容）"
    )

    return (
        f"当前时间：{current_month}月（文中涉及的季节性活动、节假日、工作安排必须与此一致，"
        f"例如{current_month}月不应出现年会、春节、开学季等不符合时间的内容）\n"
        f"人设：{profile}\n"
        f"表达风格：{style}\n"
        f"触发背景：{trigger}\n"
        f"写作模式：{mode}\n\n"
        f"{base}\n\n"
        f"{platform_prompt}"
        f"{examples_block}"
        f"{title_instruction}"
    )


def generate_article(keyword: str, platform: str, platform_prompt: str):
    """
    生成一篇文章，返回 (title, article)。

    流程：
    1. 随机选模式 / 人设 / 风格 / 触发背景
    2. 调 LLM 生成初稿
    3. 三阶轻度去AI化
    4. 评分；不达标则重新生成 + 触发 rewriter，最多 MAX_RETRY 次
    5. 生成标题
    """

    # ----- Step 1: 随机参数 -----
    mode    = random.choices(_MODE_NAMES, weights=_MODE_WEIGHTS)[0]
    profile = random_profile(platform)
    style   = random_style(profile)
    trigger = random_trigger()
    prompt  = _build_prompt(keyword, mode, profile, style, trigger, platform, platform_prompt)

    show_params(mode, profile)

    # ----- Step 1: 生成初稿（含标题）-----
    with step("[1/3] 生成初稿 + 标题"):
        raw = call_llm(prompt)
    title, article = _parse_output(raw)
    short_title = (title[:28] + "…") if len(title) > 28 else title
    show_done("[1/3] 初稿完成", short_title)

    # ----- Step 2: 去AI化（单次调用，快速模型）-----
    with step("[2/3] 去AI化"):
        article = anti_ai_pipeline(article)
    show_done("[2/3] 去AI化完成")

    # ----- Step 3: 评分 & 重试 -----
    with step("[3/3] 质量评分"):
        score = score_article(article)
    show_score(score, MIN_SCORE)

    retry = 0
    while score < MIN_SCORE and retry < MAX_RETRY:
        retry += 1
        show_retry(retry, score, MAX_RETRY)

        with step("  重新生成 + 重写"):
            raw = call_llm(prompt)
            _, article = _parse_output(raw)   # 重试时标题沿用第一次的
            article = apply_random_rewrite(article)
            article = anti_ai_pipeline(article)

        with step("  重新评分"):
            score = score_article(article)
        show_score(score, MIN_SCORE)

    return title, article
