from openai import OpenAI
from config import MODEL_TYPE
from config import DOUBAO_API_KEY, DOUBAO_BASE_URL, DOUBAO_MAIN_MODEL, DOUBAO_FAST_MODEL
from config import OPENAI_API_KEY, OPENAI_MAIN_MODEL, OPENAI_FAST_MODEL


def call_llm(prompt: str, fast: bool = False) -> str:
    """
    fast=False → 使用主模型（生成初稿）
    fast=True  → 使用快速模型（去AI化/评分），config 中留空则自动回退到主模型
    """
    if MODEL_TYPE == "doubao":
        return _call_doubao(prompt, fast)
    elif MODEL_TYPE == "openai":
        return _call_openai(prompt, fast)
    else:
        raise ValueError(f"未知 MODEL_TYPE：{MODEL_TYPE}，请在 config.py 中填写 'doubao' 或 'openai'")


def _call_doubao(prompt: str, fast: bool = False) -> str:
    model = (DOUBAO_FAST_MODEL or DOUBAO_MAIN_MODEL) if fast else DOUBAO_MAIN_MODEL
    try:
        client = OpenAI(api_key=DOUBAO_API_KEY, base_url=DOUBAO_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"豆包调用失败（{'快速' if fast else '主'}模型）：", e)
        return ""


def _call_openai(prompt: str, fast: bool = False) -> str:
    model = (OPENAI_FAST_MODEL or OPENAI_MAIN_MODEL) if fast else OPENAI_MAIN_MODEL
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT调用失败（{'快速' if fast else '主'}模型）：", e)
        return ""
