import os
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config import BRAND, OUTPUT_PER_KEYWORD, CONCURRENT_WORKERS, BATCH_SIZE
from modules.keyword import expand_one
from modules.generator import generate_article
from modules.progress import (
    console, show_header, show_total,
    show_article_header, show_saved, show_error, step, show_done,
)


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_article(platform: str, title: str, content: str, task_id: int):
    date = datetime.now().strftime("%m%d")
    folder = os.path.join("output", date)
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"{platform}_{task_id}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n{content}")

    show_saved(filename)


def process_task(task_id: int, base_keyword: str, platform: str) -> None:
    """单篇生成任务，供线程池调用"""
    # 先将核心词扩展为一个具体的长尾词，再用长尾词生成文章
    with step(f"#{task_id} 扩展长尾词"):
        keyword = expand_one(base_keyword)
    show_done(f"#{task_id} 长尾词", keyword)

    platform_prompt = load_prompt(f"prompts/platform/{platform}.txt")
    show_article_header(platform, keyword, task_id)
    title, article = generate_article(keyword, platform, platform_prompt)
    for _ in range(OUTPUT_PER_KEYWORD):
        save_article(platform, title, article, task_id)


def run():
    show_header(BRAND)

    # 直接读取 Excel 中所有关键词，随机打乱顺序
    df = pd.read_excel("data/keywords.xlsx")
    keywords = df["keyword"].dropna().tolist()
    random.shuffle(keywords)

    platforms = ["zhihu", "sohu", "baijiahao", "toutiao"]

    # 构建任务列表：随机抽取 BATCH_SIZE 篇（关键词 × 平台随机组合）
    pairs = [(kw, p) for kw in keywords for p in platforms]
    random.shuffle(pairs)
    tasks = [(i + 1, kw, p) for i, (kw, p) in enumerate(pairs[:BATCH_SIZE])]

    show_total(len(tasks))

    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = {
            executor.submit(process_task, tid, kw, p): (tid, kw, p)
            for tid, kw, p in tasks
        }

        for future in as_completed(futures):
            tid, kw, p = futures[future]
            try:
                future.result()
            except Exception as e:
                show_error(f"#{tid} {p} / {kw[:20]} 失败：{e}")


if __name__ == "__main__":
    run()
