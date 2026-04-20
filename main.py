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

    # 读取 Excel 中所有关键词，按科室分类
    df = pd.read_excel("data/keywords.xlsx")
    
    # 确保有 "科室" 列
    if "科室" not in df.columns:
        # 如果没有科室列，默认按关键词内容简单分类
        def classify_keyword(kw):
            kw = str(kw).lower()
            if any(word in kw for word in ["妇科", "白带", "月经", "人流", "备孕", "宫颈", "妇科检查"]):
                return "妇科"
            elif any(word in kw for word in ["皮肤", "痘痘", "皮炎", "湿疹", "痤疮", "毛囊炎", "激光", "光子"]):
                return "皮肤科"
            else:
                return "常规体检"
        df["科室"] = df["keyword"].apply(classify_keyword)
    
    # 按科室分组
    gyn_keywords = df[df["科室"] == "妇科"]["keyword"].dropna().tolist()
    derm_keywords = df[df["科室"] == "皮肤科"]["keyword"].dropna().tolist()
    exam_keywords = df[df["科室"] == "常规体检"]["keyword"].dropna().tolist()
    
    # 计算各科室需要的数量
    total = BATCH_SIZE
    gyn_count = max(1, int(total * 0.3))  # 至少1个
    derm_count = max(1, int(total * 0.6))  # 至少1个
    exam_count = max(1, total - gyn_count - derm_count)  # 至少1个
    
    # 随机抽取关键词
    random.shuffle(gyn_keywords)
    random.shuffle(derm_keywords)
    random.shuffle(exam_keywords)
    
    # 确保有足够的关键词
    def ensure_enough(keywords, count, category):
        if len(keywords) < count:
            # 如果关键词不足，重复使用
            while len(keywords) < count:
                keywords.extend(keywords[:count - len(keywords)])
        return keywords[:count]
    
    selected_gyn = ensure_enough(gyn_keywords, gyn_count, "妇科")
    selected_derm = ensure_enough(derm_keywords, derm_count, "皮肤科")
    selected_exam = ensure_enough(exam_keywords, exam_count, "常规体检")
    
    selected_kw = selected_gyn + selected_derm + selected_exam
    random.shuffle(selected_kw)

    platforms = ["zhihu", "sohu", "baijiahao", "toutiao"]

    # 构建任务列表：随机抽取 BATCH_SIZE 篇（关键词 × 平台随机组合）
    pairs = [(kw, p) for kw in selected_kw for p in platforms]
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
