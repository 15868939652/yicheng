# ========================= 品牌配置 =========================
BRAND = "义乌义城医院"      # 核心品牌词，全局固定

# ========================= 模型配置 =========================
# 切换模型：改这一行，填 "doubao" 或 "openai"
MODEL_TYPE = "doubao"

# --- 豆包（火山引擎）---
DOUBAO_API_KEY    = "e4911c31-ef61-4d68-bf9e-07d430466a16"  # ← 填你的豆包 API Key
DOUBAO_BASE_URL   = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MAIN_MODEL = "doubao-seed-2-0-pro-260215"  # 主模型：用于生成初稿（最强）
DOUBAO_FAST_MODEL = "doubao-seed-2-0-lite-260215"                            # 快速模型：用于去AI化/评分，留空则沿用主模型
                                                  # 示例：填入你的豆包轻量版接入点 ID

# --- GPT（OpenAI）---
OPENAI_API_KEY    = ""          # ← 填你的 OpenAI API Key
OPENAI_MAIN_MODEL = "gpt-4o"    # 主模型：用于生成初稿
OPENAI_FAST_MODEL = ""          # 快速模型：用于去AI化/评分，如 "gpt-4o-mini"，留空则沿用主模型

# ========================= 生成控制 =========================
OUTPUT_PER_KEYWORD = 1           # 每个关键词每个平台生成几篇
MIN_SCORE          = 70          # 低于此分数触发重试
MAX_RETRY          = 2           # 最多重试次数
CONCURRENT_WORKERS = 2           # 并发篇数（建议 2-3，过高易触发 API 限流）
BATCH_SIZE         = 10          # 每次运行生成的总篇数（从关键词池随机抽取）
