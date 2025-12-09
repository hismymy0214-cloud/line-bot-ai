import os
import re
import pandas as pd
from rapidfuzz import fuzz, process

print("[DEBUG] bot_core.py loaded!")

# 設定訓練檔路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("TRAINING_FILE", "training.xlsx")
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)

print(f"[DEBUG] Expect training file at: {DATA_PATH}")


def _load_knowledge() -> pd.DataFrame:
    print(f"[DEBUG] Trying to load training file at: {DATA_PATH}")
    try:
        df = pd.read_excel(DATA_PATH)
        print(f"[DEBUG] File loaded successfully! Rows={len(df)}, Columns={df.columns.tolist()}")
    except Exception as e:
        print(f"[ERROR] Failed to read Excel file: {e}")
        return pd.DataFrame(columns=["category", "year", "unit", "item", "value", "description"])

    df = df.fillna("")
    return df


# 啟動時讀資料
_KNOWLEDGE = _load_knowledge()
print(f"[DEBUG] Knowledge loaded. Total rows: {len(_KNOWLEDGE)}")


def _extract_year(text: str):
    m = re.search(r"(\d{3})年", text)
    if m:
        return m.group(1)
    return None


def _fuzzy_match(question: str, choices: list):
    """
    模糊比對工具：回傳最相似的字串
    """
    if not choices:
        return None
    result = process.extractOne(question, choices, scorer=fuzz.partial_ratio)
    if result and result[1] >= 60:  # 相似度門檻 60 分
        return result[0]
    return None


def _find_best_row(question: str):
    text = question.strip()
    if not text:
        return None

    df = _KNOWLEDGE
    if df.empty:
        print("[DEBUG] Knowledge DataFrame is EMPTY.")
        return None

    candidates = df.copy()

    # 年度（仍維持精準比對）
    year = _extract_year(text)
    if year:
        candidates = candidates[candidates["year"].astype(str) == year]

    # 🔍 模糊比對 unit
    units = candidates["unit"].unique().tolist()
    best_unit = _fuzzy_match(text, units)

    if best_unit:
        candidates = candidates[candidates["unit"] == best_unit]

    # 🔍 模糊比對 item
    items = candidates["item"].unique().tolist()
    best_item = _fuzzy_match(text, items)

    if best_item:
        candidates = candidates[candidates["item"] == best_item]

    if candidates.empty:
        print("[DEBUG] No matching candidates found.")
        return None

    return candidates.iloc[0]


def build_reply(question: str) -> str:
    row = _find_best_row(question)
    if row is None:
        return "抱歉，我在訓練資料裡找不到這個問題的答案，可以換個說法或問別的問題喔。"

    parts = []

    if row.get("category"):
        parts.append(f"【類別】{row['category']}")
    if row.get("year"):
        parts.append(f"【年度】{row['year']} 年")
    if row.get("unit"):
        parts.append(f"【單位】{row['unit']}")
    if row.get("item"):
        parts.append(f"【項目】{row['item']}")
    if row.get("value") not in ("", None):
        parts.append(f"【數值】{row['value']}")
    if row.get("description"):
        parts.append(str(row["description"]))

    return "\n".join(parts)
