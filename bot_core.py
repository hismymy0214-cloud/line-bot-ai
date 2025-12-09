# bot_core.py
import os
import re
import pandas as pd

# 設定訓練檔路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("TRAINING_FILE", "training.xlsx")
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)


def _load_knowledge() -> pd.DataFrame:
    """讀取訓練檔，如果找不到就回傳空的 DataFrame。"""
    try:
        df = pd.read_excel(DATA_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=["category", "year", "unit", "item", "value", "description"])

    df = df.fillna("")

    # 確保欄位都存在
    for col in ["category", "year", "unit", "item", "value", "description"]:
        if col not in df.columns:
            df[col] = ""

    return df


_KNOWLEDGE = _load_knowledge()


def _extract_year(text: str):
    """從問題中抓出『113年』這種字樣。"""
    m = re.search(r"(\d{3})年", text)
    if m:
        return m.group(1)
    return None


def _find_best_row(question: str):
    """很簡單的規則比對，從訓練表中挑一列當答案。"""
    text = question.strip()
    if not text:
        return None

    df = _KNOWLEDGE
    if df.empty:
        return None

    candidates = df

    # 年度
    year = _extract_year(text)
    if year:
        candidates = candidates[candidates["year"].astype(str) == year]

    # 單位
    for u in df["unit"].dropna().unique():
        if u and u in text:
            candidates = candidates[candidates["unit"] == u]
            break

    # 項目
    for it in df["item"].dropna().unique():
        if it and it in text:
            candidates = candidates[candidates["item"] == it]
            break

    if candidates.empty:
        return None

    # 目前先取第一筆
    return candidates.iloc[0]


def build_reply(question: str) -> str:
    """對外提供給 app.py 使用的函式。"""
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
