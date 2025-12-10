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
        # 保留欄位結構，避免後續程式炸掉
        return pd.DataFrame(columns=["category", "year", "unit", "item", "value", "description"])

    df = df.fillna("")
    # 加一個 year_str 欄位方便做索引（全部轉成字串）
    df["year_str"] = df["year"].astype(str)
    return df


# 啟動時讀資料
_KNOWLEDGE: pd.DataFrame = _load_knowledge()
print(f"[DEBUG] Knowledge loaded. Total rows: {len(_KNOWLEDGE)}")

# 建立 (year_str, unit, item) 的 MultiIndex，做精準查詢用
if not _KNOWLEDGE.empty:
    _INDEX_BY_KEY = _KNOWLEDGE.set_index(["year_str", "unit", "item"])
    print(f"[DEBUG] Key index built. Index size: {_INDEX_BY_KEY.shape[0]}")
else:
    _INDEX_BY_KEY = pd.DataFrame()
    print("[DEBUG] Knowledge is empty. Key index not built.")


def _extract_year(text: str):
    """
    從問題文字中抓出「113年」這種 3 碼年度。
    如果只有兩碼（例如 13年），就自動補成 113。
    """
    m = re.search(r"(\d{2,3})年", text)
    if not m:
        return None

    y = m.group(1)
    # 兩位數的年度，自動前面補 1 -> 13 -> 113
    if len(y) == 2:
        y = "1" + y
    return y


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
    """
    走原本的「自然語言 + 模糊比對」流程，
    依序比對 year、unit、item，回傳最符合的那一列。
    """
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


def _lookup_by_key(year: str, unit: str, item: str):
    """
    用 (year, unit, item) 精準查詢一列資料。
    year 允許輸入 13 / 113 之類，最後會轉成 year_str。
    """
    if _INDEX_BY_KEY.empty:
        print("[DEBUG] _INDEX_BY_KEY is EMPTY.")
        return None

    y = str(year).strip()
    if len(y) == 2:  # 13 -> 113
        y = "1" + y

    u = unit.strip()
    i = item.strip()

    try:
        row = _INDEX_BY_KEY.loc[(y, u, i)]
        # 如果剛好有重複 key，loc 可能回 DataFrame，取第一列
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row
    except KeyError:
        print(f"[DEBUG] Key not found: year={y}, unit={u}, item={i}")
        return None


def _format_row(row: pd.Series) -> str:
    """
    將一列資料轉成要回傳給 LINE 的文字。
    """
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


def build_reply(question: str) -> str:
    """
    對外主入口：
    - 若使用者輸入格式為：#查 年度,單位,項目 -> 走精準 key 查詢
    - 否則走原本的自然語言模糊比對
    """
    text = question.strip()

    #1️⃣ 特殊指令：#查 年度,單位,項目
    if text.startswith("#查"):
        payload = text[2:].strip()  # 去掉 "#查"
        # 支援中文、英文逗號
        parts = [p.strip() for p in re.split(r"[,，]", payload) if p.strip()]

        if len(parts) != 3:
            return (
                "格式錯誤，請用：#查 年度,單位,項目\n"
                "例如：#查 113,工務局,職員總數"
            )

        year, unit, item = parts
        row = _lookup_by_key(year, unit, item)

        if row is None:
            return (
                f"找不到符合條件的資料：\n"
                f"年度={year}，單位={unit}，項目={item}\n"
                "請確認 training.xlsx 是否有這一筆。"
            )

        return _format_row(row)

    # 2️⃣ 一般使用者：走模糊查詢
    row = _find_best_row(text)
    if row is None:
        return "抱歉，我在訓練資料裡找不到這個問題的答案，可以換個說法或問別的問題喔。"

    return _format_row(row)