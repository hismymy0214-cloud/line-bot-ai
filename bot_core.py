import os
import re
import pandas as pd
from rapidfuzz import fuzz, process

print("[DEBUG] bot_core.py loaded!")

# -----------------------------
# 模糊比對分數門檻
# -----------------------------
UNIT_MIN_SCORE = 70   # 單位：相似度至少 70
ITEM_MIN_SCORE = 75   # 項目：相似度至少 75（放寬一點，避免太容易被當作抓不到）

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


def _extract_years(text: str):
    """
    從字串中抓出所有像 113、112 這種年度，回傳整數 list（已去重、排序）。
    兩位數年度會自動補成 1xx。
    """
    matches = re.findall(r"(\d{2,3})年?", text)
    years = []
    for y in matches:
        if len(y) == 2:
            y = "1" + y
        try:
            years.append(int(y))
        except ValueError:
            continue
    # 去重並排序
    years = sorted(set(years))
    return years


def _guess_category(text: str):
    """
    根據問題文字猜測要用哪一種 category（統計 / 預算 / 決算）。
    沒猜到就回 None，不強制。
    """
    if any(k in text for k in ["預算", "預算數", "預算書"]):
        return "預算"
    if any(k in text for k in ["決算", "執行數", "實際支出"]):
        return "決算"
    if "統計" in text:
        return "統計"
    # 沒特別講就不限制
    return None


def _fuzzy_match(question: str, choices: list):
    """
    模糊比對工具：回傳 (最相似的字串, 分數)。
    找不到則回 (None, 0)。
    """
    if not choices:
        return None, 0

    result = process.extractOne(question, choices, scorer=fuzz.partial_ratio)
    if not result:
        return None, 0

    best_choice, score, *_ = result  # rapidfuzz.extractOne 回傳 (choice, score, index)
    return best_choice, score


def _find_best_row(question: str):
    """
    「自然語言 + 模糊比對」查詢流程：
    1. 先試著用 category（統計 / 預算 / 決算）縮小範圍
    2. 再用年度過濾
    3. 模糊比對單位；若分數太低，當作查不到
    4. 模糊比對項目；若分數太低（代表沒有明確指定項目），當作查不到
    """
    text = question.strip()
    if not text:
        return None

    df = _KNOWLEDGE
    if df.empty:
        print("[DEBUG] Knowledge DataFrame is EMPTY.")
        return None

    candidates = df.copy()

    # 先依問題文字猜 category（例如有寫「預算」「決算」）
    cat = _guess_category(text)
    if cat:
        cat_filtered = candidates[candidates["category"] == cat]
        if not cat_filtered.empty:
            candidates = cat_filtered
            print(f"[DEBUG] Category hint applied: {cat} -> rows={len(candidates)}")
        else:
            print(f"[DEBUG] Category hint '{cat}' has no rows, fallback to all categories.")

    # 年度（維持精準比對）
    year = _extract_year(text)
    if year:
        before = len(candidates)
        candidates = candidates[candidates["year"].astype(str) == year]
        print(f"[DEBUG] Year filter: {year}, rows {before} -> {len(candidates)}")

    if candidates.empty:
        print("[DEBUG] No candidates after year/category filter.")
        return None

    # 🔍 模糊比對 unit
    unit_choices = candidates["unit"].unique().tolist()
    best_unit, unit_score = _fuzzy_match(text, unit_choices)
    print(f"[DEBUG] Fuzzy unit: best={best_unit}, score={unit_score}")

    if not best_unit or unit_score < UNIT_MIN_SCORE:
        # 單位都不確定，就直接放棄
        print(f"[DEBUG] Unit not matched clearly. score={unit_score}")
        return None

    candidates = candidates[candidates["unit"] == best_unit]
    if candidates.empty:
        print("[DEBUG] No candidates after unit filter.")
        return None

    # 🔍 模糊比對 item
    item_choices = candidates["item"].unique().tolist()
    best_item, item_score = _fuzzy_match(text, item_choices)
    print(f"[DEBUG] Fuzzy item: best={best_item}, score={item_score}")

    # ⬇⬇⬇ 關鍵：項目如果不夠明確，就視為查不到，不再硬湊 description
    if not best_item or item_score < ITEM_MIN_SCORE:
        print(f"[DEBUG] Item not matched clearly. score={item_score}")
        return None

    candidates = candidates[candidates["item"] == best_item]
    if candidates.empty:
        print("[DEBUG] No matching candidates after item filter.")
        return None

    row = candidates.iloc[0]
    print(
        "[DEBUG] Final match: "
        f"category={row['category']}, year={row['year']}, "
        f"unit={row['unit']}, item={row['item']}"
    )
    return row


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


def _build_compare_answer(text: str) -> str | None:
    """
    處理『113年跟112年比較』這類問題。
    回傳比較結果字串；若比對失敗則回傳 None，讓外面走原本流程。
    """
    df = _KNOWLEDGE
    if df.empty:
        return None

    years = _extract_years(text)
    if len(years) >= 2:
        old_year, new_year = years[0], years[-1]
    elif len(years) == 1:
        new_year = years[0]
        old_year = new_year - 1  # 只有一個年份時，假設要跟前一年比
    else:
        return None

    # 建一個「只保留新年度」的問題給 _find_best_row 用
    # 例如：「113年工務局主管預算數跟112年比較」->「113年工務局主管預算數」
    base_text = text
    # 把舊年度拿掉，但保留新年度
    base_text = re.sub(rf"{old_year}年?", "", base_text)
    # 移除常見比較關鍵詞
    base_text = re.sub(r"比較|差異|變化|增減|變動|成長|相比|對比|跟|與|和", "", base_text)

    row_new = _find_best_row(base_text)
    if row_new is None:
        return None

    cat = row_new.get("category")
    unit = row_new.get("unit")
    item = row_new.get("item")

    if not cat or not unit or not item:
        return None

    # 找舊年度那一列
    subset = df[
        (df["category"] == cat)
        & (df["unit"] == unit)
        & (df["item"] == item)
        & (df["year"].astype(str) == str(old_year))
    ]
    if subset.empty:
        print(
            f"[DEBUG] No old-year row found for compare: "
            f"cat={cat}, unit={unit}, item={item}, year={old_year}"
        )
        return None

    row_old = subset.iloc[0]

    def _to_number(v):
        try:
            return float(str(v).replace(",", ""))
        except Exception:
            return None

    v_new = _to_number(row_new.get("value"))
    v_old = _to_number(row_old.get("value"))

    # 若無法轉成數值，就至少把兩年數值列出來
    if v_new is None or v_old is None:
        parts = [
            f"【類別】{cat}",
            f"【比較項目】{unit}／{item}",
            f"【{old_year} 年數值】{row_old.get('value', '')}",
            f"【{new_year} 年數值】{row_new.get('value', '')}",
        ]
        if row_old.get("description"):
            parts.append(f"{old_year} 年說明：{row_old['description']}")
        if row_new.get("description"):
            parts.append(f"{new_year} 年說明：{row_new['description']}")
        return "\n".join(parts)

    diff = v_new - v_old
    pct = None
    if v_old != 0:
        pct = diff / v_old * 100.0

    parts = [
        f"【類別】{cat}",
        f"【比較項目】{unit}／{item}",
        f"【{old_year} 年數值】{v_old:,.0f}",
        f"【{new_year} 年數值】{v_new:,.0f}",
        f"【差額】{diff:+,.0f}",
    ]
    if pct is not None:
        parts.append(f"【成長率】{pct:+.2f}%（以 {old_year} 年為基準）")

    if row_old.get("description"):
        parts.append(f"{old_year} 年說明：{row_old['description']}")
    if row_new.get("description"):
        parts.append(f"{new_year} 年說明：{row_new['description']}")

    return "\n".join(parts)


def build_reply(question: str) -> str:
    """
    對外主入口：
    - 若使用者輸入格式為：#查 年度,單位,項目 -> 走精準 key 查詢
    - 若問題中出現「比較／差異／增減／變化……」等字眼，試著做年度比較
    - 否則走自然語言模糊比對（若單位或項目不清楚，就回固定道歉訊息）
    """
    text = question.strip()
    if not text:
        return "抱歉，我在訓練資料裡找不到這個問題的答案，可以換個說法或問別的問題喔。"

    # 1️⃣ 特殊指令：#查 年度,單位,項目
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

    # 2️⃣ 比較模式（113 年 vs 112 年…）
    if any(k in text for k in ["比較", "差異", "變化", "增減", "變動", "成長", "相比", "對比"]):
        compare_ans = _build_compare_answer(text)
        if compare_ans is not None:
            return compare_ans

    # 3️⃣ 一般使用者：走模糊查詢
    row = _find_best_row(text)
    if row is None:
        return "抱歉，我在訓練資料裡找不到這個問題的答案，可以換個說法或問別的問題喔。"

    return _format_row(row)
