# bot_core.py
import os
import re
import difflib
from typing import Optional, Tuple, Dict

import pandas as pd


# === 訓練檔設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("TRAINING_FILE", "training.xlsx")
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)

# 模糊比對門檻（數值越高越嚴格）
MIN_UNIT_SCORE = 30.0   # 單位模糊比對最低分數（單年度查詢用）
MIN_ITEM_SCORE = 30.0   # 項目模糊比對最低分數（單年度查詢用）
MIN_COMPARE_SEARCH_SCORE = 40.0  # 跨年度比較時，search_text 最低分數


def _load_knowledge() -> pd.DataFrame:
    """
    讀取訓練檔並做前置整理：
    - 去除前後空白
    - year / value 轉數值
    - 建立 series_key（跨年度比較用）
    - 建立 search_text（之後如果要換嵌入模型也方便）
    """
    try:
        df = pd.read_excel(DATA_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            "category", "year", "unit", "item",
            "value", "description", "keywords",
            "series_key", "search_text",
        ])

    # 基本欄位補齊
    for col in ["category", "year", "unit", "item", "value", "description", "keywords"]:
        if col not in df.columns:
            df[col] = ""

    # 轉字串、去空白
    for col in ["category", "unit", "item", "description", "keywords"]:
        df[col] = df[col].astype(str).str.strip()

    # 年度與數值
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # series_key：同一指標不同年度，共用同一 key（跨年度比較用）
    if "series_key" not in df.columns:
        df["series_key"] = df["category"] + "|" + df["unit"] + "|" + df["item"]
    else:
        df["series_key"] = df["series_key"].astype(str).str.strip()

    # search_text：模糊比對使用
    if "search_text" not in df.columns:
        def make_search_text(row):
            parts = []
            cat = row.get("category")
            if pd.notna(cat):
                parts.append(str(cat))

            year = row.get("year")
            if pd.notna(year):
                y = int(year)
                parts.append(f"{y}年")
                parts.append(f"{y}年度")

            unit = row.get("unit")
            if pd.notna(unit):
                parts.append(str(unit))

            item = row.get("item")
            if pd.notna(item):
                parts.append(str(item))

            kw = row.get("keywords")
            if pd.notna(kw) and str(kw).strip():
                parts.append(str(kw))

            return " ".join(parts)

        df["search_text"] = df.apply(make_search_text, axis=1)
    else:
        df["search_text"] = df["search_text"].astype(str)

    df = df.fillna("")
    return df


_KNOWLEDGE = _load_knowledge()


# === 工具函式 ===

def _extract_years(text: str):
    """從問題中抓出所有『XXX年』的數字（例如 113、112）。"""
    return [int(m) for m in re.findall(r"(\d{2,3})年", text)]


def _similarity(a: str, b: str) -> float:
    """字串相似度（0~100）。"""
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def _fuzzy_best_match(query: str, choices) -> Tuple[Optional[str], float]:
    """在 choices 中找出與 query 最相近的那一個。"""
    best = None
    best_score = 0.0
    for c in choices:
        c_str = str(c)
        score = _similarity(query, c_str)
        if score > best_score:
            best_score = score
            best = c_str
    return best, best_score


def _is_compare_question(text: str) -> bool:
    """判斷是否為『跨年度比較／變動』的問題。"""
    keywords = [
        "比較", "差異", "變動", "增減", "成長率", "成長",
        "較上一年", "較上一年度", "較前一年", "較去年",
    ]
    return any(k in text for k in keywords)


def _parse_compare_years(text: str) -> Optional[Tuple[int, int]]:
    """
    解析比較問題中的新舊年度：
    回傳 (new_year, old_year)，例如 (113, 112)
    """
    years = _extract_years(text)
    if len(years) >= 2:
        # 問句同時出現兩個年份，例如「113年跟112年比較」
        years_sorted = sorted(years)
        old_year = years_sorted[0]
        new_year = years_sorted[-1]
        return new_year, old_year

    if len(years) == 1:
        year = years[0]
        # 有提到『上一年／上一年度／前一年／去年』之類，就當成 year 跟 year-1 比較
        if any(w in text for w in ["較上一年", "較上一年度", "上一年度", "上一年", "前一年", "去年"]):
            return year, year - 1

    return None


def _find_row_single_year(question: str) -> Optional[Dict]:
    """處理單一年度查詢：例如『113年工務局職員人數』。"""
    if _KNOWLEDGE.empty:
        return None

    years = _extract_years(question)
    if not years:
        return None

    year = max(years)  # 若有多個，選最新的那個
    df_year = _KNOWLEDGE[_KNOWLEDGE["year"] == year]
    if df_year.empty:
        return None

    # 先模糊比對 unit
    unit_choices = df_year["unit"].unique()
    best_unit, unit_score = _fuzzy_best_match(question, unit_choices)

    # 再模糊比對 item
    item_choices = df_year["item"].unique()
    best_item, item_score = _fuzzy_best_match(question, item_choices)

    # 分數太低就當作找不到，直接走預設回覆
    if unit_score < MIN_UNIT_SCORE or item_score < MIN_ITEM_SCORE:
        return None

    df_match = df_year[(df_year["unit"] == best_unit) & (df_year["item"] == best_item)]
    if df_match.empty:
        return None

    # 正常情況只會有一筆；多筆就取第一筆
    row = df_match.iloc[0].to_dict()
    return row


def _build_single_answer(row: Dict) -> str:
    """把單筆資料整理成回覆文字。"""
    parts = []
    if row.get("category"):
        parts.append(f"【類別】{row['category']}")
    if row.get("year") not in ("", None, pd.NA):
        parts.append(f"【年度】{row['year']} 年")
    if row.get("unit"):
        parts.append(f"【單位】{row['unit']}")
    if row.get("item"):
        parts.append(f"【項目】{row['item']}")
    if row.get("value") not in ("", None, pd.NA):
        parts.append(f"【數值】{row['value']}")
    if row.get("description"):
        parts.append(str(row["description"]))
    return "\n".join(parts)


def _build_compare_answer(question: str) -> Optional[str]:
    """
    處理『跨年度比較／較上一年度變動』的問題。
    例如：
    - 113年工務局主管預算數較上一年度變動
    - 113年工務局主管預算數跟112年比較
    """
    if _KNOWLEDGE.empty:
        return None

    year_pair = _parse_compare_years(question)
    if not year_pair:
        return None

    new_year, old_year = year_pair

    df_new = _KNOWLEDGE[_KNOWLEDGE["year"] == new_year]
    if df_new.empty:
        return None

    # 把比較相關字眼拿掉，避免干擾比對
    cleaned_q = question
    for token in [
        "較上一年", "較上一年度", "上一年度", "上一年", "前一年", "去年",
        "比較", "變動", "差異", "增減", "成長率", "成長",
    ]:
        cleaned_q = cleaned_q.replace(token, "")

    # 用 search_text 做一輪模糊比對，找出最接近的那一筆指標
    best_idx = None
    best_score = 0.0
    for idx, row in df_new.iterrows():
        score = _similarity(cleaned_q, str(row.get("search_text", "")))
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None or best_score < MIN_COMPARE_SEARCH_SCORE:
        # 分數太低，代表問句太模糊或跟資料無關
        return None

    row_new = df_new.loc[best_idx]

    # 用 series_key + 年度 找上一年度
    series_key = row_new.get("series_key")
    df_old_match = _KNOWLEDGE[
        (_KNOWLEDGE["year"] == old_year) &
        (_KNOWLEDGE["series_key"] == series_key)
    ]

    if df_old_match.empty:
        # 找不到上一年度的對應資料，就當作無法比較
        return None

    row_old = df_old_match.iloc[0]

    v_new = row_new.get("value")
    v_old = row_old.get("value")

    if pd.isna(v_new) or pd.isna(v_old):
        return None

    diff = float(v_new) - float(v_old)
    if float(v_old) != 0:
        growth_rate = diff / float(v_old) * 100.0
        growth_text = f"{growth_rate:.2f}%"
    else:
        growth_text = "（上一年度為 0，無法計算成長率）"

    # 組合回覆
    unit = row_new.get("unit", "")
    item = row_new.get("item", "")
    cat = row_new.get("category", "")

    lines = []
    title = f"{cat}－{unit}{item}（{new_year} 年 vs. {old_year} 年）"
    lines.append(title)

    lines.append(f"【{old_year} 年數值】{v_old}")
    lines.append(f"【{new_year} 年數值】{v_new}")
    lines.append(f"【差額】{diff}")
    lines.append(f"【成長率】{growth_text}")

    # 如有說明文字，可附上最新年度的描述
    desc_new = str(row_new.get("description") or "").strip()
    if desc_new:
        lines.append("")
        lines.append("【最新年度說明】")
        lines.append(desc_new)

    return "\n".join(lines)


def build_reply(question: str) -> str:
    """
    LINE Bot 對外使用的主函式。
    傳入使用者文字，回傳要顯示的回覆內容。
    """
    q = (question or "").strip()

    if not q:
        return "可以輸入想查詢的年度、單位與項目，例如：113年工務局職員人數。"

    # 後門查詢：#year,unit,item  例如：#113,工務局,職員人數
    if q.startswith("#"):
        try:
            _, payload = q.split("#", 1)
            parts = [p.strip() for p in payload.split(",") if p.strip()]
            if len(parts) >= 3:
                year_str, unit, item = parts[0], parts[1], parts[2]
                year = int(re.sub(r"[^0-9]", "", year_str))
                df_yr = _KNOWLEDGE[_KNOWLEDGE["year"] == year]
                df_match = df_yr[(df_yr["unit"] == unit) & (df_yr["item"] == item)]
                if not df_match.empty:
                    row = df_match.iloc[0].to_dict()
                    return _build_single_answer(row)
                else:
                    return "找不到符合 year/unit/item 的資料，請確認輸入是否正確。"
        except Exception:
            # 如果格式怪怪的，就當作一般問題處理
            pass

    # 先判斷是不是「跨年度比較／變動」問題
    if _is_compare_question(q):
        compare_answer = _build_compare_answer(q)
        if compare_answer:
            return compare_answer

    # 一般單年度查詢
    row = _find_row_single_year(q)
    if row is None:
        return "抱歉，我在訓練資料裡找不到這個問題的答案，可以換個說法或問別的問題喔。"

    return _build_single_answer(row)
