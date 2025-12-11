import os
import re
from typing import Dict, Tuple, Optional

import pandas as pd
import difflib

# ------------------------------------------------------------
# 讀取訓練資料
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("TRAINING_FILE", "training.xlsx")
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)

_KNOWLEDGE: Optional[pd.DataFrame] = None


def _load_knowledge() -> pd.DataFrame:
    """讀取 training.xlsx 並快取在記憶體裡。"""
    global _KNOWLEDGE
    if _KNOWLEDGE is not None:
        return _KNOWLEDGE

    print(f"[DEBUG] bot_core loading training file: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)

    # 基本清理：轉成字串，避免 NaN
    for col in ["category", "year", "unit", "item", "description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    # 數值欄位
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    _KNOWLEDGE = df
    print(f"[DEBUG] Knowledge loaded. Rows={len(df)}")
    return df


# ------------------------------------------------------------
# 共用工具
# ------------------------------------------------------------

def _extract_years(text: str):
    """從問題字串抓出所有出現的年度（例如 113、112）。"""
    years = []
    for m in re.finditer(r"(\d{2,3})年", text):
        y = m.group(1)
        if y not in years:
            years.append(y)
    return years


def _fuzzy_match(query: str, choices) -> Tuple[Optional[str], float]:
    """
    使用 difflib 做最單純的模糊比對。
    回傳 (最佳文字, 分數 0–100)
    """
    if not choices:
        return None, 0.0

    best = None
    best_score = 0.0
    for c in choices:
        c_str = str(c)
        score = difflib.SequenceMatcher(None, query, c_str).ratio() * 100
        if score > best_score:
            best = c_str
            best_score = score
    return best, best_score


def _format_number(v) -> str:
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return str(v)


# ------------------------------------------------------------
# 以 year / unit / item 直接查詢
# ------------------------------------------------------------

def lookup_by_key(year: str, unit: str, item: str) -> Optional[Dict]:
    df = _load_knowledge()
    mask = (
        (df["year"].astype(str) == str(year).strip()) &
        (df["unit"].astype(str) == str(unit).strip()) &
        (df["item"].astype(str) == str(item).strip())
    )
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


# ------------------------------------------------------------
# 一般問題：單年度查詢
# ------------------------------------------------------------

_MIN_UNIT_SCORE = 60.0
_MIN_ITEM_SCORE = 60.0
_MIN_KEY_SCORE = 70.0  # year+unit+item 三者都要蠻接近時才回應（目前沒直接用到）

def _find_best_single_row(text: str) -> Optional[Dict]:
    """
    解析像「113年工務局職員總數」這種問題。
    只處理「單一年度」的查詢。
    """
    df = _load_knowledge()

    years = _extract_years(text)
    if not years:
        return None
    year = years[0]

    # 先過濾指定年度
    df_y = df[df["year"].astype(str) == year]
    if df_y.empty:
        return None

    # 找 unit
    unit_choices = df_y["unit"].unique().tolist()
    best_unit, unit_score = _fuzzy_match(text, unit_choices)
    if not best_unit or unit_score < _MIN_UNIT_SCORE:
        return None

    df_yu = df_y[df_y["unit"] == best_unit]

    # 找 item
    item_choices = df_yu["item"].unique().tolist()
    best_item, item_score = _fuzzy_match(text, item_choices)
    if not best_item or item_score < _MIN_ITEM_SCORE:
        return None

    df_final = df_yu[df_yu["item"] == best_item]
    if df_final.empty:
        return None

    row = df_final.iloc[0].to_dict()
    row["year"] = year
    return row


# ------------------------------------------------------------
# 年度比較：113 年與 112 年比較、或「113 年較上一年變動」
# ------------------------------------------------------------

def _build_year_comparison_answer(text: str) -> Optional[str]:
    """
    偵測「比較、變動、差異」相關的問題，回傳比較用文字。
    例如：
      - 113年工務局主管預算數跟112年比較
      - 113年工務局主管預算數較上一年變動
    """
    if not re.search(r"(比較|變動|差異)", text):
        return None

    df = _load_knowledge()

    years = _extract_years(text)
    if not years:
        return None

    # 取得要比較的兩個年度：
    # case1：題目中寫兩個年度 -> 用那兩個
    # case2：只有一個年度，且有「上一年／前一年／去年」等字樣 -> 用 (該年, 該年-1)
    if len(years) >= 2:
        y1, y2 = years[0], years[1]
    else:
        y1 = years[0]
        if re.search(r"(上一年|上年度|前一年|前年度|去年)", text):
            try:
                y2 = str(int(y1) - 1)
            except Exception:
                return None
        else:
            # 只有一個年度又看不出是跟哪一年比，就放棄
            return None

    # 只留下這兩年資料
    df_2y = df[df["year"].astype(str).isin([y1, y2])]
    if df_2y["year"].nunique() < 2:
        return None

    # 找 unit
    unit_choices = df_2y["unit"].unique().tolist()
    best_unit, unit_score = _fuzzy_match(text, unit_choices)
    if not best_unit or unit_score < _MIN_UNIT_SCORE:
        return None

    df_u = df_2y[df_2y["unit"] == best_unit]

    # 找 item
    item_choices = df_u["item"].unique().tolist()
    best_item, item_score = _fuzzy_match(text, item_choices)
    if not best_item or item_score < _MIN_ITEM_SCORE:
        return None

    df_ui = df_u[df_u["item"] == best_item]
    if df_ui["year"].nunique() < 2:
        return None

    # new / old 年：以數字大小判斷
    y1_int, y2_int = int(y1), int(y2)
    new_year = str(max(y1_int, y2_int))
    old_year = str(min(y1_int, y2_int))

    row_new = df_ui[df_ui["year"].astype(str) == new_year].iloc[0].to_dict()
    row_old = df_ui[df_ui["year"].astype(str) == old_year].iloc[0].to_dict()

    v_new = row_new.get("value")
    v_old = row_old.get("value")
    if v_new is None or v_old is None:
        return None

    try:
        diff = float(v_new) - float(v_old)
        rate = (diff / float(v_old)) * 100 if float(v_old) != 0 else 0.0
    except Exception:
        return None

    category = row_new.get("category", "")
    unit = best_unit
    item = best_item

    desc_old = str(row_old.get("description", "")).strip()
    desc_new = str(row_new.get("description", "")).strip()

    lines = []
    if category:
        lines.append(f"【類別】{category}")
    lines.append(f"【比較項目】{unit}/{item}")
    lines.append(f"【{old_year} 年數值】{_format_number(v_old)}")
    lines.append(f"【{new_year} 年數值】{_format_number(v_new)}")
    lines.append(f"【差額】{_format_number(diff)}")
    lines.append(f"【成長率】{rate:.2f}%（以 {old_year} 年為基準）")

    if desc_old:
        lines.append("")
        lines.append(f"{old_year} 年說明：{desc_old}")
    if desc_new:
        lines.append(f"{new_year} 年說明：{desc_new}")

    return "\n".join(lines)


# ------------------------------------------------------------
# 回覆格式（單一年度）
# ------------------------------------------------------------

def _format_row(row: Dict) -> str:
    parts = []
    if row.get("category"):
        parts.append(f"【類別】{row['category']}")
    if row.get("year"):
        parts.append(f"【年度】{row['year']} 年")
    if row.get("unit"):
        parts.append(f"【單位】{row['unit']}")
    if row.get("item"):
        parts.append(f"【項目】{row['item']}")
    if row.get("value") is not None:
        parts.append(f"【數值】{_format_number(row['value'])}")
    if row.get("description"):
        parts.append(str(row["description"]))
    return "\n".join(parts)


# ------------------------------------------------------------
# 對外主入口：build_reply
# ------------------------------------------------------------

def build_reply(question: str) -> str:
    """依照使用者提問，自動決定要回單年查詢或年度比較。"""
    text = (question or "").strip()
    if not text:
        return "抱歉，我不太確定你的問題，可以再補充一下嗎？"

    # --------------------------------------------------------
    # 1) 管理者查詢模式：#113,工務局,職員總數
    # --------------------------------------------------------
    if text.startswith("#"):
        payload = text[1:].strip()  # 去掉 #
        parts = [p.strip() for p in re.split("[,，]", payload) if p.strip()]
        if len(parts) != 3:
            return (
                "【查詢格式錯誤】\n"
                "請使用：#年度,單位,項目\n"
                "例如：#113,工務局,職員總數"
            )
        year, unit, item = parts
        row = lookup_by_key(year, unit, item)
        if row is None:
            return "抱歉，查無對應資料，請確認年度、單位及項目是否正確。"
        return _format_row(row)

    # --------------------------------------------------------
    # 2) 年度比較問題：先嘗試建構比較答案
    # --------------------------------------------------------
    cmp_answer = _build_year_comparison_answer(text)
    if cmp_answer:
        return cmp_answer

    # --------------------------------------------------------
    # 3) 一般單年度查詢
    # --------------------------------------------------------
    row = _find_best_single_row(text)
    if row is not None:
        return _format_row(row)

    # --------------------------------------------------------
    # 4) 以上都失敗，就回預設句
    # --------------------------------------------------------
    return "抱歉，我在訓練資料裡找不到這個問題的答案，可以換個說法或問別的問題喔。"
