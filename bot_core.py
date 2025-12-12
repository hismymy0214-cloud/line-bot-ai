import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =========================
# 設定
# =========================
DEFAULT_REPLY = "抱歉，該訓練檔找不到符合的資料，請重新輸入或換個說法喔。"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("TRAINING_FILE", "training.xlsx")
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)

# 額外：常見同義/寫法修正（可再擴充）
_REPLACEMENTS = [
    ("年度", "年"),
    ("年 度", "年"),
    ("　", ""),  # 全形空白
]

_PUNCT_RE = re.compile(r"[，,。．、\s]+")


def _normalize(text: str) -> str:
    """把輸入字串做標準化，提升命中率（不做斷詞，只做輕量規範）。"""
    if text is None:
        return ""
    t = str(text).strip()
    for a, b in _REPLACEMENTS:
        t = t.replace(a, b)
    t = _PUNCT_RE.sub("", t)
    return t


def _extract_year(text: str) -> Optional[str]:
    """
    從文字抓年度（3位數：111/112/113...），回傳 '113'。
    若未出現則回傳 None。
    """
    m = re.search(r"(?P<y>\d{3})\s*年", text)
    if m:
        return m.group("y")
    # 有些人只打 113 也可能想查
    m2 = re.search(r"(?P<y>\d{3})", text)
    return m2.group("y") if m2 else None


@dataclass(frozen=True)
class Entry:
    keyword: str                 # 主 key（唯一）
    keyword_norm: str            # 正規化後的主 key
    year: Optional[str]          # 由主 key 推導的年度
    range_terms: List[str]       # 次 key（關鍵詞列表）
    description: str             # 要回覆的內容


_EXACT_MAP: Dict[str, str] = {}
_ENTRIES: List[Entry] = []


def _load_training() -> None:
    """讀取 training.xlsx，只使用：keywords / keywords_range / description。"""
    global _EXACT_MAP, _ENTRIES

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] training file not found: {DATA_PATH}")
        _EXACT_MAP = {}
        _ENTRIES = []
        return

    df = pd.read_excel(DATA_PATH, dtype=str).fillna("")
    cols = [c.strip().lower() for c in df.columns]
    colmap = {c.strip().lower(): c for c in df.columns}

    required = ["keywords", "keywords_range", "description"]
    missing = [c for c in required if c not in cols]
    if missing:
        print(f"[ERROR] training file missing columns: {missing}. Found: {list(df.columns)}")
        _EXACT_MAP = {}
        _ENTRIES = []
        return

    kw_col = colmap["keywords"]
    kr_col = colmap["keywords_range"]
    desc_col = colmap["description"]

    exact_map: Dict[str, str] = {}
    entries: List[Entry] = []

    for _, r in df.iterrows():
        kw_raw = str(r.get(kw_col, "")).strip()
        desc = str(r.get(desc_col, "")).strip()
        kr_raw = str(r.get(kr_col, "")).strip()

        if not kw_raw or not desc:
            continue

        kw_norm = _normalize(kw_raw)
        # keywords_range：用逗號分隔（支援中英文逗號）
        terms = []
        if kr_raw:
            parts = re.split(r"[，,]", kr_raw)
            terms = [p.strip() for p in parts if p.strip()]

        # 把 terms 也做 normalize，並移除過短詞（例如 1 字太容易誤判）
        norm_terms = []
        for t in terms:
            nt = _normalize(t)
            if len(nt) >= 2:
                norm_terms.append(nt)

        # 年度從 keywords 抓：113年xxxx => 113
        y = _extract_year(kw_raw)

        exact_map[kw_norm] = desc
        entries.append(Entry(
            keyword=kw_raw,
            keyword_norm=kw_norm,
            year=y,
            range_terms=norm_terms,
            description=desc
        ))

    _EXACT_MAP = exact_map
    _ENTRIES = entries
    print(f"[DEBUG] training loaded: {DATA_PATH}, entries={len(_ENTRIES)}")


_load_training()


def _match_by_exact(user_text: str) -> Optional[str]:
    """keywords 完全符合（含基本正規化後）就直接回傳 description。"""
    key = _normalize(user_text)
    if not key:
        return None
    return _EXACT_MAP.get(key)


def _score_range_match(entry: Entry, user_norm: str) -> int:
    """對 keywords_range 進行粗略打分：命中詞越多、詞越長分數越高。"""
    score = 0
    for t in entry.range_terms:
        if t and t in user_norm:
            score += len(t)
    return score


def _match_by_range(user_text: str) -> Optional[str]:
    """
    次 key：keywords_range。
    - 先用「年度」縮小候選（使用者有打年度才做過濾）
    - 再依命中詞長度加總打分
    - 避免太寬鬆：至少要命中一個「非年度」的長詞（>=3）
    """
    user_norm = _normalize(user_text)
    if not user_norm:
        return None

    user_year = _extract_year(user_text)

    candidates = _ENTRIES
    if user_year:
        candidates = [e for e in candidates if e.year == user_year]

    best: Optional[Tuple[int, int, Entry]] = None
    for e in candidates:
        if not e.range_terms:
            continue

        score = _score_range_match(e, user_norm)
        if score <= 0:
            continue

        # 強制：至少命中一個 >=3 的詞（避免只打「113年道路養護工程處」就誤帶出各種職員類）
        hit_long = any((t in user_norm) and (len(t) >= 3) for t in e.range_terms)
        if not hit_long:
            continue

        # tie-break：同分時偏好 keyword 較長（較具體）
        tie = len(e.keyword_norm)
        if best is None or (score, tie) > (best[0], best[1]):
            best = (score, tie, e)

    return best[2].description if best else None


def build_reply(user_text: str) -> str:
    """
    需求：
    1) 使用者輸入與 keywords 完全符合 -> 回傳該列 description
    2) 否則比對 keywords_range -> 回傳最符合的一列 description
    3) 都找不到 -> 回傳固定道歉訊息
    """
    text = (user_text or "").strip()
    if not text:
        return DEFAULT_REPLY

    ans = _match_by_exact(text)
    if ans:
        return ans

    ans = _match_by_range(text)
    if ans:
        return ans

    return DEFAULT_REPLY
