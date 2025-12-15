import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd

# =========================
# 設定
# =========================
DEFAULT_REPLY = "抱歉，該訓練檔找不到符合的資料，請重新輸入或換個說法喔。"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("TRAINING_FILE", "training.xlsx")
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)

# 覆蓋率門檻：keywords 至少 80% 被使用者輸入「涵蓋」才算命中
COVERAGE_THRESHOLD = float(os.environ.get("COVERAGE_THRESHOLD", "0.8"))

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
    keyword: str
    keyword_norm: str
    year: Optional[str]
    description: str


_EXACT_MAP: Dict[str, str] = {}
_ENTRIES: List[Entry] = []


def _load_training() -> None:
    """讀取 training.xlsx，只使用：keywords / description。"""
    global _EXACT_MAP, _ENTRIES

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] training file not found: {DATA_PATH}")
        _EXACT_MAP = {}
        _ENTRIES = []
        return

    df = pd.read_excel(DATA_PATH, dtype=str).fillna("")
    cols = [c.strip().lower() for c in df.columns]
    colmap = {c.strip().lower(): c for c in df.columns}

    required = ["keywords", "description"]
    missing = [c for c in required if c not in cols]
    if missing:
        print(f"[ERROR] training file missing columns: {missing}. Found: {list(df.columns)}")
        _EXACT_MAP = {}
        _ENTRIES = []
        return

    kw_col = colmap["keywords"]
    desc_col = colmap["description"]

    exact_map: Dict[str, str] = {}
    entries: List[Entry] = []

    for _, r in df.iterrows():
        kw_raw = str(r.get(kw_col, "")).strip()
        desc = str(r.get(desc_col, "")).strip()
        if not kw_raw or not desc:
            continue

        kw_norm = _normalize(kw_raw)
        y = _extract_year(kw_raw)

        exact_map[kw_norm] = desc
        entries.append(Entry(
            keyword=kw_raw,
            keyword_norm=kw_norm,
            year=y,
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


def _coverage_ratio(keyword_norm: str, user_norm: str) -> float:
    """
    覆蓋率：keywords 裡的字元，有多少也出現在 user。
    用 Counter 可以處理重複字（例如「處處」這種情況）。
    """
    if not keyword_norm:
        return 0.0
    kw = Counter(keyword_norm)
    us = Counter(user_norm)
    hit = sum(min(cnt, us.get(ch, 0)) for ch, cnt in kw.items())
    return hit / max(1, len(keyword_norm))


def _match_by_keyword_coverage(user_text: str, threshold: float = COVERAGE_THRESHOLD) -> Optional[str]:
    """
    80% 覆蓋率比對：
    - 使用者輸入需涵蓋 keywords 至少 threshold（預設 0.8）
    - 若使用者有輸入年度，先用年度縮小候選
    - 同分時偏好較長的 keywords（更具體）
    """
    user_norm = _normalize(user_text)
    if not user_norm:
        return None

    user_year = _extract_year(user_text)

    candidates = _ENTRIES
    if user_year:
        candidates = [e for e in candidates if e.year == user_year]

    best: Optional[Tuple[float, int, Entry]] = None
    for e in candidates:
        r = _coverage_ratio(e.keyword_norm, user_norm)
        if r < threshold:
            continue

        tie = len(e.keyword_norm)  # 越長越具體
        if best is None or (r, tie) > (best[0], best[1]):
            best = (r, tie, e)

    return best[2].description if best else None


def build_reply(user_text: str) -> str:
    """
    需求：
    1) 使用者輸入與 keywords 完全符合 -> 回傳該列 description
    2) 否則比對 keywords 覆蓋率 >= 80% -> 回傳最符合的一列 description
    3) 都找不到 -> 回傳固定道歉訊息
    """
    text = (user_text or "").strip()
    if not text:
        return DEFAULT_REPLY

    ans = _match_by_exact(text)
    if ans:
        return ans

    ans = _match_by_keyword_coverage(text, threshold=COVERAGE_THRESHOLD)
    if ans:
        return ans

    return DEFAULT_REPLY
