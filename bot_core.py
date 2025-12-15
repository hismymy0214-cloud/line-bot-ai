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

# 候選建議區間：60%～80% 會列出最接近的 keywords
SUGGEST_THRESHOLD = float(os.environ.get("SUGGEST_THRESHOLD", "0.6"))
SUGGEST_TOPN = int(os.environ.get("SUGGEST_TOPN", "3"))

# 輸入太短時（像「113年工務局主管」）先引導
MIN_QUERY_LEN = int(os.environ.get("MIN_QUERY_LEN", "8"))

# 額外：常見同義/寫法修正（可再擴充）
_REPLACEMENTS = [
    ("年度", "年"),
    ("年 度", "年"),
    ("　", ""),  # 全形空白
]

_PUNCT_RE = re.compile(r"[，,。．、\s]+")
_YEAR_RE = re.compile(r"(?P<y>\d{3})\s*年")


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
    m = _YEAR_RE.search(text)
    if m:
        return m.group("y")
    # 有些人只打 113 也可能想查
    m2 = re.search(r"(?P<y>\d{3})", text)
    return m2.group("y") if m2 else None


def _strip_year(text_norm: str) -> str:
    """
    移除正規化後的年度標記（例如 113年 / 113），用於「未輸入年度」時的輔助比對。
    """
    # 先移除 113年 形式
    t = _YEAR_RE.sub("", text_norm)
    # 再移除單獨的三位數（避免殘留）
    t = re.sub(r"\d{3}", "", t)
    return t


@dataclass(frozen=True)
class Entry:
    keyword: str
    keyword_norm: str
    keyword_norm_noyear: str
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
        kw_norm_noyear = _strip_year(kw_norm)

        exact_map[kw_norm] = desc
        entries.append(Entry(
            keyword=kw_raw,
            keyword_norm=kw_norm,
            keyword_norm_noyear=kw_norm_noyear,
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
    用 Counter 可以處理重複字。
    """
    if not keyword_norm:
        return 0.0
    kw = Counter(keyword_norm)
    us = Counter(user_norm)
    hit = sum(min(cnt, us.get(ch, 0)) for ch, cnt in kw.items())
    return hit / max(1, len(keyword_norm))


def _rank_matches(user_text: str, use_year_filter: bool = True) -> List[Tuple[float, int, Entry]]:
    """
    回傳依覆蓋率排序的候選清單 [(ratio, tie_len, entry), ...]
    """
    user_norm = _normalize(user_text)
    if not user_norm:
        return []

    user_year = _extract_year(user_text) if use_year_filter else None

    candidates = _ENTRIES
    if user_year and use_year_filter:
        candidates = [e for e in candidates if e.year == user_year]

    ranked: List[Tuple[float, int, Entry]] = []
    for e in candidates:
        r = _coverage_ratio(e.keyword_norm, user_norm)
        tie = len(e.keyword_norm)  # 同分偏好較長（更具體）
        ranked.append((r, tie, e))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked


def _rank_matches_noyear(user_text: str) -> List[Tuple[float, int, Entry]]:
    """
    未輸入年度時：用「去年度」版本做輔助比對。
    """
    user_norm = _normalize(user_text)
    if not user_norm:
        return []

    user_norm_noyear = _strip_year(user_norm)
    if not user_norm_noyear:
        return []

    ranked: List[Tuple[float, int, Entry]] = []
    for e in _ENTRIES:
        # 以「去年度」字串算覆蓋率
        r = _coverage_ratio(e.keyword_norm_noyear, user_norm_noyear)
        tie = len(e.keyword_norm_noyear)
        ranked.append((r, tie, e))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked


def _format_suggestions(items: List[Tuple[float, int, Entry]], topn: int) -> str:
    picks = items[:topn]
    lines = []
    for r, _, e in picks:
        pct = int(round(r * 100))
        lines.append(f"- {e.keyword}（相符約 {pct}%）")
    return "\n".join(lines)


def build_reply(user_text: str) -> str:
    """
    A) 輸入太短 -> 引導（給範例）
    B) 60%~80% -> 列出最接近 keywords 清單供複製
    C) 未輸入年度但很像某些題 -> 先詢問年度並列候選
    """
    text = (user_text or "").strip()
    if not text:
        return DEFAULT_REPLY

    user_norm = _normalize(text)
    user_year = _extract_year(text)

    # A) 太短先引導（避免亂猜）
    if len(user_norm) < MIN_QUERY_LEN:
        return (
            "請輸入更完整的查詢關鍵詞（含年度/單位/指標），例如：\n"
            "- 113年工務局主管預算數\n"
            "- 113年工務局主管經常門\n"
            "- 113年工務局暨所屬職員人數"
        )

    # 1) 完全符合
    ans = _match_by_exact(text)
    if ans:
        return ans

    # 2) 年度一致下的覆蓋率比對
    ranked = _rank_matches(text, use_year_filter=True)
    if ranked:
        best_r, _, best_e = ranked[0]
        if best_r >= COVERAGE_THRESHOLD:
            return best_e.description

    # C) 沒輸入年度：用去年度比對判斷是否應該先問年度
    if not user_year:
        ranked_noyear = _rank_matches_noyear(text)
        if ranked_noyear:
            best_r2, _, _ = ranked_noyear[0]
            # 如果「去年度」比對已經很高，代表多半只是漏打年度
            if best_r2 >= COVERAGE_THRESHOLD:
                # 列出不同年度的候選（最多 3）
                suggestions = _format_suggestions(ranked_noyear, SUGGEST_TOPN)
                return (
                    "我看起來你可能少打了『年度』。\n"
                    "請補上年度（例如：113年），或直接複製下列其中一筆再送出：\n"
                    f"{suggestions}"
                )

    # B) 60%~80%：給候選清單（帶年度過濾後）
    if ranked:
        best_r, _, _ = ranked[0]
        if best_r >= SUGGEST_THRESHOLD:
            suggestions = _format_suggestions(ranked, SUGGEST_TOPN)
            return (
                "我找到了接近的查詢項目，但關鍵詞還不夠完整。\n"
                "請直接複製下列其中一筆 keywords 再送出：\n"
                f"{suggestions}"
            )

    return DEFAULT_REPLY
