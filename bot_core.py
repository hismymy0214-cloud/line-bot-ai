import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd

# =========================
# 基本設定
# =========================
DEFAULT_REPLY = "抱歉，該訓練檔找不到符合的資料，請重新輸入或換個說法喔。"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("TRAINING_FILE", "training.xlsx")
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)

COVERAGE_THRESHOLD = 0.8
SUGGEST_THRESHOLD = 0.6
SUGGEST_TOPN = 3

MAX_YEAR_SPAN = 10
MIN_QUERY_LEN = 8

RESULT_HEADER = "查詢結果如下："

_ANALYSIS_WORDS = ["比較", "變動", "異動", "差異", "增減"]

_YEAR_RE = re.compile(r"(\d{3})\s*年?")
_CHANGE_RE = re.compile(r"(較|比).*(上|前).*年|變動|增減|差額")

# =========================
# 工具函式
# =========================
def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[，,。．、\s]+", "", text.strip())


def _extract_year(text: str) -> Optional[int]:
    m = _YEAR_RE.search(text or "")
    return int(m.group(1)) if m else None


def _is_change_query(text: str) -> bool:
    return bool(_CHANGE_RE.search(text or ""))


def _split_desc_and_source(desc: str) -> Tuple[str, str]:
    if not desc:
        return "", ""
    lines = [l.strip() for l in desc.splitlines()]
    for i, line in enumerate(lines):
        if "資料來源" in line:
            head = "\n".join(lines[:i]).strip()
            if "：" in line:
                return head, line.split("：", 1)[1].strip()
            if i + 1 < len(lines):
                return head, lines[i + 1].strip()
            return head, ""
    return desc.strip(), ""


def _clean_source(src: str) -> str:
    if not src:
        return ""
    s = src.strip()
    if s.lower() == "nan":
        return ""
    if s in {"）", "(", "）", "("}:
        return ""
    return s


# =========================
# 資料結構
# =========================
@dataclass
class Entry:
    keyword: str
    keyword_norm: str
    year: Optional[int]
    description: str
    unit: str
    source_url: str


@dataclass
class ChangeEntry:
    keyword: str
    keyword_norm: str
    year: int
    value: int
    unit: str
    source_name: str


# =========================
# 載入資料
# =========================
_ENTRIES: List[Entry] = []
_EXACT: Dict[str, Entry] = {}

_CHANGE_ENTRIES: List[ChangeEntry] = []
_CHANGE_AVAILABLE = False


def _load_training() -> None:
    global _ENTRIES, _EXACT, _CHANGE_ENTRIES, _CHANGE_AVAILABLE

    if not os.path.exists(DATA_PATH):
        return

    # -------- sheet1 --------
    df = pd.read_excel(DATA_PATH, sheet_name=0, dtype=str).fillna("")
    entries: List[Entry] = {}
    exact = {}

    for _, r in df.iterrows():
        kw = r.get("keywords", "").strip()
        desc = r.get("description", "").strip()
        if not kw or not desc:
            continue

        year = _extract_year(kw)
        e = Entry(
            keyword=kw,
            keyword_norm=_normalize(kw),
            year=year,
            description=desc,
            unit=r.get("unit", "").strip(),
            source_url=r.get("source_url", "").strip(),
        )
        entries.setdefault(year, []).append(e)
        exact[_normalize(kw)] = e

    _ENTRIES = [e for v in entries.values() for e in v]
    _EXACT = exact

    # -------- 變動 --------
    try:
        cdf = pd.read_excel(DATA_PATH, sheet_name="變動", dtype=str).fillna("")
        for _, r in cdf.iterrows():
            kw = r.get("keywords", "").strip()
            val = r.get("value", "").strip()
            if not kw or not val:
                continue
            year = _extract_year(kw)
            if not year:
                continue
            _CHANGE_ENTRIES.append(
                ChangeEntry(
                    keyword=kw,
                    keyword_norm=_normalize(kw),
                    year=year,
                    value=int(val.replace(",", "")),
                    unit=r.get("unit", "").strip(),
                    source_name=r.get("source_url_name", "").strip(),
                )
            )
        _CHANGE_AVAILABLE = len(_CHANGE_ENTRIES) > 0
    except Exception:
        _CHANGE_AVAILABLE = False


_load_training()

# =========================
# 回覆格式
# =========================
def _format_answer(entry: Entry) -> str:
    desc = entry.description or ""
    head, src_desc = _split_desc_and_source(desc)

    src = _clean_source(src_desc)
    if not src:
        src = _clean_source(entry.source_url)

    head = head.strip()

    if head and src:
        return f"{head}\n\n（資料來源）\n{src}"
    if head:
        return head
    if src:
        return f"（資料來源）\n{src}"
    return ""


# =========================
# 主邏輯
# =========================
def _find_entry(year: int, topic: str) -> Optional[Entry]:
    topic_norm = _normalize(topic)
    for e in _ENTRIES:
        if e.year == year and topic_norm in e.keyword_norm:
            return e
    return None


def _format_change_reply(text: str) -> str:
    year = _extract_year(text)
    if not year:
        return "請輸入年度後再查詢。"

    topic = re.sub(_YEAR_RE, "", text)
    topic = re.sub(r"(較|比).*(上|前).*年|變動|增減|差額", "", topic)
    topic = topic.strip()

    cur = next((e for e in _CHANGE_ENTRIES if e.year == year and topic in e.keyword), None)
    prev = next((e for e in _CHANGE_ENTRIES if e.year == year - 1 and topic in e.keyword), None)

    if not cur or not prev:
        return DEFAULT_REPLY

    diff = cur.value - prev.value
    sign = "增加" if diff > 0 else "減少" if diff < 0 else "持平"
    pct = diff / prev.value * 100 if prev.value else 0

    lines = [
        f"{year}年{topic}總計{cur.value:,}{cur.unit}。",
        f"{year}年較{year-1}年{sign}{abs(diff):,}{cur.unit}（{pct:+.2f}%）。",
    ]

    src = _clean_source(cur.source_name)
    if src:
        lines.append("（資料來源）")
        lines.append(src)

    return "\n".join(lines)


def build_reply(user_text: str) -> str:
    text = (user_text or "").strip()
    if not text:
        return DEFAULT_REPLY

    # 較上一年度
    if _is_change_query(text):
        reply = _format_change_reply(text)
        return f"{RESULT_HEADER}\n{reply}"

    # 一般（sheet1）
    key = _normalize(text)
    if key in _EXACT:
        return f"{RESULT_HEADER}\n{_format_answer(_EXACT[key])}"

    year = _extract_year(text)
    if year:
        topic = re.sub(_YEAR_RE, "", text).strip()
        e = _find_entry(year, topic)
        if e:
            return f"{RESULT_HEADER}\n{_format_answer(e)}"

    return DEFAULT_REPLY
