import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

# 多年度（區間）最多支援 5 年（含）
MAX_MULTIYEAR = 5

MIN_QUERY_LEN = 8
RESULT_HEADER = "查詢結果如下："

_YEAR_RE = re.compile(r"(\d{3})\s*年?")

# 多年度區間：109-113 / 109~113 / 109至113 / 109到113
_YEAR_RANGE_RE = re.compile(r"(?P<y1>\d{3})\s*(?:[-~－—]|至|到)\s*(?P<y2>\d{3})\s*年?")

# =========================
# 「較上一年度變動」觸發規則（可擴充）
# =========================
CHANGE_TIME_KEYWORDS = [
    "較上一年度",
    "較上年度",
    "比上一年度",
    "比上年度",
    "前一年度",
    "前年度",
    "去年",
    "上年度",
    "上一年度",
]

CHANGE_ACTION_KEYWORDS = [
    "變動",
    "異動",
    "增減",
    "差額",
    "差距",
    "成長",
    "增加",
    "減少",
    "上升",
    "下降",
]

ALLOW_ACTION_ONLY_CHANGE = False


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


def _extract_year_range(text: str) -> Optional[List[int]]:
    """
    解析多年度區間（最多 MAX_MULTIYEAR 年）：
      109-113年 / 109~113年 / 109至113年 / 109到113年
    回傳升冪年份清單，例如 [109,110,111,112,113]
    不是區間則回 None
    """
    m = _YEAR_RANGE_RE.search(text or "")
    if not m:
        return None

    y1 = int(m.group("y1"))
    y2 = int(m.group("y2"))
    lo, hi = (y1, y2) if y1 <= y2 else (y2, y1)

    years = list(range(lo, hi + 1))
    if len(years) > MAX_MULTIYEAR:
        return ["__TOO_LONG__"]  # type: ignore
    return years


def _strip_year_range(text: str) -> str:
    """移除區間年度表達，留下主題。"""
    return _YEAR_RANGE_RE.sub("", text or "").strip()


def _is_change_query(text: str) -> bool:
    """
    判斷是否為「較上一年度變動」類查詢。
    規則（預設較安全）：
      - 同時包含「時間比較」關鍵字 + 「變動行為」關鍵字
    """
    if not text:
        return False

    t = str(text)
    has_time = any(k in t for k in CHANGE_TIME_KEYWORDS)
    has_action = any(k in t for k in CHANGE_ACTION_KEYWORDS)

    if has_time and has_action:
        return True

    if ALLOW_ACTION_ONLY_CHANGE and has_action:
        return True

    return False


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
    if s in {"）", "(", "（", ")"}:
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
    entries: Dict[Optional[int], List[Entry]] = {}
    exact: Dict[str, Entry] = {}

    for _, r in df.iterrows():
        kw = str(r.get("keywords", "")).strip()
        desc = str(r.get("description", "")).strip()
        if not kw or not desc:
            continue

        year = _extract_year(kw)
        e = Entry(
            keyword=kw,
            keyword_norm=_normalize(kw),
            year=year,
            description=desc,
            unit=str(r.get("unit", "")).strip(),
            source_url=str(r.get("source_url", "")).strip(),
        )
        entries.setdefault(year, []).append(e)
        exact[_normalize(kw)] = e

    _ENTRIES = [e for v in entries.values() for e in v]
    _EXACT = exact

    # -------- 變動 --------
    _CHANGE_ENTRIES = []
    try:
        cdf = pd.read_excel(DATA_PATH, sheet_name="變動", dtype=str).fillna("")
        for _, r in cdf.iterrows():
            kw = str(r.get("keywords", "")).strip()
            val = str(r.get("value", "")).strip()
            if not kw or not val:
                continue

            year = _extract_year(kw)
            if not year:
                continue

            # 允許 value 含逗號或其他符號
            v = re.sub(r"[^\d\-]", "", val)
            if v == "" or v == "-":
                continue

            _CHANGE_ENTRIES.append(
                ChangeEntry(
                    keyword=kw,
                    keyword_norm=_normalize(kw),
                    year=year,
                    value=int(v),
                    unit=str(r.get("unit", "")).strip(),
                    source_name=str(r.get("source_url_name", "")).strip(),
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
        return f"{head}\n\n資料來源：{src}"
    if head:
        return head
    if src:
        return f"資料來源：{src}"
    return ""


def _format_answer_body_only(entry: Entry) -> Tuple[str, str]:
    """
    回傳：(body_only, source)
    - body_only：不含「資料來源」的主內容
    - source：清理後的資料來源文字（可能為空）
    """
    desc = entry.description or ""
    head, src_desc = _split_desc_and_source(desc)
    src = _clean_source(src_desc) or _clean_source(entry.source_url)
    return head.strip(), src


# =========================
# 主邏輯
# =========================
def _find_entry(year: int, topic: str) -> Optional[Entry]:
    topic_norm = _normalize(topic)
    for e in _ENTRIES:
        if e.year == year and topic_norm in e.keyword_norm:
            return e
    return None


def _format_multiyear_reply(text: str) -> str:
    years = _extract_year_range(text)
    if not years:
        return ""

    if years == ["__TOO_LONG__"]:
        return f"多年度查詢目前最多支援 {MAX_MULTIYEAR} 年，請縮小查詢範圍（例如：109-113年）。"

    topic = _strip_year_range(text)
    topic = re.sub(r"[？\?！!。．，,]+", "", topic).strip()
    if not topic:
        return "請在年度區間後補充查詢主題，例如：109-113年工務局所屬職員人數。"

    lines: List[str] = []
    picked_source = ""

    for y in years:
        e = _find_entry(y, topic)
        if not e:
            lines.append(f"{y}年{topic}：查無資料")
            continue

        body, src = _format_answer_body_only(e)
        m = re.search(rf"{y}年.*?總計[\d,]+[^，。,]*", body)
        if m:
            lines.append(m.group(0) + "。")
        else:
            lines.append(f"{y}年{topic}：查無資料")
        if not picked_source and src:
            picked_source = src

    if picked_source:
        lines.append(f"\n資料來源：{picked_source}")

    return "\n".join(lines)


def _format_change_reply(text: str) -> str:
    """
    以「變動」工作表計算：year vs year-1。
    同時列出：
      - 當年度總計
      - 前一年度總計
      - 差額與百分比
    """
    year = _extract_year(text)
    if not year:
        return "請輸入年度後再查詢。"

    topic = re.sub(_YEAR_RE, "", text)
    for w in CHANGE_TIME_KEYWORDS + CHANGE_ACTION_KEYWORDS:
        topic = topic.replace(w, "")
    topic = topic.strip()

    cur = next((e for e in _CHANGE_ENTRIES if e.year == year and topic and topic in e.keyword), None)
    prev = next((e for e in _CHANGE_ENTRIES if e.year == year - 1 and topic and topic in e.keyword), None)

    if not cur or not prev:
        return DEFAULT_REPLY

    diff = cur.value - prev.value
    sign = "增加" if diff > 0 else "減少" if diff < 0 else "持平"
    pct = diff / prev.value * 100 if prev.value else 0

    lines = [
        f"{year}年{topic}總計{cur.value:,}{cur.unit}。",
        f"{year-1}年{topic}總計{prev.value:,}{cur.unit}。",
        f"{year}年較{year-1}年{sign}{abs(diff):,}{cur.unit}（{pct:+.2f}%）。",
    ]

    src = _clean_source(cur.source_name)
    if src:
        lines.append(src)

    return "\n".join(lines)


def build_reply(user_text: str) -> str:
    text = (user_text or "").strip()
    if not text:
        return DEFAULT_REPLY

    # 1) 較上一年度（走「變動」工作表）
    if _is_change_query(text):
        reply = _format_change_reply(text)
        return f"{RESULT_HEADER}\n{reply}"

    # 2) 多年度區間（最多 5 年；逐年列示）
    multi = _format_multiyear_reply(text)
    if multi:
        return f"{RESULT_HEADER}\n{multi}"

    # 3) 一般（sheet1）
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
