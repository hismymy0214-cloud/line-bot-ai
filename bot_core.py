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

# 接近門檻：>= 0.6 且 < 0.8 時，回候選提示
SUGGEST_THRESHOLD = float(os.environ.get("SUGGEST_THRESHOLD", "0.6"))
SUGGEST_TOPN = int(os.environ.get("SUGGEST_TOPN", "3"))

# 多年度最多允許查詢多少年（避免有人輸入 80-120 年把 Bot 打爆）
MAX_YEAR_SPAN = int(os.environ.get("MAX_YEAR_SPAN", "10"))

# 輸入太短時先引導
MIN_QUERY_LEN = int(os.environ.get("MIN_QUERY_LEN", "8"))

# 年度差異摘要「開關」關鍵字：只有出現這些字才顯示摘要
ANALYSIS_KEYWORDS = ["比較", "變化", "異動", "差異", "增減", "趨勢"]

# ===== 查詢結果標頭 =====
RESULT_HEADER = "查詢結果如下："

# ===== 滿意度調查（查到/查不到 分流）=====
SURVEY_URL = os.environ.get("SURVEY_URL", "")  # 留空=不在回覆中顯示（建議改用圖文選單）

SURVEY_FOOTER_SUCCESS = ""

SURVEY_FOOTER_FALLBACK = ""

# 額外：常見同義/寫法修正（可再擴充）
_REPLACEMENTS = [
    ("年度", "年"),
    ("年 度", "年"),
    ("　", ""),  # 全形空白
]

_PUNCT_RE = re.compile(r"[，,。．、\s]+")
_YEAR_RE = re.compile(r"(?P<y>\d{3})\s*年")


def _wants_summary(user_text: str) -> bool:
    """輸入含「比較/變化/異動...」才顯示年度差異摘要（含趨勢一句話）。"""
    t = str(user_text or "")
    return any(k in t for k in ANALYSIS_KEYWORDS)


def _strip_analysis_keywords(text: str) -> str:
    """把『比較/變化/異動...』等分析詞從查詢中移除，避免影響題庫匹配。"""
    t = str(text or "")
    for k in ANALYSIS_KEYWORDS:
        t = t.replace(k, "")
    return t.strip()


def _normalize(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    for a, b in _REPLACEMENTS:
        t = t.replace(a, b)
    t = _PUNCT_RE.sub("", t)
    return t


def _extract_year(text: str) -> Optional[str]:
    """抓第一個年度（113年 / 113）"""
    m = _YEAR_RE.search(text)
    if m:
        return m.group("y")
    m2 = re.search(r"(?P<y>\d{3})", text)
    return m2.group("y") if m2 else None


def _strip_year(text_norm: str) -> str:
    t = _YEAR_RE.sub("", text_norm)
    t = re.sub(r"\d{3}", "", t)
    return t


def extract_years(text: str) -> List[int]:
    """
    支援多年度輸入：
      - 112-113年
      - 112~113年
      - 112至113年 / 112到113年
      - 112,113年 / 112、113年（會取出所有三位數年度）
    回傳：升冪年份清單，例如 [112, 113]
    """
    s = str(text or "")

    # 1) 範圍（含「年」或不含都可）
    m = re.search(r"(\d{3})\s*[-~－—]\s*(\d{3})\s*年?", s)
    if not m:
        m = re.search(r"(\d{3})\s*(?:至|到)\s*(\d{3})\s*年?", s)

    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        lo, hi = min(y1, y2), max(y1, y2)
        span = hi - lo + 1
        if span > MAX_YEAR_SPAN:
            return []
        return list(range(lo, hi + 1))

    # 2) 非範圍：抓出所有三位數年度（去重）
    years = re.findall(r"(\d{3})\s*年?", s)
    if years:
        uniq = sorted({int(y) for y in years})
        return uniq

    return []


def strip_year_expression(text: str) -> str:
    """
    把文字中的「年度表達」移除，留下「主題」：
      112-113年工務局暨所屬職員人數 -> 工務局暨所屬職員人數
    """
    s = str(text or "")

    # 先去掉範圍
    s = re.sub(r"\d{3}\s*[-~－—]\s*\d{3}\s*年?", "", s)
    s = re.sub(r"\d{3}\s*(?:至|到)\s*\d{3}\s*年?", "", s)

    # 再去掉單一年（避免殘留）
    s = re.sub(r"\d{3}\s*年", "", s)
    s = re.sub(r"\d{3}", "", s)

    return s.strip()


@dataclass(frozen=True)
class Entry:
    keyword: str
    keyword_norm: str
    keyword_norm_noyear: str
    year: Optional[str]
    description: str
    unit: str
    source_url: str


_EXACT_MAP: Dict[str, Entry] = {}
_ENTRIES: List[Entry] = []


def _format_answer(entry: Entry) -> str:
    """
    回覆內容僅回傳 description（不在訊息中顯示連結），
    以免佔用 LINE 版面；連結建議集中於圖文選單。
    """
    return entry.description


def _load_training() -> None:
    global _EXACT_MAP, _ENTRIES

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] training file not found: {DATA_PATH}")
        _EXACT_MAP = {}
        _ENTRIES = []
        return

    df = pd.read_excel(DATA_PATH, dtype=str).fillna("")
    cols = [c.strip().lower() for c in df.columns]
    colmap = {c.strip().lower(): c for c in df.columns}

    required = ["keywords", "description", "source_url"]
    missing = [c for c in required if c not in cols]
    if missing:
        print(f"[ERROR] training file missing columns: {missing}. Found: {list(df.columns)}")
        _EXACT_MAP = {}
        _ENTRIES = []
        return

    kw_col = colmap["keywords"]
    desc_col = colmap["description"]
    src_col = colmap["source_url"]
    unit_col = colmap.get("unit")  # unit 欄可有可無

    exact_map: Dict[str, Entry] = {}
    entries: List[Entry] = []

    for _, r in df.iterrows():
        kw_raw = str(r.get(kw_col, "")).strip()
        desc = str(r.get(desc_col, "")).strip()
        src = str(r.get(src_col, "")).strip()
        unit = str(r.get(unit_col, "")).strip() if unit_col else ""

        if not kw_raw or not desc:
            continue

        kw_norm = _normalize(kw_raw)
        y = _extract_year(kw_raw)
        kw_norm_noyear = _strip_year(kw_norm)

        e = Entry(
            keyword=kw_raw,
            keyword_norm=kw_norm,
            keyword_norm_noyear=kw_norm_noyear,
            year=y,
            description=desc,
            unit=unit,
            source_url=src,
        )

        exact_map[kw_norm] = e
        entries.append(e)

    _EXACT_MAP = exact_map
    _ENTRIES = entries
    print(f"[DEBUG] training loaded: {DATA_PATH}, entries={len(_ENTRIES)}")


_load_training()


def _match_by_exact(user_text: str) -> Optional[Entry]:
    key = _normalize(user_text)
    if not key:
        return None
    return _EXACT_MAP.get(key)


def _coverage_ratio(keyword_norm: str, user_norm: str) -> float:
    if not keyword_norm:
        return 0.0
    kw = Counter(keyword_norm)
    us = Counter(user_norm)
    hit = sum(min(cnt, us.get(ch, 0)) for ch, cnt in kw.items())
    return hit / max(1, len(keyword_norm))


def _rank_matches(user_text: str, use_year_filter: bool = True) -> List[Tuple[float, int, Entry]]:
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
        tie = len(e.keyword_norm)
        ranked.append((r, tie, e))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked


def _rank_matches_noyear(user_text: str) -> List[Tuple[float, int, Entry]]:
    user_norm = _normalize(user_text)
    if not user_norm:
        return []

    user_norm_noyear = _strip_year(user_norm)
    if not user_norm_noyear:
        return []

    ranked: List[Tuple[float, int, Entry]] = []
    for e in _ENTRIES:
        r = _coverage_ratio(e.keyword_norm_noyear, user_norm_noyear)
        tie = len(e.keyword_norm_noyear)
        ranked.append((r, tie, e))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked


def build_reply_single_year(user_text: str) -> str:
    """
    單年度查詢邏輯（保留原本互動文案/候選提示）。
    """
    text = (user_text or "").strip()
    if not text:
        return DEFAULT_REPLY

    user_norm = _normalize(text)
    user_year = _extract_year(text)

    # 1) 少打年度：先提醒補年度（優先於太短引導）
    if not user_year:
        ranked_noyear = _rank_matches_noyear(text)
        if ranked_noyear:
            best_r2, _, _ = ranked_noyear[0]
            if best_r2 >= COVERAGE_THRESHOLD:
                return (
                    "看起來您可能少輸入「年度」。\n"
                    "請在問題前面加上年度（例如：113年）再查詢一次。"
                )

    # 2) 太短引導
    if len(user_norm) < MIN_QUERY_LEN:
        return (
            "請輸入更完整的查詢關鍵詞（含年度/單位/指標），例如：\n"
            "- 113年工務局主管預算數\n"
            "- 113年工務局主管經常門\n"
            "- 113年工務局暨所屬職員人數"
        )

    # 3) 完全符合
    e_exact = _match_by_exact(text)
    if e_exact:
        return _format_answer(e_exact)

    # 4) 年度一致下的覆蓋率比對
    ranked = _rank_matches(text, use_year_filter=True)
    if ranked:
        best_r, _, best_e = ranked[0]
        if best_r >= COVERAGE_THRESHOLD:
            return _format_answer(best_e)

    # 5) 關鍵詞不夠完整：列出最接近 3 筆（不顯示相符率）
    if ranked:
        best_r, _, _ = ranked[0]
        if best_r >= SUGGEST_THRESHOLD:
            picks = ranked[:SUGGEST_TOPN]
            lines = "\n".join([f"- {e.keyword}" for _, _, e in picks])
            return (
                "您是不是要找下列資料：\n"
                f"{lines}\n"
                "（若都不是，請再補充更完整的關鍵詞，例如：年度＋單位＋項目＋職等/門別）"
            )

    return DEFAULT_REPLY


# =========================
# 多年度專用
# =========================
def _get_entry_for_year_query(query_text: str) -> Optional[Entry]:
    """
    多年度用：給定「已含年度」的 query（例如：113年工務局職員人數），
    直接回傳最可能的 Entry；找不到就回 None。
    """
    text = (query_text or "").strip()
    if not text:
        return None

    # 1) 完全符合
    e_exact = _match_by_exact(text)
    if e_exact:
        return e_exact

    # 2) 年度一致下的覆蓋率比對（多年度不做候選提示）
    ranked = _rank_matches(text, use_year_filter=True)
    if ranked:
        best_r, _, best_e = ranked[0]
        if best_r >= COVERAGE_THRESHOLD:
            return best_e

    return None


def _extract_total_value(desc_text: str) -> Optional[int]:
    """從 description 抓總計/總數/合計後面的數字（允許逗號）。"""
    if not desc_text:
        return None
    m = re.search(r"(總計|總數|合計)\s*([\d,]+)", desc_text)
    if not m:
        return None
    return int(m.group(2).replace(",", ""))


def _fallback_extract_unit(desc_text: str) -> str:
    """
    若 unit 欄沒填，從 description 嘗試抓短單位（避免完全沒單位）。
    例：總計524人 / 總計8,194,228千元
    """
    if not desc_text:
        return ""
    m = re.search(r"(總計|總數|合計)\s*[\d,]+\s*([^\d\s，。；;、()（）]{1,8})", desc_text)
    return (m.group(2) or "").strip() if m else ""


def _extract_first_url(ans_text: str) -> str:
    """從回覆文字中抓第一個 URL。"""
    if not ans_text:
        return ""
    m = re.search(r"(https?://\S+)", ans_text)
    return m.group(1) if m else ""


def _extract_source_text_and_url(ans_text: str) -> Tuple[str, str]:
    """
    從單年度回覆中擷取：
    - 資料來源文字（例如：高雄市政府工務局性別統計年報。）
    - 第一個 URL
    """
    if not ans_text:
        return "", ""

    lines = [l.strip() for l in ans_text.splitlines() if l.strip()]

    source_text = ""
    source_url = ""

    for i, line in enumerate(lines):
        if not source_url:
            m = re.search(r"(https?://\S+)", line)
            if m:
                source_url = m.group(1)

        if "資料來源" in line and i + 1 < len(lines):
            source_text = lines[i + 1]

    return source_text, source_url


def _format_multiyear_reply(
    years: List[int],
    year_to_entry: Dict[int, Optional[Entry]],
    base_topic: str,
    show_summary: bool,
) -> str:
    """
    多年度格式化：
    - 一行一年度：113年XXX總計NN{unit}
    - 缺漏年度集中列示
    - 資料來源顯示一次（來源文字 + URL）
    - show_summary=True：加趨勢摘要 + 年度差異摘要（用同一單位）
    """
    if not years:
        return DEFAULT_REPLY

    lines_out: List[str] = []
    missing: List[int] = []

    totals: Dict[int, int] = {}

    source_text = ""
    source_url = ""

    def _topic_no_suffix(topic: str) -> str:
        t = (topic or "").strip()
        # 若主題以「人數」結尾，顯示時去掉「人數」
        if t.endswith("人數"):
            t = t[:-2]
        return t

    def _format_multiyear_line(year: int, topic: str, total: Optional[int], unit: str) -> str:
        t = _topic_no_suffix(topic)
        if total is None:
            return f"{year}年{t}（查無總計數字）"
        # 多年度數字加千分位
        return f"{year}年{t}總計{total:,}{unit}"

    def _pick_summary_unit(years_sorted: List[int], unit_map: Dict[int, str]) -> str:
        for y in sorted(years_sorted, reverse=True):
            u = (unit_map.get(y) or "").strip()
            if u:
                return u
        return ""

    def _trend_sentence(years2: List[int], totals2: Dict[int, int], unit: str) -> str:
        ys = sorted([y for y in years2 if y in totals2])
        if len(ys) < 2:
            return ""

        first_y, last_y = ys[0], ys[-1]
        first_v, last_v = totals2[first_y], totals2[last_y]
        diff = last_v - first_v
        base = first_v if first_v != 0 else 1
        diff_pct = diff / base * 100.0

        series = [totals2[y] for y in ys]
        avg = sum(series) / max(1, len(series))
        rng = max(series) - min(series)
        vol_ratio = (rng / avg) if avg != 0 else 0.0

        if abs(diff_pct) < 1.0:
            overall = "整體大致持平"
        else:
            overall = (
                "整體呈現小幅成長"
                if diff > 0 and abs(diff_pct) < 5.0
                else (
                    "整體呈現成長"
                    if diff > 0
                    else ("整體呈現小幅下降" if abs(diff_pct) < 5.0 else "整體呈現下降")
                )
            )

        if vol_ratio <= 0.03:
            volatility = "相對穩定"
        elif vol_ratio <= 0.08:
            volatility = "呈現小幅波動"
        else:
            volatility = "波動較明顯"

        recent_phrase = ""
        if len(ys) >= 2:
            prev_y = ys[-2]
            prev_v = totals2[prev_y]
            recent_diff = last_v - prev_v
            if recent_diff > 0:
                recent_phrase = f"{last_y}年較前期略為回升"
            elif recent_diff < 0:
                recent_phrase = f"{last_y}年較前期略為下滑"
            else:
                recent_phrase = f"{last_y}年與前期持平"

        period = f"{first_y}–{last_y}年"
        if overall == "整體大致持平":
            main = f"{period}整體{volatility}"
        else:
            main = f"{period}{overall}，走勢{volatility}" if volatility == "相對穩定" else f"{period}整體{volatility}"

        unit_hint = f"（單位：{unit}）" if unit else ""
        return (
            f"（趨勢摘要）\n{main}，{recent_phrase}。{unit_hint}"
            if recent_phrase
            else f"（趨勢摘要）\n{main}。{unit_hint}"
        )

    unit_map: Dict[int, str] = {}

    for y in sorted(years, reverse=True):
        e = year_to_entry.get(y)
        if not e:
            missing.append(y)
            continue

        if not source_url and not source_text:
            st, su = _extract_source_text_and_url(_format_answer(e))
            source_text = st
            source_url = su
        elif not source_url:
            source_url = _extract_first_url(_format_answer(e))

        total = _extract_total_value(e.description)
        unit = (e.unit or "").strip()
        if not unit:
            unit = _fallback_extract_unit(e.description)

        unit_map[y] = unit

        if total is not None:
            totals[y] = total

        lines_out.append(_format_multiyear_line(y, base_topic, total, unit))

    body = "\n".join(lines_out) if lines_out else "（本次範圍內皆查無符合資料）"

    if missing:
        miss = "、".join([f"{m}年" for m in sorted(missing, reverse=True)])
        body = f"{body}\n\n（查無資料年度：{miss}）"
    # （資料來源）連結不在回覆中顯示：已移至圖文選單

    if show_summary and len(totals) >= 2:
        summary_unit = _pick_summary_unit(years, unit_map)
        trend = _trend_sentence(years, totals, summary_unit)
        if trend:
            body = f"{body}\n\n{trend}"

    if show_summary and len(totals) >= 2:
        ys = sorted(totals.keys())
        summary_unit = _pick_summary_unit(ys, unit_map)

        summary_lines = ["（年度差異摘要）"]
        for i in range(1, len(ys)):
            y1, y2 = ys[i - 1], ys[i]
            v1, v2 = totals[y1], totals[y2]
            diff = v2 - v1
            pct = (diff / v1 * 100) if v1 != 0 else 0.0
            sign = "+" if diff >= 0 else ""
            # 差異摘要 diff 也加千分位
            summary_lines.append(f"{y2}年較{y1}年 {sign}{diff:,}{summary_unit}（{sign}{pct:.2f}%）")

        body = f"{body}\n\n" + "\n".join(summary_lines)

    return body


# =========================
# footer 分流：查到 / 查不到
# =========================
def _is_success_reply(reply: str) -> bool:
    """
    判斷「是否查到資料」：
    - DEFAULT_REPLY -> 失敗
    - 引導/提醒/候選 -> 視為未查到（使用 fallback 文案）
    - 多年度全無 -> 視為未查到
    - 其餘 -> 視為查到（使用 success 文案）
    """
    r = (reply or "").strip()
    if not r:
        return False

    if r == DEFAULT_REPLY:
        return False

    if r.startswith("請輸入更完整的查詢關鍵詞"):
        return False
    if r.startswith("看起來您可能少輸入「年度」"):
        return False
    if r.startswith("您是不是要找下列資料："):
        return False

    if "（本次範圍內皆查無符合資料）" in r:
        return False

    return True


def _prepend_result_header(reply: str) -> str:
    """
    只有「查到資料」時才加上『查詢結果如下：』，避免干擾引導/候選訊息。
    且避免重複加標頭。
    """
    r = (reply or "").strip()
    if not r:
        return r
    if r.startswith(RESULT_HEADER):
        return r
    if _is_success_reply(r):
        return f"{RESULT_HEADER}\n{r}"
    return r


def _append_survey_footer(reply: str) -> str:
    """
    版面精簡：不在回覆中附加「滿意度調查/回饋連結」。
    建議改由圖文選單提供「滿意度調查」「意見回饋」「資料來源」等入口。
    """
    return (reply or "").rstrip()


def build_reply(user_text: str) -> str:
    """
    多年度入口：偵測到「年度範圍」就拆成多筆單年度查詢，最後合併回覆。
    否則走單年度流程。

    流程：
    1) 產出 reply
    2) 若查到資料 → 前置「查詢結果如下：」
    3) 依查到/查不到 → 附加滿意度問卷 footer
    """
    text = (user_text or "").strip()
    if not text:
        return _append_survey_footer(DEFAULT_REPLY)

    years = extract_years(text)

    if len(years) >= 2:
        show_summary = _wants_summary(text)

        cleaned = _strip_analysis_keywords(text)
        base_topic = strip_year_expression(cleaned)

        year_to_entry: Dict[int, Optional[Entry]] = {}
        for y in years:
            q = f"{y}年{base_topic}"
            year_to_entry[y] = _get_entry_for_year_query(q)

        reply = _format_multiyear_reply(years, year_to_entry, base_topic, show_summary)
        reply = _prepend_result_header(reply)
        return _append_survey_footer(reply)

    reply = build_reply_single_year(text)
    reply = _prepend_result_header(reply)
    return _append_survey_footer(reply)
