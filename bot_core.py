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

# ===== 滿意度調查（查到/查不到 分流）=====
SURVEY_URL = os.environ.get("SURVEY_URL", "https://forms.gle/HCLmWz3br3egcBRN8")

SURVEY_FOOTER_SUCCESS = (
    "────────────────\n"
    "📊 滿意度調查（約 30 秒）\n"
    "為持續精進「AI 工務局主計問答系統」，誠摯邀請您填寫使用體驗回饋與建議：\n"
    f"👉 {SURVEY_URL}\n"
    "（本問卷不蒐集個人資料，僅作系統改善參考，感謝您的協助）"
)

SURVEY_FOOTER_FALLBACK = (
    "────────────────\n"
    "📝 回饋與建議\n"
    "若本系統回覆不符合您的查詢需求，可否協助提供您寶貴的建議事項，"
    "作為本系統後續精進與資料更新之參考：\n"
    f"👉 {SURVEY_URL}"
)

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
            # 太長就不做多年度（避免被濫用），回空讓後面走單年度提示
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
    source_url: str


_EXACT_MAP: Dict[str, Entry] = {}
_ENTRIES: List[Entry] = []


def _format_answer(entry: Entry) -> str:
    """
    LINE 不支援 markdown hyperlink，但會把純網址自動轉成可點連結，
    所以用「內容 + 換行 + URL」最穩。
    """
    if entry.source_url:
        return f"{entry.description}\n{entry.source_url}"
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

    exact_map: Dict[str, Entry] = {}
    entries: List[Entry] = []

    for _, r in df.iterrows():
        kw_raw = str(r.get(kw_col, "")).strip()
        desc = str(r.get(desc_col, "")).strip()
        src = str(r.get(src_col, "")).strip()

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
    單年度查詢邏輯（完整保留）。
    """
    text = (user_text or "").strip()
    if not text:
        return DEFAULT_REPLY

    user_norm = _normalize(text)
    user_year = _extract_year(text)

    # A) 太短先引導
    if len(user_norm) < MIN_QUERY_LEN:
        return (
            "請輸入更完整的查詢關鍵詞（含年度/單位/指標），例如：\n"
            "- 113年工務局主管預算數\n"
            "- 113年工務局主管經常門\n"
            "- 113年工務局暨所屬職員人數"
        )

    # 1) 完全符合
    e_exact = _match_by_exact(text)
    if e_exact:
        return _format_answer(e_exact)

    # 2) 年度一致下的覆蓋率比對
    ranked = _rank_matches(text, use_year_filter=True)
    if ranked:
        best_r, _, best_e = ranked[0]
        if best_r >= COVERAGE_THRESHOLD:
            return _format_answer(best_e)

    # C) 少打年度：只提醒補年度（不列候選）
    if not user_year:
        ranked_noyear = _rank_matches_noyear(text)
        if ranked_noyear:
            best_r2, _, _ = ranked_noyear[0]
            if best_r2 >= COVERAGE_THRESHOLD:
                return (
                    "看起來您可能少輸入「年度」。\n"
                    "請在問題前面加上年度（例如：113年）再查詢一次。"
                )

    # B) 關鍵詞不夠完整：列出最接近 3 筆（不顯示相符率）
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


def _extract_total_people(ans_text: str) -> Optional[int]:
    """
    從回覆文字中抓總計人數（只抓總計/總數/合計後面的數字）：
    例如：總計524人 / 總數524人 / 合計524人
    """
    if not ans_text:
        return None
    m = re.search(r"(總計|總數|合計)\s*(\d+)\s*人", ans_text)
    if m:
        return int(m.group(2))
    return None


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
        # URL
        if not source_url:
            m = re.search(r"(https?://\S+)", line)
            if m:
                source_url = m.group(1)

        # 來源文字（通常在「(資料來源)」或「資料來源」後一行）
        if "資料來源" in line and i + 1 < len(lines):
            source_text = lines[i + 1]

    return source_text, source_url


def _format_multiyear_reply(
    years: List[int],
    year_to_text: Dict[int, Optional[str]],
    base_topic: str,
    show_summary: bool,
) -> str:
    """
    多年度格式化（保持你目前版本）：
    - 只列出各年度「總計」(精簡版)
    - 缺漏年度集中列示
    - 必要時附年度差異摘要（仍以「總計」計算）
    - 資料來源顯示一次（來源文字 + URL）
    - 選項A：若 show_summary=True，額外附「趨勢摘要」一句話
    """
    if not years:
        return DEFAULT_REPLY

    blocks: List[str] = []
    missing: List[int] = []

    totals: Dict[int, int] = {}

    source_text = ""
    source_url = ""

    def _format_multiyear_compact_line(year: int, base_topic2: str, total: Optional[int]) -> str:
        topic = (base_topic2 or "").strip()
        if topic.endswith("人數"):
            topic = topic[:-2]
        if total is None:
            return f"【{year}年】\n{year}年{topic}（查無總計數字）"
        return f"【{year}年】\n{year}年{topic}總計{total}人"

    def _trend_sentence_from_totals(years2: List[int], totals2: Dict[int, int]) -> str:
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
            overall = "整體呈現小幅成長" if diff > 0 and abs(diff_pct) < 5.0 else (
                "整體呈現成長" if diff > 0 else ("整體呈現小幅下降" if abs(diff_pct) < 5.0 else "整體呈現下降")
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

        return f"（趨勢摘要）\n{main}，{recent_phrase}。" if recent_phrase else f"（趨勢摘要）\n{main}。"

    # 年度資料（新到舊）
    for y in sorted(years, reverse=True):
        ans = year_to_text.get(y)
        if not ans or ans == DEFAULT_REPLY:
            missing.append(y)
            continue

        if not source_url and not source_text:
            st, su = _extract_source_text_and_url(ans)
            source_text = st
            source_url = su
        elif not source_url:
            source_url = _extract_first_url(ans)

        t = _extract_total_people(ans)
        if t is not None:
            totals[y] = t

        blocks.append(_format_multiyear_compact_line(y, base_topic, t))

    body = "\n\n".join(blocks) if blocks else "（本次範圍內皆查無符合資料）"

    if missing:
        miss = "、".join([f"{m}年" for m in sorted(missing, reverse=True)])
        body = f"{body}\n\n（查無資料年度：{miss}）"

    if source_text or source_url:
        body = f"{body}\n\n（資料來源）"
        if source_text:
            body += f"\n{source_text}"
        if source_url:
            body += f"\n{source_url}"

    if show_summary and len(totals) >= 2:
        trend = _trend_sentence_from_totals(years, totals)
        if trend:
            body = f"{body}\n\n{trend}"

    if show_summary and len(totals) >= 2:
        ys = sorted(totals.keys())
        summary_lines = ["（年度差異摘要）"]
        for i in range(1, len(ys)):
            y1, y2 = ys[i - 1], ys[i]
            v1, v2 = totals[y1], totals[y2]
            diff = v2 - v1
            pct = (diff / v1 * 100) if v1 != 0 else 0.0
            sign = "+" if diff >= 0 else ""
            summary_lines.append(f"{y2}年較{y1}年 {sign}{diff}人（{sign}{pct:.2f}%）")
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

    # 引導/提醒/候選（這些都不算「查到資料」）
    if r.startswith("請輸入更完整的查詢關鍵詞"):
        return False
    if r.startswith("看起來您可能少輸入「年度」"):
        return False
    if r.startswith("您是不是要找下列資料："):
        return False

    # 多年度全無
    if "（本次範圍內皆查無符合資料）" in r:
        return False

    return True


def _append_survey_footer(reply: str) -> str:
    """
    依「查到/查不到」附上不同文案（避免重複附加）。
    """
    r = (reply or "").rstrip()
    if SURVEY_URL in r:
        return r  # 已附過就不再附

    footer = SURVEY_FOOTER_SUCCESS if _is_success_reply(r) else SURVEY_FOOTER_FALLBACK
    return f"{r}\n\n{footer}" if r else footer


def build_reply(user_text: str) -> str:
    """
    多年度入口：偵測到「年度範圍」就拆成多筆單年度查詢，最後合併回覆。
    否則走單年度流程。

    最後依「查到/查不到」附上不同滿意度/建議文案。
    """
    text = (user_text or "").strip()
    if not text:
        return _append_survey_footer(DEFAULT_REPLY)

    years = extract_years(text)

    # 多年度：至少 2 年才進入合併模式
    if len(years) >= 2:
        show_summary = _wants_summary(text)

        cleaned = _strip_analysis_keywords(text)
        base_topic = strip_year_expression(cleaned)

        year_to_text: Dict[int, Optional[str]] = {}
        for y in years:
            q = f"{y}年{base_topic}"
            year_to_text[y] = build_reply_single_year(q)

        reply = _format_multiyear_reply(years, year_to_text, base_topic, show_summary)
        return _append_survey_footer(reply)

    # 單年度
    reply = build_reply_single_year(text)
    return _append_survey_footer(reply)
