# -*- coding: utf-8 -*-
"""
bot_core.py
- Loads training.xlsx
- Searches *description* field by keyword / fuzzy match
- Returns ONLY matched description(s) (no category/year/unit/item/value blocks)
"""

from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Tuple

import pandas as pd

# Optional: use rapidfuzz if available (better & faster); fallback to difflib otherwise.
try:
    from rapidfuzz import fuzz  # type: ignore
    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    import difflib
    _HAS_RAPIDFUZZ = False


# =========================
# Load training.xlsx
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("TRAINING_FILE", "training.xlsx")
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)

print("[DEBUG] bot_core.py loaded!")
print(f"[DEBUG] Expect training file at: {DATA_PATH}")

_REQUIRED_COLS = ["description"]  # minimal requirement


def _safe_str(x: Any) -> str:
    """Convert to string safely; treat NaN/NA as empty."""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _normalize(text: str) -> str:
    """
    Normalization for matching:
    - lowercase
    - remove spaces and common punctuation
    - unify 年度/年/度 variants (keep digits)
    """
    t = _safe_str(text).lower()
    # unify year wording a bit
    t = t.replace("年度", "年").replace("學年度", "年")
    # remove whitespace
    t = re.sub(r"\s+", "", t)
    # remove most punctuation/symbols (keep chinese/english/digits)
    t = re.sub(r"[~`!@#$%^&*()\-\_=+\[\]{}\\|;:'\",.<>/?，。！？、：；「」『』（）【】《》…·]", "", t)
    return t


def _score(query_norm: str, doc_norm: str) -> float:
    """Return similarity score 0-100."""
    if not query_norm or not doc_norm:
        return 0.0
    if query_norm in doc_norm:
        # direct hit should be very high
        return 100.0
    if _HAS_RAPIDFUZZ:
        # partial_ratio works well for "short query inside longer text"
        return float(fuzz.partial_ratio(query_norm, doc_norm))
    else:  # pragma: no cover
        return float(difflib.SequenceMatcher(None, query_norm, doc_norm).ratio() * 100)


def _build_doc_text(row: Dict[str, Any]) -> str:
    """
    Build a searchable text for each row.
    We focus on description, but also append year/unit/item/category/value to help matching.
    """
    parts = [
        _safe_str(row.get("description", "")),
        _safe_str(row.get("year", "")),
        _safe_str(row.get("category", "")),
        _safe_str(row.get("unit", "")),
        _safe_str(row.get("item", "")),
        _safe_str(row.get("value", "")),
    ]
    return "".join(parts)


# Load dataframe once at import
try:
    print(f"[DEBUG] Trying to load training file at: {DATA_PATH}")
    _df = pd.read_excel(DATA_PATH, dtype=str).fillna("")
    # normalize col names (strip)
    _df.columns = [c.strip() for c in _df.columns]
    for col in _REQUIRED_COLS:
        if col not in _df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Build cached docs
    _rows: List[Dict[str, Any]] = _df.to_dict(orient="records")
    _docs_norm: List[str] = [_normalize(_build_doc_text(r)) for r in _rows]

    print(f"[DEBUG] File loaded successfully! Rows={len(_rows)}, Columns={list(_df.columns)}")
except Exception as e:
    print("[ERROR] Failed to load training.xlsx:", repr(e))
    _rows = []
    _docs_norm = []


# =========================
# Public API
# =========================

_NOT_FOUND_MSG = "抱歉，我在訓練資料裡找不到這個問題的答案，可以換個說法或問別的問題喔。"


def build_reply(question: str) -> str:
    """
    Main entry for app.py
    - Search training rows by the user's keyword query.
    - Return matched description(s) only.
    """
    q = _safe_str(question)
    if not q:
        return _NOT_FOUND_MSG

    qn = _normalize(q)

    # Hard reject very short queries (too ambiguous)
    if len(qn) < 3:
        return _NOT_FOUND_MSG

    # Score all rows
    scored: List[Tuple[float, int]] = []
    for i, dn in enumerate(_docs_norm):
        s = _score(qn, dn)
        if s >= 75:  # threshold
            scored.append((s, i))

    if not scored:
        # If nothing passes threshold, try a softer rule: token contains (year+unit keywords etc.)
        # This catches cases like "113年工務局主管決算數" where user omits exact item name.
        # We do this by removing common filler words and checking containment.
        soft_q = qn.replace("較上一年", "").replace("變動", "").replace("比較", "")
        soft_q = soft_q.replace("決算數", "決算").replace("預算數", "預算")
        for i, dn in enumerate(_docs_norm):
            if soft_q and soft_q in dn:
                scored.append((90.0, i))

    if not scored:
        return _NOT_FOUND_MSG

    # Sort by score desc then keep top N
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:5]

    # Deduplicate descriptions while preserving order
    seen = set()
    answers: List[str] = []
    for _, idx in top:
        desc = _safe_str(_rows[idx].get("description", ""))
        if not desc:
            continue
        if desc in seen:
            continue
        seen.add(desc)
        answers.append(desc)

    if not answers:
        return _NOT_FOUND_MSG

    # Return all matched descriptions (each in its own paragraph)
    return "\n\n".join(answers)
