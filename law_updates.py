"""Utilities for fetching and analysing latest law revision updates."""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def re_split(text: str) -> List[str]:
    """Split Japanese sentences while keeping heuristics lightweight."""

    pattern = re.compile(r"(?<=[。！？])\s*|\n+")
    parts = pattern.split(text)
    return [part for part in parts if part is not None]


@dataclass
class LawRevisionDocument:
    """A normalized representation of a fetched law revision document."""

    law_name: str
    body: str
    title: str
    revision_year: Optional[int] = None
    effective_date: Optional[str] = None
    source: str = ""
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LawUpdateResult:
    """Summary returned after executing a synchronization."""

    source: str
    fetched_at: dt.datetime
    status: str
    message: str
    revisions_detected: int
    questions_generated: int


class LawRevisionAnalyzer:
    """Light-weight NLP pipeline for extracting key points from law text."""

    def __init__(self) -> None:
        try:
            import spacy  # type: ignore

            # ja_core_news_sm is the smallest Japanese model and fast to load. The
            # call is wrapped in a try/except so that the pipeline gracefully
            # degrades when the model is not present in the execution environment.
            with self._suppress_exceptions():
                self._nlp = spacy.load("ja_core_news_sm")  # type: ignore[arg-type]
            if not hasattr(self, "_nlp"):
                self._nlp = None
        except Exception:  # pragma: no cover - optional dependency
            self._nlp = None

        self._fallback_stopwords = {"こと", "もの", "ため", "改正", "施行"}

    class _suppress_exceptions:
        def __enter__(self) -> "LawRevisionAnalyzer._suppress_exceptions":
            return self

        def __exit__(
            self,
            exc_type: Optional[type],
            exc: Optional[BaseException],
            traceback: Optional[Any],
        ) -> bool:
            return True

    def summarize(self, text: str, max_sentences: int = 2) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return text.strip()
        ranked = sorted(
            sentences,
            key=lambda s: (-self._score_sentence(s), sentences.index(s)),
        )
        summary = "\n".join(ranked[:max_sentences])
        return summary.strip()

    def extract_keywords(self, text: str, limit: int = 5) -> List[str]:
        tokens: List[str] = []
        if getattr(self, "_nlp", None) is not None:
            doc = self._nlp(text)  # type: ignore[operator]
            for token in doc:
                if token.pos_ in {"NOUN", "PROPN", "NUM"} and token.text not in self._fallback_stopwords:
                    tokens.append(token.text)
        else:
            tokens = [tok for tok in self._simple_tokenize(text) if tok not in self._fallback_stopwords]
        unique_tokens: List[str] = []
        for tok in tokens:
            if tok not in unique_tokens and len(tok) > 1:
                unique_tokens.append(tok)
            if len(unique_tokens) >= limit:
                break
        return unique_tokens

    def build_cloze_question(self, law_name: str, text: str) -> Dict[str, Any]:
        summary = self.summarize(text)
        keywords = self.extract_keywords(summary, limit=4)
        if not keywords:
            keywords = [law_name]
        answer = keywords[0]
        distractors = self._build_distractors(answer, keywords[1:])
        statement = summary.replace(answer, "＿＿＿", 1)
        if statement == summary:
            statement = f"{summary}\n空欄: ＿＿＿"
        question = f"{law_name}の改正ポイントについて、空欄に入る語句はどれか。\n{statement}"
        choices = [answer] + distractors
        random.shuffle(choices)
        correct_idx = choices.index(answer) + 1
        return {
            "question": question,
            "choices": choices,
            "correct": correct_idx,
            "summary": summary,
            "cloze": statement,
        }

    def _split_sentences(self, text: str) -> List[str]:
        cleaned = text.replace("\r", " ").replace("\n", "\n")
        sentences = [seg.strip() for seg in re_split(cleaned) if seg.strip()]
        return sentences

    def _score_sentence(self, sentence: str) -> float:
        score = 0.0
        keywords = ["改正", "施行", "義務", "新設", "追加", "緩和", "強化", "必要"]
        for kw in keywords:
            if kw in sentence:
                score += 1.5
        score += min(len(sentence) / 40.0, 2.0)
        return score

    def _simple_tokenize(self, text: str) -> List[str]:
        tokens = []
        buffer = ""
        for char in text:
            if char.isspace():
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
            elif ord(char) < 128 and not char.isalnum():
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
            else:
                buffer += char
        if buffer:
            tokens.append(buffer)
        return tokens

    def _build_distractors(self, answer: str, candidates: Sequence[str]) -> List[str]:
        distractors: List[str] = list(dict.fromkeys([c for c in candidates if c != answer]))
        if len(distractors) >= 3:
            return distractors[:3]
        fallback_terms = ["届出", "免許", "説明", "手続", "報告", "期限"]
        while len(distractors) < 3:
            term = random.choice(fallback_terms)
            if term not in distractors and term != answer:
                distractors.append(term)
        return distractors[:3]


DEFAULT_LAW_SOURCES = [
    {
        "name": "REAL_ESTATE_PROMOTION_ORG",
        "url": "https://example.com/ftri/latest.json",
        "format": "json",
    },
    {
        "name": "SPECIALIST_SCHOOL_FEED",
        "url": "https://example.com/prep-school/law-updates.json",
        "format": "json",
    },
]


class LawUpdateSyncService:
    """Synchronize external law revision feeds and persist generated questions."""

    def __init__(
        self,
        sources: Optional[Iterable[Dict[str, Any]]] = None,
        analyzer: Optional[LawRevisionAnalyzer] = None,
        id_builder: Optional[Callable[[pd.Series], str]] = None,
    ) -> None:
        self.sources = list(sources or DEFAULT_LAW_SOURCES)
        self.analyzer = analyzer or LawRevisionAnalyzer()
        self.id_builder = id_builder
        self.sample_feed_path = Path("docs/law_revision_sample_feed.json")

    def run(self, db: "DBManager") -> List[LawUpdateResult]:  # pragma: no cover - streamlit integration
        results: List[LawUpdateResult] = []
        for source in self.sources:
            res = self._process_source(db, source)
            results.append(res)
        return results

    def _process_source(self, db: "DBManager", source: Dict[str, Any]) -> LawUpdateResult:
        source_name = source.get("name", "UNKNOWN")
        fetched_at = dt.datetime.utcnow()
        try:
            payload = self._fetch_with_retry(source)
            documents = self._parse_payload(payload, source)
            question_rows = self._generate_questions(documents)
            if question_rows:
                df = pd.DataFrame(question_rows)
                if self.id_builder is not None:
                    df["id"] = df.apply(self.id_builder, axis=1)
                inserted, updated = db.upsert_law_revision_questions(df)
                message = f"{inserted} inserted / {updated} updated"
                status = "success"
            else:
                message = "No questions generated"
                status = "empty"
            revisions_detected = len(documents)
            questions_generated = len(question_rows)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Law update synchronization failed: %s", exc)
            message = str(exc)
            status = "error"
            revisions_detected = 0
            questions_generated = 0
        result = LawUpdateResult(
            source=source_name,
            fetched_at=fetched_at,
            status=status,
            message=message,
            revisions_detected=revisions_detected,
            questions_generated=questions_generated,
        )
        if hasattr(db, "record_law_revision_sync"):
            db.record_law_revision_sync(result)
        return result

    def _fetch_with_retry(self, source: Dict[str, Any]) -> Any:
        url = source.get("url")
        if not url:
            raise ValueError("Feed URL is not configured")
        attempts = int(source.get("retries", 2))
        backoff = float(source.get("backoff", 1.5))
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                if source.get("format") == "json":
                    return response.json()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Failed to fetch %s (attempt %s/%s): %s", url, attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(backoff * attempt)
        logger.error("Falling back to bundled sample feed due to fetch failure: %s", last_error)
        if self.sample_feed_path.exists():
            with self.sample_feed_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        raise last_error or RuntimeError("Failed to fetch law revision feed")

    def _parse_payload(self, payload: Any, source: Dict[str, Any]) -> List[LawRevisionDocument]:
        documents: List[LawRevisionDocument] = []
        items: Iterable[Dict[str, Any]]
        if isinstance(payload, dict) and "items" in payload:
            items = payload.get("items", [])
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("Unsupported feed payload")
        source_name = source.get("name", "UNKNOWN")
        for item in items:
            law_name = item.get("law_name") or item.get("title") or "不明な法令"
            body = item.get("body") or item.get("summary") or ""
            if not body:
                continue
            revision_year = _safe_int(item.get("revision_year"))
            effective_date = item.get("effective_date") or item.get("effective")
            documents.append(
                LawRevisionDocument(
                    law_name=law_name,
                    body=body,
                    title=item.get("title", law_name),
                    revision_year=revision_year,
                    effective_date=effective_date,
                    source=source_name,
                    url=item.get("url"),
                    metadata=item,
                )
            )
        return documents

    def _generate_questions(self, documents: Sequence[LawRevisionDocument]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for doc in documents:
            cloze = self.analyzer.build_cloze_question(doc.law_name, doc.body)
            row = {
                "label": doc.title,
                "law_name": doc.law_name,
                "revision_year": doc.revision_year,
                "effective_date": doc.effective_date or "",
                "category": "法改正",  # default category when absent
                "topic": doc.metadata.get("topic") if doc.metadata else "",
                "source": doc.source,
                "question": cloze["question"],
                "choice1": cloze["choices"][0],
                "choice2": cloze["choices"][1],
                "choice3": cloze["choices"][2],
                "choice4": cloze["choices"][3],
                "correct": cloze["correct"],
                "explanation": cloze["summary"],
                "difficulty": 3,
                "tags": "法改正;自動生成",
                "auto_summary": cloze["summary"],
                "auto_cloze": cloze["cloze"],
                "review_status": "pending",
                "generated_from": doc.source,
                "fetched_at": dt.datetime.utcnow(),
            }
            rows.append(row)
        return rows


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "LawRevisionAnalyzer",
    "LawRevisionDocument",
    "LawUpdateResult",
    "LawUpdateSyncService",
    "DEFAULT_LAW_SOURCES",
]
