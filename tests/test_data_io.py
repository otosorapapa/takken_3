"""Tests for data import and loading helpers."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, insert

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


class DummyUploadedFile:
    """Simple stand-in for ``streamlit`` uploaded files."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class StreamlitStub:
    """Minimal stub that captures Streamlit status messages."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.session_state: dict[str, str] = {}

    @staticmethod
    def expander(*_args, **_kwargs):
        class _Expander:
            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb) -> bool:
                return False

        return _Expander()

    def warning(self, message: str) -> None:
        self.messages.append(("warning", str(message)))

    def error(self, message: str) -> None:
        self.messages.append(("error", str(message)))

    def info(self, message: str) -> None:
        self.messages.append(("info", str(message)))

    def success(self, message: str) -> None:
        self.messages.append(("success", str(message)))

    @staticmethod
    def markdown(*_args, **_kwargs) -> None:
        return None

    @staticmethod
    def dataframe(*_args, **_kwargs) -> None:
        return None

    @staticmethod
    def caption(*_args, **_kwargs) -> None:
        return None


def build_sample_csvs(base_dir: Path) -> None:
    questions_csv = textwrap.dedent(
        """
        year,q_no,category,topic,question,choice1,choice2,choice3,choice4,explanation,difficulty,tags
        2023,1,民法,基礎,宅建業法の定義は?,宅建士,宅地,建物,その他,詳しい説明,3,重要
        """
    ).strip()
    answers_csv = textwrap.dedent(
        """
        year,q_no,correct_number,correct_label,correct_text,explanation
        2023,1,1,A,宅建士,正しい選択肢
        """
    ).strip()
    (base_dir / "questions.csv").write_text(questions_csv, encoding="utf-8")
    (base_dir / "answers.csv").write_text(answers_csv, encoding="utf-8")


def test_initialize_from_csv_imports_questions(tmp_path, monkeypatch):
    data_dir = tmp_path / "import_data"
    data_dir.mkdir()
    build_sample_csvs(data_dir)

    engine = create_engine(f"sqlite:///{tmp_path / 'takken.db'}", future=True)
    app.metadata.create_all(engine)

    monkeypatch.setattr(app, "DATA_DIR", data_dir)
    monkeypatch.setattr(app, "rebuild_tfidf_cache", lambda: None)

    db = app.DBManager(engine)
    db.initialize_from_csv()

    imported = db.load_dataframe(app.questions_table)
    assert len(imported) == 1
    row = imported.iloc[0]
    assert row["question"].startswith("宅建業法")
    assert row["correct"] == 1


def test_execute_quick_import_merges_and_reports(monkeypatch):
    stub = StreamlitStub()
    monkeypatch.setattr(app, "st", stub)

    rebuild_calls: list[None] = []
    monkeypatch.setattr(app, "rebuild_tfidf_cache", lambda: rebuild_calls.append(None))

    class DummyDB:
        def __init__(self) -> None:
            self.upserted_df: pd.DataFrame | None = None

        def upsert_questions(self, df: pd.DataFrame) -> tuple[int, int]:
            self.upserted_df = df
            return len(df), 0

    questions_file = DummyUploadedFile(
        "questions.csv",
        textwrap.dedent(
            """
            year,q_no,category,topic,question,choice1,choice2,choice3,choice4,explanation,difficulty,tags
            2023,1,民法,基礎,宅建業法の定義は?,宅建士,宅地,建物,その他,詳しい説明,3,重要
            """
        ).strip().encode("utf-8"),
    )
    answers_file = DummyUploadedFile(
        "answers.csv",
        textwrap.dedent(
            """
            year,q_no,correct_number,correct_label,correct_text,explanation
            2023,1,1,A,宅建士,正しい選択肢
            """
        ).strip().encode("utf-8"),
    )

    db = DummyDB()
    app.execute_quick_import(db, questions_file, answers_file)

    assert db.upserted_df is not None
    assert len(db.upserted_df) == 1
    assert any(kind == "success" for kind, _ in stub.messages)
    assert rebuild_calls, "rebuild_tfidf_cache should be triggered after import"


def test_load_questions_df_uses_configured_engine(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'takken.db'}", future=True)
    app.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            insert(app.questions_table),
            {
                "id": "q-2023-1",
                "year": 2023,
                "q_no": 1,
                "category": "民法",
                "topic": "概要",
                "question": "宅建業法の定義は?",
                "choice1": "宅建士",
                "choice2": "宅地",
                "choice3": "建物",
                "choice4": "その他",
                "correct": 1,
                "explanation": "詳しい説明",
                "difficulty": 3,
                "tags": "重要",
            },
        )

    monkeypatch.setattr(app, "get_engine", lambda: engine)
    app.load_questions_df.clear()
    df = app.load_questions_df()
    assert not df.empty
    assert df.iloc[0]["question"] == "宅建業法の定義は?"
