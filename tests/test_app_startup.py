"""Streamlit app startup behaviour tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


def test_main_navigation_flow(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'takken.db'}", future=True)
    app.metadata.create_all(engine)

    monkeypatch.setattr(app, "get_engine", lambda: engine)
    monkeypatch.setattr(app, "apply_user_preferences", lambda: None)

    fake_df = pd.DataFrame(
        {
            "id": ["q-2023-1"],
            "year": [2023],
            "q_no": [1],
            "category": ["民法"],
            "question": ["宅建業法の定義は?"],
        }
    )
    monkeypatch.setattr(app, "load_questions_df", lambda: fake_df)

    render_calls: list[str] = []

    def _record(name: str):
        def _inner(*_args, **_kwargs) -> None:
            render_calls.append(name)

        return _inner

    class FakeDB:
        def __init__(self, _engine) -> None:
            self.engine = _engine
            self.initialized = False

        def initialize_from_csv(self) -> None:
            self.initialized = True

    created_dbs: list[FakeDB] = []

    def fake_db_manager(engine_arg):
        db = FakeDB(engine_arg)
        created_dbs.append(db)
        return db

    monkeypatch.setattr(app, "DBManager", fake_db_manager)
    monkeypatch.setattr(app, "render_learning", _record("学習"))
    monkeypatch.setattr(app, "render_mock_exam", _record("模試"))
    monkeypatch.setattr(app, "render_stats", _record("統計"))
    monkeypatch.setattr(app, "render_settings", _record("設定"))

    def run_app():
        import app as app_module

        app_module.main()

    app_test = AppTest.from_function(run_app)
    app_test.run()

    assert created_dbs and all(db.initialized for db in created_dbs)
    assert render_calls[-1] == "学習"

    menu = app_test.sidebar.radio[0]
    menu.set_value("設定")
    app_test.run()

    assert render_calls[-1] == "設定"
    assert app_test.session_state["nav"] == "設定"
