import datetime as dt
import hashlib
import html as html_module
import io
import json
import logging
import random
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import (TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set,
                    Tuple)
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import (JSON, Column, Date, DateTime, Float, Integer, MetaData,
                        String, Table, UniqueConstraint, create_engine, func,
                        select, tuple_)
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql import insert as sa_insert, text, update

from law_updates import (DEFAULT_LAW_SOURCES, LawRevisionAnalyzer,
                         LawUpdateResult, LawUpdateSyncService)
from integrations import (GoogleCalendarClient, GoogleCalendarConfig,
                          IntegrationConfigError, IntegrationError,
                          NotionClient, NotionConfig, OAuthCredentials)

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "takken.db"
UPLOAD_DIR = DATA_DIR / "uploads"
REJECT_DIR = DATA_DIR / "rejects"
OFFLINE_EXPORT_DIR = DATA_DIR / "offline_exports"
MAPPING_KIND_QUESTIONS = "questions"
MAPPING_KIND_ANSWERS = "answers"
SCHEMA_GUIDE_PATH = Path("docs") / "data_schema.md"

MAPPING_KIND_LABELS = {
    MAPPING_KIND_QUESTIONS: "設問データ",
    MAPPING_KIND_ANSWERS: "正答データ",
}
DEFAULT_CATEGORY_MAP = {
    "宅建業法": "宅建業法",
    "業法": "宅建業法",
    "権利関係": "権利関係",
    "民法": "権利関係",
    "法令上の制限": "法令上の制限",
    "制限": "法令上の制限",
    "税・その他": "税・その他",
    "税その他": "税・その他",
}
CATEGORY_CHOICES = ["宅建業法", "権利関係", "法令上の制限", "税・その他"]
CATEGORY_BADGE_STYLE = {
    "宅建業法": {"icon": "🏢", "class": "takken-tag--category-business"},
    "権利関係": {"icon": "⚖️", "class": "takken-tag--category-rights"},
    "法令上の制限": {"icon": "🛣️", "class": "takken-tag--category-regulation"},
    "税・その他": {"icon": "💰", "class": "takken-tag--category-tax"},
}
CATEGORY_BADGE_DEFAULT = {"icon": "📘", "class": "takken-tag--category-default"}
DIFFICULTY_DEFAULT = 3
LAW_REFERENCE_BASE_URL = "https://elaws.e-gov.go.jp/search?q={query}"

LAW_NAME_SUFFIXES = (
    "法",
    "条例",
    "規則",
    "規程",
    "基準",
    "告示",
    "指針",
    "要綱",
    "判例",
)
LAW_NAME_KEYWORDS = {
    "宅建業法",
    "宅地建物取引業法",
    "民法",
    "借地借家法",
    "不動産登記法",
    "区分所有法",
    "都市計画法",
    "建築基準法",
    "農地法",
    "国土利用計画法",
    "住宅瑕疵担保履行法",
    "景品表示法",
}
LAW_NAME_PATTERN = re.compile(
    r"(?P<law>[一-龠A-Za-z0-9・（）()]+?(?:法|条例|規則|規程|基準|告示|指針|要綱|判例))"
)
OUTLINE_CACHE_KEY = "_outline_insights_cache"

FONT_SIZE_SCALE = {
    "やや小さい": 0.95,
    "標準": 1.0,
    "やや大きい": 1.1,
    "大きい": 1.2,
}


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

SUBJECT_PRESETS = {
    "バランスよく10問": {
        "categories": CATEGORY_CHOICES,
        "difficulty": (1, 5),
        "review_only": False,
        "topics": [],
    },
    "民法・権利関係を集中演習": {
        "categories": ["権利関係"],
        "difficulty": (2, 5),
        "review_only": False,
        "topics": [],
    },
    "弱点復習に集中": {
        "categories": CATEGORY_CHOICES,
        "difficulty": (1, 4),
        "review_only": True,
        "topics": [],
    },
}

metadata = MetaData()

questions_table = Table(
    "questions",
    metadata,
    Column("id", String, primary_key=True),
    Column("year", Integer, nullable=False),
    Column("q_no", Integer, nullable=False),
    Column("category", String, nullable=False),
    Column("topic", String),
    Column("question", String, nullable=False),
    Column("choice1", String, nullable=False),
    Column("choice2", String, nullable=False),
    Column("choice3", String, nullable=False),
    Column("choice4", String, nullable=False),
    Column("correct", Integer),
    Column("explanation", String),
    Column("difficulty", Integer, default=DIFFICULTY_DEFAULT),
    Column("tags", String),
    Column("dup_note", String),
    UniqueConstraint("year", "q_no", name="uq_questions_year_qno"),
)

predicted_questions_table = Table(
    "predicted_questions",
    metadata,
    Column("id", String, primary_key=True),
    Column("label", String),
    Column("year", String),
    Column("q_no", String),
    Column("category", String),
    Column("topic", String),
    Column("source", String),
    Column("question", String, nullable=False),
    Column("choice1", String, nullable=False),
    Column("choice2", String, nullable=False),
    Column("choice3", String, nullable=False),
    Column("choice4", String, nullable=False),
    Column("correct", Integer),
    Column("explanation", String),
    Column("difficulty", Integer, default=DIFFICULTY_DEFAULT),
    Column("tags", String),
    Column("auto_summary", String),
    Column("auto_cloze", String),
    Column("review_status", String, default="pending"),
    Column("reviewed_at", DateTime),
    Column("generated_from", String),
    Column("fetched_at", DateTime),
    Column("created_at", DateTime, server_default=func.now()),
)

law_revision_questions_table = Table(
    "law_revision_questions",
    metadata,
    Column("id", String, primary_key=True),
    Column("label", String),
    Column("law_name", String),
    Column("revision_year", Integer),
    Column("effective_date", String),
    Column("category", String),
    Column("topic", String),
    Column("source", String),
    Column("question", String, nullable=False),
    Column("choice1", String, nullable=False),
    Column("choice2", String, nullable=False),
    Column("choice3", String, nullable=False),
    Column("choice4", String, nullable=False),
    Column("correct", Integer),
    Column("explanation", String),
    Column("difficulty", Integer, default=DIFFICULTY_DEFAULT),
    Column("tags", String),
    Column("created_at", DateTime, server_default=func.now()),
)

law_revision_sync_logs_table = Table(
    "law_revision_sync_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", String, nullable=False),
    Column("fetched_at", DateTime),
    Column("status", String, nullable=False),
    Column("message", String),
    Column("revisions_detected", Integer, default=0),
    Column("questions_generated", Integer, default=0),
    Column("created_at", DateTime, server_default=func.now()),
)

attempts_table = Table(
    "attempts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("question_id", String, nullable=False),
    Column("selected", Integer),
    Column("is_correct", Integer),
    Column("seconds", Integer),
    Column("mode", String),
    Column("exam_id", Integer),
    Column("confidence", Integer),
    Column("grade", Integer),
    Column("created_at", DateTime, server_default=func.now()),
)

outline_notes_table = Table(
    "outline_notes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("question_id", String, nullable=False),
    Column("summary", String, nullable=False),
    Column("law_references", JSON),
    Column("question_label", String),
    Column("tags", String),
    Column("created_at", DateTime, server_default=func.now()),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now()),
    UniqueConstraint("question_id", "summary", name="uq_outline_notes_question_summary"),
)

exams_table = Table(
    "exams",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("year_mode", String),
    Column("score", Integer),
)

srs_table = Table(
    "questions_srs",
    metadata,
    Column("question_id", String, primary_key=True),
    Column("repetition", Integer, default=0),
    Column("interval", Integer, default=1),
    Column("ease", Float, default=2.5),
    Column("due_date", Date),
    Column("last_grade", Integer),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now()),
)

import_logs_table = Table(
    "import_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("files", Integer),
    Column("inserted", Integer),
    Column("updated", Integer),
    Column("rejected", Integer),
    Column("conflicts", Integer),
    Column("seconds", Float),
    Column("policy", String),
)

mapping_profiles_table = Table(
    "mapping_profiles",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("kind", String, nullable=False),
    Column("mapping_json", JSON, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
)

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    REJECT_DIR.mkdir(exist_ok=True)
    OFFLINE_EXPORT_DIR.mkdir(exist_ok=True)


def inject_style(css: str, style_id: str) -> None:
    sanitized_css = css.strip()
    if not sanitized_css:
        return
    css_payload = json.dumps(sanitized_css)
    style_id_payload = json.dumps(style_id)
    script = Template(
        """
        <script>
        (function() {
            const css = $css;
            const styleId = $style_id;
            const doc = window.parent ? window.parent.document : document;
            let styleTag = doc.getElementById(styleId);
            if (!styleTag) {
                styleTag = doc.createElement('style');
                styleTag.id = styleId;
                doc.head.appendChild(styleTag);
            }
            styleTag.innerHTML = css;
        })();
        </script>
        """
    ).substitute(css=css_payload, style_id=style_id_payload)
    st_html(script, height=0)


def ensure_schema_migrations(engine: Engine) -> None:
    inspector = inspect(engine)
    with engine.begin() as conn:
        existing_tables = set(inspector.get_table_names())
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                table.create(conn)
                existing_tables.add(table.name)

        if attempts_table.name not in existing_tables:
            attempts_table.create(conn)
            existing_tables.add(attempts_table.name)

        if outline_notes_table.name not in existing_tables:
            outline_notes_table.create(conn)
            existing_tables.add(outline_notes_table.name)

        conn_inspector = inspect(conn)
        attempt_columns = {col["name"] for col in conn_inspector.get_columns("attempts")}

        if "confidence" not in attempt_columns:
            conn.execute(text("ALTER TABLE attempts ADD COLUMN confidence INTEGER"))
        if "grade" not in attempt_columns:
            conn.execute(text("ALTER TABLE attempts ADD COLUMN grade INTEGER"))

        if "outline_notes" in existing_tables:
            outline_columns = {
                col["name"] for col in conn_inspector.get_columns("outline_notes")
            }
            outline_schema_updates = {
                "question_label": "TEXT",
                "tags": "TEXT",
            }
            for column_name, sql_type in outline_schema_updates.items():
                if column_name not in outline_columns:
                    conn.execute(
                        text(
                            f"ALTER TABLE outline_notes ADD COLUMN {column_name} {sql_type}"
                        )
                    )

            if "updated_at" not in outline_columns:
                conn.execute(
                    text("ALTER TABLE outline_notes ADD COLUMN updated_at DATETIME")
                )
                conn.execute(
                    text(
                        "UPDATE outline_notes SET updated_at = created_at WHERE updated_at IS NULL"
                    )
                )

        if "law_revision_questions" in existing_tables:
            lr_columns = {
                col["name"] for col in conn_inspector.get_columns("law_revision_questions")
            }
            schema_updates = {
                "auto_summary": "TEXT",
                "auto_cloze": "TEXT",
                "review_status": "TEXT",
                "reviewed_at": "TEXT",
                "generated_from": "TEXT",
                "fetched_at": "TEXT",
            }
            for column_name, sql_type in schema_updates.items():
                if column_name not in lr_columns:
                    conn.execute(
                        text(
                            f"ALTER TABLE law_revision_questions ADD COLUMN {column_name} {sql_type}"
                        )
                    )

        if "law_revision_sync_logs" not in existing_tables:
            law_revision_sync_logs_table.create(conn)


def build_theme_css(theme: str) -> str:
    if theme == "ダーク":
        return """
:root {
    color-scheme: dark;
    --takken-color-bg: #1c1c1e;
    --takken-color-bg-sidebar: rgba(28, 28, 30, 0.92);
    --takken-color-surface: rgba(44, 44, 46, 0.94);
    --takken-color-surface-elevated: rgba(58, 58, 60, 0.94);
    --takken-color-border: rgba(255, 255, 255, 0.08);
    --takken-color-text: #f5f5f7;
    --takken-color-text-secondary: rgba(235, 235, 245, 0.6);
    --takken-color-text-tertiary: rgba(235, 235, 245, 0.38);
    --takken-color-accent: #0a84ff;
    --takken-color-accent-soft: rgba(10, 132, 255, 0.18);
    --takken-color-positive: #30d158;
    --takken-color-negative: #ff453a;
    --takken-focus-ring: 0 0 0 3px rgba(10, 132, 255, 0.45);
    --takken-shadow-strong: 0 24px 60px rgba(0, 0, 0, 0.55);
    --takken-radius-large: 18px;
}
body {
    background-color: var(--takken-color-bg);
    color: var(--takken-color-text);
    font-family: "SF Pro JP", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
    font-feature-settings: "palt";
}
[data-testid="stAppViewContainer"] {
    background: var(--takken-color-bg);
    color: var(--takken-color-text);
}
[data-testid="stAppViewContainer"] a {
    color: var(--takken-color-accent);
}
[data-testid="stSidebar"] {
    background: var(--takken-color-bg-sidebar);
    color: var(--takken-color-text);
    border-right: 1px solid var(--takken-color-border);
    backdrop-filter: blur(24px);
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {
    color: var(--takken-color-text-secondary);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 999px;
    color: var(--takken-color-text-tertiary);
    border: 1px solid transparent;
    transition: color 0.2s ease, background 0.2s ease, border-color 0.2s ease;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--takken-color-accent-soft);
    color: var(--takken-color-text);
    border-color: rgba(10, 132, 255, 0.4);
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--takken-color-text);
    border-color: rgba(10, 132, 255, 0.25);
}
.stMetric,
.stAlert,
.stDataFrame,
.stTable,
.stExpander,
[data-testid="stMarkdownContainer"] {
    background: var(--takken-color-surface);
    border-radius: var(--takken-radius-large);
    border: 1px solid var(--takken-color-border);
    box-shadow: var(--takken-shadow-strong);
}
.stExpander > div:first-child {
    background: transparent;
    color: var(--takken-color-text);
}
.element-container > .stMetric,
.element-container > .stAlert {
    padding: 1.1rem 1.2rem;
}
.home-dropzone-card {
    background: rgba(10, 132, 255, 0.12);
    border: 1px solid rgba(10, 132, 255, 0.48);
    box-shadow: 0 22px 60px rgba(10, 132, 255, 0.25);
}
.home-data-card {
    background: var(--takken-color-surface);
    border: 1px solid var(--takken-color-border);
    box-shadow: var(--takken-shadow-strong);
}
.takken-choice-button button,
.stButton > button {
    background: linear-gradient(135deg, #0a84ff 0%, #64d2ff 100%);
    color: #ffffff;
    border: none;
    border-radius: 14px;
    font-weight: 600;
    box-shadow: 0 16px 40px rgba(10, 132, 255, 0.28);
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.3s ease;
}
.takken-choice-button button:hover,
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 20px 48px rgba(10, 132, 255, 0.38);
}
.takken-choice-button button:focus,
.stButton > button:focus {
    outline: none;
    box-shadow: var(--takken-focus-ring);
}
.takken-choice-button.is-correct button {
    background: linear-gradient(135deg, #30d158 0%, #63e6b3 100%);
    box-shadow: 0 18px 46px rgba(48, 209, 88, 0.32);
}
.takken-choice-button.is-incorrect button,
.stButton > button.is-error {
    background: linear-gradient(135deg, #ff453a 0%, #ff9f0a 100%);
    box-shadow: 0 18px 46px rgba(255, 69, 58, 0.3);
}
input[type="text"],
input[type="number"],
input[type="password"],
input[type="email"],
textarea,
select {
    background: var(--takken-color-surface-elevated);
    border: 1px solid var(--takken-color-border);
    color: var(--takken-color-text);
    border-radius: 14px;
    padding: 0.55rem 0.9rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
input[type="text"]::placeholder,
input[type="number"]::placeholder,
input[type="password"]::placeholder,
input[type="email"]::placeholder,
textarea::placeholder {
    color: var(--takken-color-text-tertiary);
}
input[type="text"]:focus,
input[type="number"]:focus,
input[type="password"]:focus,
input[type="email"]:focus,
textarea:focus,
select:focus {
    outline: none;
    border-color: var(--takken-color-accent);
    box-shadow: var(--takken-focus-ring);
}
label, legend {
    color: var(--takken-color-text-secondary);
}
code, pre {
    background: rgba(44, 44, 46, 0.9);
    color: var(--takken-color-text);
    border-radius: 12px;
    border: 1px solid var(--takken-color-border);
}
[data-testid="stDataFrame"] table,
[data-testid="stTable"] table {
    background: transparent;
    color: var(--takken-color-text);
}
[data-testid="stDataFrame"] table thead th,
[data-testid="stTable"] table thead th {
    background: rgba(255, 255, 255, 0.04);
    color: var(--takken-color-text-secondary);
    font-weight: 600;
}
[data-testid="stDataFrame"] table tbody td,
[data-testid="stTable"] table tbody td {
    border-color: var(--takken-color-border);
}
.stProgress > div > div {
    background: var(--takken-color-accent);
}
.stProgress > div {
    background: rgba(10, 132, 255, 0.12);
}
.stSlider > div > div > div[data-baseweb="slider"] > div:first-child {
    background: rgba(255, 255, 255, 0.08);
}
.stSlider > div > div > div[data-baseweb="slider"] > div:nth-child(2) {
    background: var(--takken-color-accent);
}
.stSlider > div > div > div[data-baseweb="slider"] > div:nth-child(4) {
    color: var(--takken-color-text-secondary);
}
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
.stMarkdown h5 {
    color: var(--takken-color-text);
    letter-spacing: 0.02em;
}
.stMarkdown p,
.stMarkdown li {
    color: var(--takken-color-text-secondary);
}
hr {
    border-color: rgba(255, 255, 255, 0.08);
}
"""
    if theme == "セピア":
        return """
[data-testid="stAppViewContainer"] {
    background-color: #f3e9d2;
    color: #4a3f35;
}
[data-testid="stSidebar"] {
    background-color: #f9f1e0;
}
.stMetric, .stAlert {
    background-color: rgba(74, 63, 53, 0.06);
}
"""
    return """
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc;
    color: #1f2933;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
}
"""


def apply_user_preferences() -> None:
    settings = st.session_state.get("settings", {})
    theme = settings.get("theme", "セピア")
    font_label = settings.get("font_size", "標準")
    scale = FONT_SIZE_SCALE.get(font_label, 1.0)
    base_css = f"""
:root {{
    --takken-font-scale: {scale};
}}
[data-testid="stAppViewContainer"] * {{
    font-size: calc(1rem * var(--takken-font-scale));
}}
body {{
    font-family: "SF Pro JP", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
}}
.takken-search-suggestions .stButton>button {{
    width: 100%;
    margin-bottom: 0.35rem;
}}
.takken-search-suggestions .stButton>button:hover {{
    border-color: #6366f1;
}}
"""
    theme_css = build_theme_css(theme)
    inject_style(base_css + theme_css, "takken-theme-styles")


def safe_rerun() -> None:
    rerun = getattr(st, "rerun", None)
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()
    elif callable(experimental_rerun):
        experimental_rerun()



def with_rerun(callback: Callable[..., None], *args, **kwargs) -> Callable[[], None]:
    def _inner() -> None:
        callback(*args, **kwargs)

    return _inner


def handle_nav_change() -> None:
    st.session_state["nav"] = st.session_state.get("_nav_widget", "ホーム")


def navigate_to(section: str) -> None:
    st.session_state["nav"] = section
    st.session_state["_nav_widget"] = section
    safe_rerun()


def render_specialized_header(parent_label: str, current_label: str, key_suffix: str) -> None:
    breadcrumb = f"{parent_label} ＞ {current_label}"
    cols = st.columns([5, 1])
    with cols[0]:
        st.markdown(f"**{breadcrumb}**")
    with cols[1]:
        st.button(
            "戻る",
            key=f"back_{key_suffix}",
            use_container_width=True,
            help=f"{parent_label}に戻ります。",
            on_click=with_rerun(navigate_to, parent_label),
        )


QUESTION_TEMPLATE_COLUMNS = [
    "year",
    "q_no",
    "category",
    "topic",
    "question",
    "choice1",
    "choice2",
    "choice3",
    "choice4",
    "explanation",
    "difficulty",
    "tags",
]

ANSWER_TEMPLATE_COLUMNS = [
    "year",
    "q_no",
    "correct_number",
    "correct_label",
    "correct_text",
    "explanation",
    "difficulty",
    "tags",
]

PREDICTED_TEMPLATE_COLUMNS = [
    "label",
    "category",
    "topic",
    "source",
    "question",
    "choice1",
    "choice2",
    "choice3",
    "choice4",
    "correct",
    "explanation",
    "year",
    "q_no",
    "difficulty",
    "tags",
]

LAW_REVISION_TEMPLATE_COLUMNS = [
    "label",
    "law_name",
    "revision_year",
    "effective_date",
    "category",
    "topic",
    "source",
    "question",
    "choice1",
    "choice2",
    "choice3",
    "choice4",
    "correct",
    "explanation",
    "difficulty",
    "tags",
    "auto_summary",
    "auto_cloze",
    "review_status",
    "reviewed_at",
    "generated_from",
    "fetched_at",
]


@st.cache_data(show_spinner=False)
def get_template_archive() -> bytes:
    question_template = pd.DataFrame(
        [
            {
                "year": dt.datetime.now().year,
                "q_no": 1,
                "category": CATEGORY_CHOICES[0],
                "topic": "小分類の例",
                "question": "ここに問題文を入力してください。",
                "choice1": "選択肢1",
                "choice2": "選択肢2",
                "choice3": "選択肢3",
                "choice4": "選択肢4",
                "explanation": "解説を入力できます。",
                "difficulty": DIFFICULTY_DEFAULT,
                "tags": "タグ1;タグ2",
            }
        ],
        columns=QUESTION_TEMPLATE_COLUMNS,
    )
    answer_template = pd.DataFrame(
        [
            {
                "year": dt.datetime.now().year,
                "q_no": 1,
                "correct_number": 1,
                "correct_label": "A",
                "correct_text": "選択肢1",
                "explanation": "正答の解説を入力できます。",
                "difficulty": DIFFICULTY_DEFAULT,
                "tags": "タグ1;タグ2",
            }
        ],
        columns=ANSWER_TEMPLATE_COLUMNS,
    )
    predicted_template = pd.DataFrame(
        [
            {
                "label": "予想問題001",
                "category": CATEGORY_CHOICES[0],
                "topic": "直前対策",
                "source": "講師予想",
                "question": "ここに予想問題の本文を入力してください。",
                "choice1": "選択肢1",
                "choice2": "選択肢2",
                "choice3": "選択肢3",
                "choice4": "選択肢4",
                "correct": 1,
                "explanation": "根拠となる条文や理由を記載できます。",
                "year": dt.datetime.now().year + 1,
                "q_no": "予想1",
                "difficulty": DIFFICULTY_DEFAULT,
                "tags": "予想;重要論点",
            }
        ],
        columns=PREDICTED_TEMPLATE_COLUMNS,
    )
    law_revision_template = pd.DataFrame(
        [
            {
                "label": "2024年改正ポイント01",
                "law_name": "宅建業法",
                "revision_year": dt.datetime.now().year,
                "effective_date": f"{dt.datetime.now().year}-04-01",
                "category": CATEGORY_CHOICES[0],
                "topic": "重要事項説明",
                "source": "国交省告示",
                "question": "最近の法改正内容に関する確認問題を入力してください。",
                "choice1": "改正により重要事項説明書への記載が義務化された。",
                "choice2": "改正前後で手続は変わらない。",
                "choice3": "改正で免除規定が新設された。",
                "choice4": "法改正とは無関係の記載である。",
                "correct": 1,
                "explanation": "改正ポイントの概要や根拠条文を記載します。",
                "difficulty": DIFFICULTY_DEFAULT,
                "tags": "法改正;直前対策",
            }
        ],
        columns=LAW_REVISION_TEMPLATE_COLUMNS,
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("questions_template.csv", question_template.to_csv(index=False))
        zf.writestr("answers_template.csv", answer_template.to_csv(index=False))
        zf.writestr("predicted_template.csv", predicted_template.to_csv(index=False))
        zf.writestr("law_revision_template.csv", law_revision_template.to_csv(index=False))
        q_excel = io.BytesIO()
        with pd.ExcelWriter(q_excel, engine="openpyxl") as writer:
            question_template.to_excel(writer, index=False, sheet_name="questions")
        zf.writestr("questions_template.xlsx", q_excel.getvalue())
        a_excel = io.BytesIO()
        with pd.ExcelWriter(a_excel, engine="openpyxl") as writer:
            answer_template.to_excel(writer, index=False, sheet_name="answers")
        zf.writestr("answers_template.xlsx", a_excel.getvalue())
        p_excel = io.BytesIO()
        with pd.ExcelWriter(p_excel, engine="openpyxl") as writer:
            predicted_template.to_excel(writer, index=False, sheet_name="predicted")
        zf.writestr("predicted_template.xlsx", p_excel.getvalue())
        lr_excel = io.BytesIO()
        with pd.ExcelWriter(lr_excel, engine="openpyxl") as writer:
            law_revision_template.to_excel(writer, index=False, sheet_name="law_revision")
        zf.writestr("law_revision_template.xlsx", lr_excel.getvalue())
        if SCHEMA_GUIDE_PATH.exists():
            zf.writestr(
                "data_schema.md",
                SCHEMA_GUIDE_PATH.read_text(encoding="utf-8"),
            )
        description = (
            "questions_template は設問データ、answers_template は正答データ、predicted_template は予想問題データ、"
            "law_revision_template は法改正予想問題データのサンプルです。\n"
            "不要な行は削除し、ご自身のデータを入力してからアップロードしてください。\n"
            "各列の詳細仕様は data_schema.md を参照してください。"
        )
        zf.writestr("README.txt", description)
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_resource
def get_engine() -> Engine:
    ensure_directories()
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    metadata.create_all(engine)
    ensure_schema_migrations(engine)
    return engine


class DBManager:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def load_dataframe(self, table: Table) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(select(table), conn)
        except OperationalError:
            column_names = [column.name for column in table.columns]
            return pd.DataFrame(columns=column_names)
        return df

    def load_predicted_questions(self) -> pd.DataFrame:
        return self.load_dataframe(predicted_questions_table)

    def load_law_revision_questions(self) -> pd.DataFrame:
        return self.load_dataframe(law_revision_questions_table)

    def load_law_revision_sync_logs(self, limit: int = 20) -> pd.DataFrame:
        with self.engine.connect() as conn:
            stmt = select(law_revision_sync_logs_table).order_by(
                law_revision_sync_logs_table.c.created_at.desc()
            )
            if limit:
                stmt = stmt.limit(limit)
            df = pd.read_sql(stmt, conn)
        return df

    def upsert_questions(self, df: pd.DataFrame) -> Tuple[int, int]:
        return self.bulk_upsert_questions(df)

    def bulk_upsert_questions(
        self,
        df: pd.DataFrame,
        batch_size: int = 200,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[int, int]:
        records = df.to_dict(orient="records")
        total = len(records)
        if total == 0:
            if on_progress is not None:
                try:
                    on_progress(0, 0)
                except Exception:
                    logger.exception("Progress callback failed during empty bulk upsert")
            return 0, 0

        ids = [rec["id"] for rec in records if "id" in rec]
        year_qno_pairs = {
            (rec["year"], rec["q_no"]) for rec in records if "year" in rec and "q_no" in rec
        }

        inserted = 0
        updated = 0
        processed = 0

        with self.engine.begin() as conn:
            if ids:
                existing_ids: Set[str] = set(
                    conn.execute(
                        select(questions_table.c.id).where(questions_table.c.id.in_(ids))
                    ).scalars()
                )
            else:
                existing_ids = set()

            if year_qno_pairs:
                existing_pairs = {
                    (row.year, row.q_no): row.id
                    for row in conn.execute(
                        select(
                            questions_table.c.year,
                            questions_table.c.q_no,
                            questions_table.c.id,
                        ).where(
                            tuple_(questions_table.c.year, questions_table.c.q_no).in_(
                                list(year_qno_pairs)
                            )
                        )
                    )
                }
            else:
                existing_pairs = {}

            for start in range(0, total, max(batch_size, 1)):
                batch = records[start : start + max(batch_size, 1)]
                for rec in batch:
                    rec_id = rec.get("id")
                    year_qno = (rec.get("year"), rec.get("q_no"))
                    update_values = {k: v for k, v in rec.items() if k != "id"}

                    if rec_id in existing_ids:
                        conn.execute(
                            update(questions_table)
                            .where(questions_table.c.id == rec_id)
                            .values(**update_values)
                        )
                        updated += 1
                    elif year_qno in existing_pairs:
                        existing_id = existing_pairs[year_qno]
                        conn.execute(
                            update(questions_table)
                            .where(questions_table.c.id == existing_id)
                            .values(**update_values)
                        )
                        updated += 1
                    else:
                        conn.execute(sa_insert(questions_table).values(**rec))
                        inserted += 1
                        if rec_id:
                            existing_ids.add(rec_id)
                        if None not in year_qno:
                            existing_pairs[year_qno] = rec_id

                processed += len(batch)
                if on_progress is not None:
                    try:
                        on_progress(processed, total)
                    except Exception:
                        logger.exception("Progress callback failed during bulk upsert")

        return inserted, updated

    def upsert_predicted_questions(self, df: pd.DataFrame) -> Tuple[int, int]:
        records = df.to_dict(orient="records")
        ids = [rec["id"] for rec in records if "id" in rec]
        inserted = 0
        updated = 0
        with self.engine.begin() as conn:
            existing_ids: Set[str] = set()
            if ids:
                existing_ids = set(
                    conn.execute(
                        select(predicted_questions_table.c.id).where(
                            predicted_questions_table.c.id.in_(ids)
                        )
                    ).scalars()
                )
            for rec in records:
                rec_id = rec.get("id")
                if rec_id in existing_ids:
                    conn.execute(
                        update(predicted_questions_table)
                        .where(predicted_questions_table.c.id == rec_id)
                        .values(**{k: v for k, v in rec.items() if k != "id"})
                    )
                    updated += 1
                else:
                    conn.execute(sa_insert(predicted_questions_table).values(**rec))
                    inserted += 1
                    if rec_id:
                        existing_ids.add(rec_id)
        return inserted, updated

    def upsert_law_revision_questions(self, df: pd.DataFrame) -> Tuple[int, int]:
        records = df.to_dict(orient="records")
        ids = [rec["id"] for rec in records if "id" in rec]
        inserted = 0
        updated = 0
        with self.engine.begin() as conn:
            existing_ids: Set[str] = set()
            if ids:
                existing_ids = set(
                    conn.execute(
                        select(law_revision_questions_table.c.id).where(
                            law_revision_questions_table.c.id.in_(ids)
                        )
                    ).scalars()
                )
            for rec in records:
                rec_id = rec.get("id")
                payload = {k: v for k, v in rec.items() if k != "id"}
                if rec_id in existing_ids:
                    conn.execute(
                        update(law_revision_questions_table)
                        .where(law_revision_questions_table.c.id == rec_id)
                        .values(**payload)
                    )
                    updated += 1
                else:
                    conn.execute(sa_insert(law_revision_questions_table).values(**rec))
                    inserted += 1
                    if rec_id:
                        existing_ids.add(rec_id)
        return inserted, updated

    def update_law_revision_review_status(
        self,
        question_ids: Sequence[str],
        status: str,
        reviewer: Optional[str] = None,
    ) -> None:
        if not question_ids:
            return
        now = dt.datetime.utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                update(law_revision_questions_table)
                .where(law_revision_questions_table.c.id.in_(list(question_ids)))
                .values(
                    review_status=status,
                    reviewed_at=now,
                    tags=func.trim(
                        func.replace(
                            func.coalesce(law_revision_questions_table.c.tags, ""),
                            "要レビュー",
                            "",
                        )
                    ),
                )
            )

    def record_law_revision_sync(self, result: LawUpdateResult) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sa_insert(law_revision_sync_logs_table).values(
                    source=result.source,
                    fetched_at=result.fetched_at,
                    status=result.status,
                    message=result.message,
                    revisions_detected=result.revisions_detected,
                    questions_generated=result.questions_generated,
                )
            )

    def fetch_question(self, question_id: str) -> Optional[pd.Series]:
        with self.engine.connect() as conn:
            df = pd.read_sql(select(questions_table).where(questions_table.c.id == question_id), conn)
        if df.empty:
            return None
        return df.iloc[0]

    def record_attempt(
        self,
        question_id: str,
        selected: Optional[int],
        is_correct: bool,
        seconds: int,
        mode: str,
        exam_id: Optional[int] = None,
        confidence: Optional[int] = None,
        grade: Optional[int] = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sa_insert(attempts_table).values(
                    question_id=question_id,
                    selected=selected,
                    is_correct=int(is_correct),
                    seconds=seconds,
                    mode=mode,
                    exam_id=exam_id,
                    confidence=confidence,
                    grade=grade,
                )
            )

    def bulk_insert_attempts(self, attempts: Sequence[Dict[str, object]]) -> int:
        if not attempts:
            return 0
        payloads = []
        for item in attempts:
            payload = {
                "question_id": item.get("question_id"),
                "selected": item.get("selected"),
                "is_correct": item.get("is_correct"),
                "seconds": item.get("seconds"),
                "mode": item.get("mode"),
                "exam_id": item.get("exam_id"),
                "confidence": item.get("confidence"),
                "grade": item.get("grade"),
            }
            created_at = item.get("created_at")
            if created_at is not None:
                payload["created_at"] = created_at
            payloads.append(payload)
        with self.engine.begin() as conn:
            conn.execute(sa_insert(attempts_table), payloads)
        return len(payloads)

    def fetch_srs(self, question_id: str) -> Optional[pd.Series]:
        with self.engine.connect() as conn:
            df = pd.read_sql(
                select(srs_table).where(srs_table.c.question_id == question_id),
                conn,
            )
        if df.empty:
            return None
        return df.iloc[0]

    def log_exam_result(self, payload: Dict[str, object]) -> Optional[int]:
        with self.engine.begin() as conn:
            result = conn.execute(sa_insert(exams_table).values(**payload))
            inserted = result.inserted_primary_key
        if inserted:
            return inserted[0]
        return None

    def update_question_fields(
        self,
        question_id: str,
        fields: Dict[str, Optional[str]],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(questions_table)
                .where(questions_table.c.id == question_id)
                .values(**fields)
            )

    def get_attempt_stats(self) -> pd.DataFrame:
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT
                        a.question_id,
                        q.category,
                        q.topic,
                        q.year,
                        a.is_correct,
                        a.created_at,
                        a.seconds,
                        a.selected,
                        a.mode,
                        a.confidence,
                        a.grade
                    FROM attempts a
                    JOIN questions q ON q.id = a.question_id
                    """
                ),
                conn,
            )
        return df

    def get_due_srs(self, upcoming_days: int = 1) -> pd.DataFrame:
        today = dt.date.today()
        upcoming_days = max(int(upcoming_days or 0), 0)
        upcoming_limit = today + dt.timedelta(days=upcoming_days)
        with self.engine.connect() as conn:
            df = pd.read_sql(
                select(
                    srs_table,
                    questions_table.c.question,
                    questions_table.c.category,
                )
                .where(srs_table.c.question_id == questions_table.c.id)
                .where(
                    (srs_table.c.due_date <= upcoming_limit)
                    | (srs_table.c.due_date.is_(None))
                ),
                conn,
            )
        if df.empty:
            return df

        df["due_date"] = pd.to_datetime(df["due_date"])

        due_in_days: List[Optional[int]] = []
        due_status: List[str] = []
        for value in df["due_date"]:
            if pd.isna(value):
                due_in_days.append(None)
                due_status.append("unscheduled")
                continue
            if isinstance(value, pd.Timestamp):
                date_value = value.date()
            else:
                date_value = value
            delta = (date_value - today).days
            due_in_days.append(delta)
            if delta < 0:
                due_status.append("overdue")
            elif delta == 0:
                due_status.append("due_today")
            elif delta == 1:
                due_status.append("due_tomorrow")
            elif delta <= upcoming_days:
                due_status.append("upcoming")
            else:
                due_status.append("scheduled")

        df["due_in_days"] = due_in_days
        df["due_status"] = due_status
        df["due_date"] = df["due_date"].dt.date
        return df

    def fetch_outline_notes(self, question_id: Optional[str] = None) -> pd.DataFrame:
        stmt = select(outline_notes_table)
        if question_id:
            stmt = stmt.where(outline_notes_table.c.question_id == question_id)
        stmt = stmt.order_by(outline_notes_table.c.updated_at.desc())

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(stmt, conn)
        except OperationalError:
            column_names = [column.name for column in outline_notes_table.columns]
            return pd.DataFrame(columns=column_names)
        return df

    def save_outline_note(
        self,
        question_id: str,
        summary: str,
        references: Sequence[Dict[str, str]],
        question_label: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> None:
        references_payload = [
            {"label": ref.get("label"), "url": ref.get("url")}
            for ref in references
            if isinstance(ref, dict)
        ]
        with self.engine.begin() as conn:
            stmt = sqlite_insert(outline_notes_table).values(
                question_id=question_id,
                summary=summary,
                law_references=references_payload,
                question_label=question_label,
                tags=tags,
            )
            do_update = stmt.on_conflict_do_update(
                index_elements=[
                    outline_notes_table.c.question_id,
                    outline_notes_table.c.summary,
                ],
                set_={
                    "law_references": stmt.excluded.law_references,
                    "question_label": stmt.excluded.question_label,
                    "tags": stmt.excluded.tags,
                    "updated_at": func.now(),
                },
            )
            conn.execute(do_update)

    def upsert_srs(self, question_id: str, payload: Dict[str, Optional[str]]) -> None:
        with self.engine.begin() as conn:
            stmt = sqlite_insert(srs_table).values(question_id=question_id, **payload)
            do_update = stmt.on_conflict_do_update(
                index_elements=[srs_table.c.question_id],
                set_={key: getattr(stmt.excluded, key) for key in payload},
            )
            conn.execute(do_update)

    def log_import(self, payload: Dict[str, Optional[str]]) -> None:
        with self.engine.begin() as conn:
            conn.execute(sa_insert(import_logs_table).values(**payload))

    def save_mapping_profile(self, name: str, kind: str, mapping: Dict[str, str]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sa_insert(mapping_profiles_table).values(
                    name=name,
                    kind=kind,
                    mapping_json=mapping,
                )
            )

    def fetch_mapping_profiles(self, kind: Optional[str] = None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            stmt = select(mapping_profiles_table)
            if kind:
                stmt = stmt.where(mapping_profiles_table.c.kind == kind)
            df = pd.read_sql(stmt, conn)
        return df

    def initialize_from_csv(self) -> None:
        with self.engine.connect() as conn:
            existing = conn.execute(select(func.count()).select_from(questions_table)).scalar()
        if existing and existing > 0:
            return
        questions_path = DATA_DIR / "questions_sample.csv"
        answers_path = DATA_DIR / "answers_sample.csv"
        if not questions_path.exists() or not answers_path.exists():
            return
        df_q = pd.read_csv(questions_path)
        df_a = pd.read_csv(answers_path)
        df_q = normalize_questions(df_q)
        df_a = normalize_answers(df_a)
        merged, *_ = merge_questions_answers(df_q, df_a, policy={"explanation": "overwrite", "tags": "merge"})
        self.upsert_questions(merged)
        rebuild_tfidf_cache()


@st.cache_data(show_spinner=False)
def load_questions_df() -> pd.DataFrame:
    engine = get_engine()
    db = DBManager(engine)
    df = db.load_dataframe(questions_table)
    return df


@st.cache_resource
def get_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words=None, max_features=5000)


@st.cache_resource
def get_law_revision_analyzer() -> LawRevisionAnalyzer:
    return LawRevisionAnalyzer()


def get_law_update_service(db: DBManager) -> LawUpdateSyncService:
    analyzer = get_law_revision_analyzer()
    return LawUpdateSyncService(
        sources=DEFAULT_LAW_SOURCES,
        analyzer=analyzer,
        id_builder=generate_law_revision_question_id,
    )


def rebuild_tfidf_cache() -> None:
    load_questions_df.clear()
    get_vectorizer.clear()
    load_questions_df()


def get_question_texts(df: pd.DataFrame) -> pd.Series:
    return df["question"].fillna("") + "\n" + df["explanation"].fillna("")


def compute_similarity(target_id: str, top_n: int = 3) -> pd.DataFrame:
    df = load_questions_df()
    if df.empty or target_id not in df["id"].values:
        return pd.DataFrame()
    texts = get_question_texts(df)
    vectorizer = get_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    index = df.index[df["id"] == target_id][0]
    target_vec = matrix[index]
    sims = cosine_similarity(target_vec, matrix).flatten()
    df = df.assign(similarity=sims)
    df = df[df["id"] != target_id].nlargest(top_n, "similarity")
    return df[["id", "year", "q_no", "category", "question", "similarity"]]


def normalize_questions(df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = df.copy()
    if mapping:
        df = df.rename(columns=mapping)
    required_cols = [
        "year",
        "q_no",
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"必要な列が不足しています: {col}")
    df["year"] = df["year"].astype(int)
    df["q_no"] = df["q_no"].astype(int)
    df["category"] = df.get("category", "").fillna("").apply(normalize_category)
    df["category"] = df["category"].replace("", CATEGORY_CHOICES[0])
    df["topic"] = df.get("topic", "").fillna("")
    df["explanation"] = df.get("explanation", "").fillna("")
    df["difficulty"] = (
        df.get("difficulty")
        .fillna(DIFFICULTY_DEFAULT)
        .replace("", DIFFICULTY_DEFAULT)
        .astype(int)
    )
    df["tags"] = df.get("tags", "").fillna("")
    if "id" not in df.columns or df["id"].isna().any():
        df["id"] = df.apply(generate_question_id, axis=1)
    df = df.drop_duplicates(subset=["id"])
    df = df[
        [
            "id",
            "year",
            "q_no",
            "category",
            "topic",
            "question",
            "choice1",
            "choice2",
            "choice3",
            "choice4",
            "explanation",
            "difficulty",
            "tags",
        ]
    ]
    return df


def normalize_answers(df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = df.copy()
    if mapping:
        df = df.rename(columns=mapping)
    if "year" not in df.columns or "q_no" not in df.columns:
        raise ValueError("year と q_no は必須です")
    df["year"] = df["year"].astype(int)
    df["q_no"] = df["q_no"].astype(int)
    for col in ["explanation", "tags"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def generate_predicted_question_id(row: pd.Series) -> str:
    base = f"predicted|{row.get('label', '')}|{str(row.get('question', ''))[:80]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def generate_law_revision_question_id(row: pd.Series) -> str:
    base = (
        f"lawrev|{row.get('law_name', '')}|{row.get('revision_year', '')}|"
        f"{str(row.get('question', ''))[:80]}"
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def normalize_predicted_questions(
    df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    df = df.copy()
    if mapping:
        df = df.rename(columns=mapping)
    required_cols = ["question", "choice1", "choice2", "choice3", "choice4"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"必要な列が不足しています: {col}")
    for col in required_cols:
        df[col] = df[col].fillna("").astype(str)
    df["label"] = df.get("label", "").fillna("").astype(str)
    df["category"] = df.get("category", "").fillna("").astype(str)
    df["topic"] = df.get("topic", "").fillna("").astype(str)
    df["source"] = df.get("source", "").fillna("").astype(str)
    df["year"] = df.get("year", "").fillna("").astype(str)
    df["q_no"] = df.get("q_no", "").fillna("").astype(str)
    df["explanation"] = df.get("explanation", "").fillna("").astype(str)
    df["difficulty"] = (
        df.get("difficulty")
        .fillna(DIFFICULTY_DEFAULT)
        .replace("", DIFFICULTY_DEFAULT)
        .astype(int)
    )
    df["tags"] = df.get("tags", "").fillna("").astype(str)
    if "correct" in df.columns:
        df["correct"] = (
            pd.to_numeric(df["correct"], errors="coerce")
            .where(lambda x: x.isin([1, 2, 3, 4]))
            .astype("Int64")
        )
    else:
        df["correct"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    if "id" not in df.columns or df["id"].isna().any() or (df["id"].astype(str).str.strip() == "").any():
        df["id"] = df.apply(generate_predicted_question_id, axis=1)
    df["id"] = df["id"].astype(str)
    df = df.drop_duplicates(subset=["id"])
    columns = [
        "id",
        "label",
        "year",
        "q_no",
        "category",
        "topic",
        "source",
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
        "correct",
        "explanation",
        "difficulty",
        "tags",
    ]
    df = df[columns]
    return df


def normalize_law_revision_questions(
    df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    df = df.copy()
    if mapping:
        df = df.rename(columns=mapping)
    required_cols = ["law_name", "question", "choice1", "choice2", "choice3", "choice4"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"必要な列が不足しています: {col}")
    for col in ["question", "choice1", "choice2", "choice3", "choice4"]:
        df[col] = df[col].fillna("").astype(str)
    df["law_name"] = df.get("law_name", "").fillna("").astype(str)
    optional_str_cols = [
        "label",
        "category",
        "topic",
        "source",
        "effective_date",
        "explanation",
        "tags",
        "auto_summary",
        "auto_cloze",
        "review_status",
        "generated_from",
        "fetched_at",
    ]
    for col in optional_str_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "review_status" in df.columns:
        df["review_status"] = df["review_status"].replace("", "pending")
    if "reviewed_at" not in df.columns:
        df["reviewed_at"] = ""
    df["reviewed_at"] = pd.to_datetime(df["reviewed_at"], errors="coerce")
    if "fetched_at" in df.columns:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    df["revision_year"] = (
        pd.to_numeric(df.get("revision_year"), errors="coerce")
        .astype("Int64")
    )
    if "difficulty" not in df.columns:
        df["difficulty"] = DIFFICULTY_DEFAULT
    df["difficulty"] = (
        df.get("difficulty")
        .fillna(DIFFICULTY_DEFAULT)
        .replace("", DIFFICULTY_DEFAULT)
        .astype(int)
    )
    if "correct" in df.columns:
        df["correct"] = (
            pd.to_numeric(df["correct"], errors="coerce")
            .where(lambda x: x.isin([1, 2, 3, 4]))
            .astype("Int64")
        )
    else:
        df["correct"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    if "id" not in df.columns or df["id"].isna().any() or (df["id"].astype(str).str.strip() == "").any():
        df["id"] = df.apply(generate_law_revision_question_id, axis=1)
    df["id"] = df["id"].astype(str)
    df = df.drop_duplicates(subset=["id"])
    columns = LAW_REVISION_TEMPLATE_COLUMNS.copy()
    df = df[columns]
    return df


def validate_question_records(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    required_cols = [
        "year",
        "q_no",
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"必須列が不足しています: {', '.join(missing)}")
        return errors
    working = df.reset_index(drop=True)
    if "id" in working.columns:
        dup_ids = working[working["id"].notna() & working["id"].duplicated()]["id"].unique()
        if dup_ids.size > 0:
            errors.append(f"重複したIDが存在します: {', '.join(map(str, dup_ids[:5]))}")
    dup_keys = working.duplicated(subset=["year", "q_no"], keep=False)
    if dup_keys.any():
        duplicates = working.loc[dup_keys, ["year", "q_no"]].reset_index()
        sample = ", ".join(
            f"{row.year}年問{row.q_no} (行{row['index'] + 2})" for _, row in duplicates.head(5).iterrows()
        )
        errors.append(f"年度と問番の組み合わせが重複しています: {sample}")
    for row_number, row in enumerate(working.itertuples(index=False), start=2):
        year = getattr(row, "year", "?")
        q_no = getattr(row, "q_no", "?")
        label = f"{year}年問{q_no} (行{row_number})"
        question_text = str(getattr(row, "question", ""))
        if not question_text.strip():
            errors.append(f"{label}：問題文が空欄です。")
        choices = [str(getattr(row, f"choice{i}", "")).strip() for i in range(1, 5)]
        if any(choice == "" for choice in choices):
            errors.append(f"{label}：空欄の選択肢があります。")
        non_empty = [c for c in choices if c]
        if len(set(non_empty)) < len(non_empty):
            errors.append(f"{label}：選択肢が重複しています。")
    return errors


def validate_answer_records(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    required_cols = ["year", "q_no", "correct_number"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"必須列が不足しています: {', '.join(missing)}")
        return errors
    working = df.reset_index(drop=True)
    dup_keys = working.duplicated(subset=["year", "q_no"], keep=False)
    if dup_keys.any():
        duplicates = working.loc[dup_keys, ["year", "q_no"]].reset_index()
        sample = ", ".join(
            f"{row.year}年問{row.q_no} (行{row['index'] + 2})" for _, row in duplicates.head(5).iterrows()
        )
        errors.append(f"年度と問番の組み合わせが重複しています: {sample}")
    if working["correct_number"].isna().any():
        rows = (working["correct_number"].isna().to_numpy().nonzero()[0] + 2).tolist()
        rows_text = ", ".join(map(str, rows[:5]))
        errors.append(f"correct_number に空欄があります (行 {rows_text})。")
    try:
        invalid = pd.to_numeric(working["correct_number"], errors="coerce")
    except Exception:
        invalid = pd.Series([np.nan] * len(working))
    out_of_range = working[(invalid < 1) | (invalid > 4) | invalid.isna()]
    if not out_of_range.empty:
        sample_rows = ", ".join(
            f"{row.year}年問{row.q_no} (行{row['index'] + 2})"
            for _, row in out_of_range.reset_index().head(5).iterrows()
        )
        errors.append(f"correct_number は1〜4の範囲で指定してください: {sample_rows}")
    return errors


def validate_predicted_records(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    required_cols = ["question", "choice1", "choice2", "choice3", "choice4"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"必須列が不足しています: {', '.join(missing)}")
        return errors
    working = df.reset_index(drop=True)
    for row_number, row in enumerate(working.itertuples(index=False), start=2):
        label = getattr(row, "label", "") or f"行{row_number}"
        question_text = str(getattr(row, "question", "")).strip()
        if not question_text:
            errors.append(f"{label}: 問題文が空欄です。")
        choices = [str(getattr(row, f"choice{i}", "")).strip() for i in range(1, 5)]
        if any(choice == "" for choice in choices):
            errors.append(f"{label}: 空欄の選択肢があります。")
        non_empty = [c for c in choices if c]
        if len(set(non_empty)) < len(non_empty):
            errors.append(f"{label}: 選択肢が重複しています。")
    if "id" in working.columns:
        dup_ids = working[working["id"].notna() & working["id"].astype(str).str.strip().duplicated()]["id"].unique()
        if dup_ids.size > 0:
            errors.append(f"重複したIDが存在します: {', '.join(map(str, dup_ids[:5]))}")
    return errors


def validate_law_revision_records(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    required_cols = ["law_name", "question", "choice1", "choice2", "choice3", "choice4"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"必須列が不足しています: {', '.join(missing)}")
        return errors
    working = df.reset_index(drop=True)
    for row_number, row in enumerate(working.itertuples(index=False), start=2):
        label = getattr(row, "label", "") or f"行{row_number}"
        law_name = str(getattr(row, "law_name", "")).strip()
        if not law_name:
            errors.append(f"{label}: 法令名 (law_name) を入力してください。")
        question_text = str(getattr(row, "question", "")).strip()
        if not question_text:
            errors.append(f"{label}: 問題文が空欄です。")
        choices = [str(getattr(row, f"choice{i}", "")).strip() for i in range(1, 5)]
        if any(choice == "" for choice in choices):
            errors.append(f"{label}: 空欄の選択肢があります。")
        non_empty = [c for c in choices if c]
        if len(set(non_empty)) < len(non_empty):
            errors.append(f"{label}: 選択肢が重複しています。")
        revision_year_value = getattr(row, "revision_year", None)
        if revision_year_value not in (None, "", pd.NA):
            try:
                int(str(revision_year_value).strip())
            except Exception:
                errors.append(f"{label}: revision_year は西暦の数値で入力してください。")
        correct_value = getattr(row, "correct", None)
        if correct_value not in (None, "", pd.NA):
            try:
                numeric_correct = int(correct_value)
            except Exception:
                errors.append(f"{label}: correct は1〜4の数値で入力してください。")
            else:
                if numeric_correct not in {1, 2, 3, 4}:
                    errors.append(f"{label}: correct は1〜4の範囲で指定してください。")
    if "id" in working.columns:
        dup_ids = working[working["id"].notna() & working["id"].astype(str).str.strip().duplicated()]["id"].unique()
        if dup_ids.size > 0:
            errors.append(f"重複したIDが存在します: {', '.join(map(str, dup_ids[:5]))}")
    return errors


def build_answers_export(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        correct_number = row.get("correct")
        if pd.isna(correct_number):
            correct_number = None
            correct_label = ""
            correct_text = ""
        else:
            correct_number = int(correct_number)
            correct_label = ["A", "B", "C", "D"][correct_number - 1]
            correct_text = str(row.get(f"choice{correct_number}", ""))
        records.append(
            {
                "year": row.get("year"),
                "q_no": row.get("q_no"),
                "correct_number": correct_number,
                "correct_label": correct_label,
                "correct_text": correct_text,
                "explanation": row.get("explanation", ""),
                "difficulty": row.get("difficulty"),
                "tags": row.get("tags", ""),
            }
        )
    return pd.DataFrame(records, columns=ANSWER_TEMPLATE_COLUMNS)


def build_sample_questions_csv() -> str:
    sample_rows = [
        {
            "year": 2023,
            "q_no": 1,
            "category": "宅建業法",
            "topic": "免許",
            "question": "宅地建物取引業者の免許について正しいものはどれか。",
            "choice1": "免許権者は必ず国土交通大臣である。",
            "choice2": "法人が免許を受ける場合、専任取引士は不要である。",
            "choice3": "免許替えの際は旧免許の有効期間を引き継げる。",
            "choice4": "知事免許業者が二以上の都道府県に事務所を設けるときは大臣免許が必要である。",
            "explanation": "宅建業法上、二以上の都道府県に事務所を設ける場合は大臣免許が必要。",
            "difficulty": 3,
            "tags": "宅建業法;免許",
        },
        {
            "year": 2023,
            "q_no": 2,
            "category": "権利関係",
            "topic": "物権変動",
            "question": "不動産物権変動の対抗要件に関する記述として正しいものはどれか。",
            "choice1": "不動産の贈与は口頭でも第三者に対抗できる。",
            "choice2": "所有権移転登記を備えなければ第三者に対抗できない。",
            "choice3": "仮登記のままでも常に第三者に優先する。",
            "choice4": "地上権設定は登記簿の記載を要しない。",
            "explanation": "不動産物権変動の対抗要件は原則として登記である。",
            "difficulty": 2,
            "tags": "権利関係;物権変動",
        },
    ]
    buffer = io.StringIO()
    pd.DataFrame(sample_rows, columns=QUESTION_TEMPLATE_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue()


def build_sample_answers_csv() -> str:
    sample_rows = [
        {
            "year": 2023,
            "q_no": 1,
            "correct_number": 4,
            "correct_label": "D",
            "correct_text": "知事免許業者が二以上の都道府県に事務所を設けるときは大臣免許が必要。",
            "explanation": "宅建業法の免許制度に基づき、複数都道府県で営業する場合は大臣免許が必要です。",
            "difficulty": 3,
            "tags": "宅建業法;免許",
        },
        {
            "year": 2023,
            "q_no": 2,
            "correct_number": 2,
            "correct_label": "B",
            "correct_text": "所有権移転登記を備えなければ第三者に対抗できない。",
            "explanation": "不動産物権変動の対抗要件は登記が原則です。",
            "difficulty": 2,
            "tags": "権利関係;物権変動",
        },
    ]
    buffer = io.StringIO()
    pd.DataFrame(sample_rows, columns=ANSWER_TEMPLATE_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue()


def build_sample_predicted_csv() -> str:
    sample_rows = [
        {
            "label": "予想問題001",
            "category": "宅建業法",
            "topic": "重要事項説明",
            "source": "講師予想",
            "question": "宅地建物取引業者が重要事項説明を行う際の留意点について正しいものはどれか。",
            "choice1": "専任取引士以外でも宅建士証の提示があれば説明できる。",
            "choice2": "重要事項説明書は電磁的方法で交付できる。",
            "choice3": "35条書面は売主が直接説明する必要がある。",
            "choice4": "買主の承諾があれば口頭説明のみでよい。",
            "correct": 2,
            "explanation": "重要事項説明書は一定要件の下で電磁的方法による交付が認められています。",
            "year": dt.datetime.now().year + 1,
            "q_no": "予想1",
            "difficulty": 3,
            "tags": "予想;重要事項説明",
        },
        {
            "label": "予想問題002",
            "category": "権利関係",
            "topic": "借地借家法",
            "source": "模試作成チーム",
            "question": "定期借家契約に関する次の記述のうち、適切なものはどれか。",
            "choice1": "書面で契約すれば期間満了前でも常に解約できる。",
            "choice2": "更新を前提としない旨を口頭で合意すれば定期借家となる。",
            "choice3": "公正証書等の書面で契約しなければ効力を生じない。",
            "choice4": "定期借家契約では中途解約は一切認められない。",
            "correct": 3,
            "explanation": "定期借家契約は公正証書等の書面による契約が必要です。",
            "year": dt.datetime.now().year + 1,
            "q_no": "予想2",
            "difficulty": 2,
            "tags": "予想;借地借家法",
        },
    ]
    buffer = io.StringIO()
    pd.DataFrame(sample_rows, columns=PREDICTED_TEMPLATE_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue()


def build_sample_law_revision_csv() -> str:
    sample_rows = [
        {
            "label": "法改正対策001",
            "law_name": "宅建業法",
            "revision_year": dt.datetime.now().year,
            "effective_date": f"{dt.datetime.now().year}-04-01",
            "category": "宅建業法",
            "topic": "改正ポイント",
            "source": "官報",
            "question": "最新の宅建業法改正で追加された説明義務について正しいものはどれか。",
            "choice1": "重要事項説明書に改正内容を追記する必要がある。",
            "choice2": "買主が希望すれば省略できる。",
            "choice3": "宅建士証の提示義務が免除された。",
            "choice4": "宅地建物以外の取引に限り適用される。",
            "correct": 1,
            "explanation": "改正条文のポイントを記載します。",
            "difficulty": DIFFICULTY_DEFAULT,
            "tags": "法改正;直前対策",
            "auto_summary": "宅建業法の改正により重要事項説明書へ改正内容の追記が求められる。",
            "auto_cloze": "宅建業法の改正により重要事項説明書に＿＿＿を追記する必要がある。",
            "review_status": "approved",
            "reviewed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "generated_from": "サンプルデータ",
            "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
    ]
    buffer = io.StringIO()
    pd.DataFrame(sample_rows, columns=LAW_REVISION_TEMPLATE_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue()


def merge_questions_answers(
    questions: pd.DataFrame,
    answers: pd.DataFrame,
    policy: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = questions.copy()
    if "correct" not in merged.columns:
        merged["correct"] = np.nan
    rejects_q = []
    rejects_a = []

    answer_map = {}
    for _, row in answers.iterrows():
        key = (row["year"], row["q_no"])
        answer_map[key] = row

    def determine_correct(row: pd.Series, ans_row: Optional[pd.Series]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        if ans_row is None:
            return None, None, None
        correct_number = ans_row.get("correct_number")
        if pd.notna(correct_number):
            try:
                correct_number = int(correct_number)
                if correct_number in [1, 2, 3, 4]:
                    return correct_number, ans_row.get("explanation"), ans_row.get("tags")
            except ValueError:
                pass
        label = ans_row.get("correct_label")
        if pd.notna(label):
            label = str(label).strip().upper()
            mapping = {"Ａ": 1, "A": 1, "Ｂ": 2, "B": 2, "Ｃ": 3, "C": 3, "Ｄ": 4, "D": 4}
            if label in mapping:
                return mapping[label], ans_row.get("explanation"), ans_row.get("tags")
        text_answer = ans_row.get("correct_text")
        if pd.notna(text_answer) and text_answer:
            choices = [row.get(f"choice{i}", "") for i in range(1, 5)]
            for idx, choice in enumerate(choices, start=1):
                if str(choice).strip() == str(text_answer).strip():
                    return idx, ans_row.get("explanation"), ans_row.get("tags")
            ratios = [fuzz.ratio(str(choice), str(text_answer)) for choice in choices]
            max_ratio = max(ratios)
            if max_ratio >= 90:
                idx = ratios.index(max_ratio) + 1
                return idx, ans_row.get("explanation"), ans_row.get("tags")
            else:
                rejects_a.append({**ans_row.to_dict(), "reason": "選択肢と一致せず"})
                return None, None, None
        rejects_a.append({**ans_row.to_dict(), "reason": "正答情報が不足"})
        return None, None, None

    conflicts = []
    for idx, row in merged.iterrows():
        key = (row["year"], row["q_no"])
        ans_row = answer_map.get(key)
        correct, exp_override, tags_new = determine_correct(row, ans_row)
        if correct is not None:
            if pd.notna(row.get("correct")) and row.get("correct") != correct:
                conflicts.append({
                    "id": row["id"],
                    "year": row["year"],
                    "q_no": row["q_no"],
                    "existing": row.get("correct"),
                    "incoming": correct,
                })
            merged.at[idx, "correct"] = correct
        if ans_row is not None:
            if policy.get("explanation", "overwrite") == "overwrite" and pd.notna(ans_row.get("explanation")):
                merged.at[idx, "explanation"] = ans_row.get("explanation")
            elif policy.get("explanation") == "append":
                merged.at[idx, "explanation"] = (str(row.get("explanation", "")) + "\n" + str(ans_row.get("explanation", ""))).strip()
            if tags_new:
                if policy.get("tags", "merge") == "merge" and row.get("tags"):
                    tags_combined = set(str(row.get("tags", "")).split(";")) | set(str(tags_new).split(";"))
                    merged.at[idx, "tags"] = ";".join(sorted({t.strip() for t in tags_combined if t.strip()}))
                else:
                    merged.at[idx, "tags"] = tags_new
            if pd.notna(ans_row.get("difficulty")):
                merged.at[idx, "difficulty"] = int(ans_row.get("difficulty"))
    merged["correct"] = merged["correct"].fillna(0).astype(int).replace(0, np.nan)
    rejects_q_df = pd.DataFrame(rejects_q)
    rejects_a_df = pd.DataFrame(rejects_a)
    conflicts_df = pd.DataFrame(conflicts)
    return merged, rejects_q_df, rejects_a_df, conflicts_df


def normalize_category(value: str) -> str:
    if not value:
        return CATEGORY_CHOICES[0]
    value = str(value).strip()
    if value in CATEGORY_CHOICES:
        return value
    for key, target in DEFAULT_CATEGORY_MAP.items():
        if key in value:
            return target
    scores = {cat: fuzz.partial_ratio(value, cat) for cat in CATEGORY_CHOICES}
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] >= 70:
        return best_cat
    return CATEGORY_CHOICES[0]


def generate_question_id(row: pd.Series) -> str:
    base = f"{row['year']}|{row['q_no']}|{str(row['question'])[:80]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


@dataclass
class ExamSession:
    id: Optional[int]
    name: str
    questions: List[str]
    started_at: dt.datetime
    year_mode: str
    mode: str


@dataclass
class QuestionNavigation:
    has_prev: bool = False
    has_next: bool = False
    on_prev: Optional[Callable[[], None]] = None
    on_next: Optional[Callable[[], None]] = None
    label: Optional[str] = None


def select_random_questions(
    df: pd.DataFrame,
    count: int,
    weights: Optional[Sequence[float]] = None,
) -> List[str]:
    if df.empty or count <= 0:
        return []
    ids = list(df["id"])
    population = len(ids)
    draw = min(count, population)
    if draw == 0:
        return []
    if weights is not None:
        weight_array = np.asarray(list(weights), dtype=float)
        if weight_array.shape[0] != population:
            raise ValueError("weights length must match dataframe length")
        weight_array = np.where(np.isnan(weight_array) | (weight_array < 0), 0.0, weight_array)
        if float(weight_array.sum()) <= 0.0:
            weight_array = np.ones(population, dtype=float)
        probabilities = weight_array / weight_array.sum()
        indices = np.random.choice(population, size=draw, replace=False, p=probabilities)
        return [ids[i] for i in indices]
    if draw == population:
        return ids
    return random.sample(ids, draw)


def build_question_priority(df: pd.DataFrame, attempts: pd.DataFrame) -> pd.DataFrame:
    settings = st.session_state.get("settings", {})
    low_conf_threshold = int(settings.get("review_low_confidence_threshold", 60))
    elapsed_days_threshold = max(int(settings.get("review_elapsed_days", 7)), 1)
    base = df[["id", "category", "difficulty"]].copy()
    base["difficulty"] = base["difficulty"].fillna(DIFFICULTY_DEFAULT).astype(float)
    metrics_defaults = {
        "attempts_count": 0,
        "correct_count": 0,
        "accuracy": 0.5,
        "last_confidence": float(low_conf_threshold),
        "days_since_last_attempt": float(elapsed_days_threshold),
    }
    for column, default in metrics_defaults.items():
        base[column] = default
    if not attempts.empty:
        attempts = attempts.copy()
        attempts["created_at"] = pd.to_datetime(attempts["created_at"])
        summary = (
            attempts.groupby("question_id")
            .agg(
                attempts_count=("is_correct", "count"),
                correct_count=("is_correct", "sum"),
                last_attempt_at=("created_at", "max"),
            )
            .reset_index()
        )
        last_conf = (
            attempts.sort_values("created_at")
            .groupby("question_id")
            .agg(last_confidence=("confidence", "last"))
            .reset_index()
        )
        summary = summary.merge(last_conf, on="question_id", how="left")
        summary["accuracy"] = summary["correct_count"] / summary["attempts_count"].replace(0, np.nan)
        summary["accuracy"] = summary["accuracy"].clip(lower=0.0, upper=1.0)
        now = dt.datetime.now()
        summary["days_since_last_attempt"] = (
            (now - summary["last_attempt_at"]).dt.total_seconds() / 86400.0
        )
        base = base.merge(
            summary,
            left_on="id",
            right_on="question_id",
            how="left",
            suffixes=("", "_summary"),
        )
        base.drop(
            columns=[col for col in ["question_id", "last_attempt_at"] if col in base.columns],
            inplace=True,
        )
        for column, default in metrics_defaults.items():
            summary_col = f"{column}_summary"
            if summary_col in base.columns:
                base[column] = base[summary_col]
                base.drop(columns=[summary_col], inplace=True)
        base["attempts_count"] = base["attempts_count"].fillna(0).astype(int)
        base["correct_count"] = base["correct_count"].fillna(0).astype(int)
        base["accuracy"] = base["accuracy"].fillna(0.5)
        base["last_confidence"] = pd.to_numeric(base["last_confidence"], errors="coerce")
        base["last_confidence"] = base["last_confidence"].fillna(low_conf_threshold)
        base["days_since_last_attempt"] = base["days_since_last_attempt"].fillna(float(elapsed_days_threshold))
    base["accuracy"] = base["accuracy"].clip(lower=0.0, upper=1.0)
    base["last_confidence"] = base["last_confidence"].clip(lower=0.0, upper=100.0)
    base["days_since_last_attempt"] = base["days_since_last_attempt"].clip(lower=0.0)
    accuracy_component = 1.0 + (1.0 - base["accuracy"])
    conf_gap = np.clip((low_conf_threshold - base["last_confidence"]) / max(low_conf_threshold, 1), 0.0, 1.0)
    confidence_component = 1.0 + conf_gap
    days_ratio = np.clip(base["days_since_last_attempt"] / float(elapsed_days_threshold), 0.0, 3.0)
    days_component = 1.0 + days_ratio * 0.4
    difficulty_component = 1.0 + (base["difficulty"] - 3.0) * 0.15
    base["raw_weight"] = (accuracy_component + confidence_component + days_component) * difficulty_component
    category_stats = (
        base.groupby("category")
        .agg(
            category_accuracy=("accuracy", "mean"),
            category_difficulty=("difficulty", "mean"),
        )
        .reset_index()
    )
    category_stats["category_accuracy"] = category_stats["category_accuracy"].fillna(0.5)
    category_stats["category_multiplier"] = 1.0 + (
        (1.0 - category_stats["category_accuracy"]) * 1.2
    ) + np.maximum(category_stats["category_difficulty"] - 3.0, 0.0) * 0.1
    multiplier_map = dict(zip(category_stats["category"], category_stats["category_multiplier"]))
    base["category_multiplier"] = base["category"].map(multiplier_map).fillna(1.0)
    base["weight"] = base["raw_weight"] * base["category_multiplier"]
    return base


def stratified_exam(
    df: pd.DataFrame, weight_map: Optional[Dict[str, float]] = None
) -> List[str]:
    quotas = {"宅建業法": 20, "権利関係": 14, "法令上の制限": 8, "税・その他": 8}
    selected = []
    remaining = df.copy()
    for category, quota in quotas.items():
        subset = remaining[remaining["category"] == category]
        subset_weights = None
        if weight_map is not None:
            subset_weights = [weight_map.get(qid, 1.0) for qid in subset["id"]]
        chosen = select_random_questions(subset, quota, subset_weights)
        selected.extend(chosen)
        remaining = remaining[~remaining["id"].isin(chosen)]
    if len(selected) < 50:
        additional_weights = None
        if weight_map is not None:
            additional_weights = [weight_map.get(qid, 1.0) for qid in remaining["id"]]
        additional = select_random_questions(remaining, 50 - len(selected), additional_weights)
        selected.extend(additional)
    return selected


def sm2_update(row: Optional[pd.Series], grade: int, initial_ease: float = 2.5) -> Dict[str, object]:
    today = dt.date.today()
    if row is None:
        repetition = 0
        prev_interval = 0
        ease = initial_ease
    else:
        repetition = row.get("repetition", 0) or 0
        prev_interval = row.get("interval", 0) or 0
        ease = row.get("ease", 2.5) or 2.5
    if grade >= 3:
        if repetition == 0:
            interval = 1
        elif repetition == 1:
            interval = 6
        else:
            interval = int(round(max(prev_interval, 1) * ease))
            interval = max(interval, 1)
        repetition += 1
    else:
        repetition = 0
        interval = 1
    ease = ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    ease = max(ease, 1.3)
    due_date = today + dt.timedelta(days=interval)
    return {
        "repetition": repetition,
        "interval": interval,
        "ease": ease,
        "due_date": due_date,
        "last_grade": grade,
        "updated_at": dt.datetime.now(),
    }


def inject_ui_styles() -> None:
    if st.session_state.get("_ui_styles_injected"):
        return
    inject_style(
        """
.takken-choice-button {
    transition: filter 0.2s ease;
}
.takken-choice-button button {
    width: 100%;
    min-height: 56px;
    font-size: 1.05rem;
    border-radius: 0.8rem;
    display: flex;
    align-items: flex-start;
    justify-content: flex-start;
    text-align: left;
    line-height: 1.5;
    white-space: normal;
    padding: 0.75rem 1rem;
    gap: 0.75rem;
    border-width: 1.5px;
}
.takken-choice-button.is-selected:not(.is-graded) button {
    border-color: #4c6ef5 !important;
    box-shadow: 0 0 0 2px rgba(76, 110, 245, 0.18);
}
.takken-choice-button.is-correct button {
    background-color: #f0f9f4 !important;
    border-color: #2f9e44 !important;
    color: #1b4332 !important;
}
.takken-choice-button.is-incorrect button {
    background-color: #fff5f5 !important;
    border-color: #e03131 !important;
    color: #871b1b !important;
}
.takken-choice-button.is-dimmed button {
    filter: grayscale(0.25);
    opacity: 0.7;
}
.takken-choice-button.is-graded button {
    cursor: default;
}
.takken-choice-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.takken-action-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin: 1rem 0;
}
.takken-action-item {
    flex: 1 1 160px;
}
@media (max-width: 768px) {
    .takken-choice-grid {
        grid-template-columns: 1fr;
    }
    .takken-action-bar {
        flex-direction: column;
    }
    .takken-action-item {
        width: 100%;
    }
}
.takken-inline-actions button {
    min-height: 48px;
}
[data-testid="stMarkdownContainer"] {
    text-align: left;
}
[data-testid="stMarkdownContainer"] * {
    text-align: left !important;
}
.takken-feedback-summary {
    margin-top: 0.5rem;
    padding: 0.75rem 1rem;
    background: #f5f7fb;
    border-radius: 0.75rem;
    font-weight: 500;
    color: #36435a;
    display: inline-flex;
    gap: 0.5rem;
    align-items: flex-start;
}
.takken-feedback-summary strong {
    font-weight: 600;
}
.takken-tag-group {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.25rem 0 0.75rem;
}
.takken-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.65rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    line-height: 1.2;
    border: 1px solid transparent;
    background-color: rgba(108, 117, 125, 0.16);
    color: #374151;
}
[data-theme="dark"] .takken-tag {
    background-color: rgba(148, 163, 184, 0.16);
    color: #e2e8f0;
}
.takken-tag-icon {
    font-size: 0.95rem;
    line-height: 1;
}
.takken-tag-label {
    line-height: 1.2;
}
.takken-tag--category-business {
    background-color: rgba(76, 110, 245, 0.18);
    border-color: rgba(76, 110, 245, 0.35);
    color: #2f4f9e;
}
[data-theme="dark"] .takken-tag--category-business {
    background-color: rgba(118, 145, 255, 0.22);
    border-color: rgba(118, 145, 255, 0.5);
    color: #dbe4ff;
}
.takken-tag--category-rights {
    background-color: rgba(34, 197, 94, 0.18);
    border-color: rgba(34, 197, 94, 0.35);
    color: #1b4332;
}
[data-theme="dark"] .takken-tag--category-rights {
    background-color: rgba(52, 211, 153, 0.22);
    border-color: rgba(52, 211, 153, 0.5);
    color: #bbf7d0;
}
.takken-tag--category-regulation {
    background-color: rgba(245, 158, 11, 0.18);
    border-color: rgba(245, 158, 11, 0.35);
    color: #854d0e;
}
[data-theme="dark"] .takken-tag--category-regulation {
    background-color: rgba(251, 191, 36, 0.22);
    border-color: rgba(251, 191, 36, 0.5);
    color: #fde68a;
}
.takken-tag--category-tax {
    background-color: rgba(236, 72, 153, 0.18);
    border-color: rgba(236, 72, 153, 0.35);
    color: #831843;
}
[data-theme="dark"] .takken-tag--category-tax {
    background-color: rgba(244, 114, 182, 0.22);
    border-color: rgba(244, 114, 182, 0.5);
    color: #fbcfe8;
}
.takken-tag--category-default {
    background-color: rgba(100, 116, 139, 0.2);
    border-color: rgba(100, 116, 139, 0.3);
    color: #1f2937;
}
[data-theme="dark"] .takken-tag--category-default {
    background-color: rgba(148, 163, 184, 0.24);
    border-color: rgba(148, 163, 184, 0.45);
    color: #e2e8f0;
}
.takken-tag--topic {
    background-color: rgba(79, 70, 229, 0.15);
    border-color: rgba(79, 70, 229, 0.3);
    color: #3730a3;
}
[data-theme="dark"] .takken-tag--topic {
    background-color: rgba(129, 140, 248, 0.22);
    border-color: rgba(129, 140, 248, 0.45);
    color: #c7d2fe;
}
.takken-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0;
    font-size: 0.9rem;
}
.takken-table th,
.takken-table td {
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.35);
    vertical-align: top;
}
[data-theme="dark"] .takken-table th,
[data-theme="dark"] .takken-table td {
    border-bottom-color: rgba(203, 213, 225, 0.2);
}
.takken-table tbody tr:last-child th,
.takken-table tbody tr:last-child td {
    border-bottom: none;
}
.takken-table thead th {
    font-weight: 700;
    color: inherit;
}
""",
        "takken-ui-styles",
    )
    st.session_state["_ui_styles_injected"] = True


def _build_tag_html(label: str, icon: Optional[str], class_name: str) -> str:
    if not label:
        return ""
    safe_label = html_module.escape(label)
    icon_html = (
        f'<span class="takken-tag-icon">{html_module.escape(icon)}</span>'
        if icon
        else ""
    )
    return (
        f'<span class="takken-tag {class_name}">{icon_html}'
        f'<span class="takken-tag-label">{safe_label}</span></span>'
    )


def render_category_tag(category: Optional[str]) -> str:
    if category is None or (isinstance(category, float) and pd.isna(category)):
        return ""
    normalized = str(category).strip()
    if not normalized:
        return ""
    canonical = DEFAULT_CATEGORY_MAP.get(normalized, normalized)
    style = CATEGORY_BADGE_STYLE.get(canonical, CATEGORY_BADGE_DEFAULT)
    style_class = style.get("class", CATEGORY_BADGE_DEFAULT["class"])
    icon = style.get("icon", CATEGORY_BADGE_DEFAULT["icon"])
    return _build_tag_html(canonical, icon, style_class)


def render_topic_tag(topic: Optional[str]) -> str:
    if topic is None or (isinstance(topic, float) and pd.isna(topic)):
        return ""
    normalized = str(topic).strip()
    if not normalized:
        return ""
    return _build_tag_html(normalized, "🧠", "takken-tag--topic")


def render_category_topic_tags(
    category: Optional[str], topic: Optional[str]
) -> str:
    inject_ui_styles()
    tags: List[str] = []
    category_tag = render_category_tag(category)
    if category_tag:
        tags.append(category_tag)
    topic_tag = render_topic_tag(topic)
    if topic_tag:
        tags.append(topic_tag)
    if not tags:
        return ""
    return f'<div class="takken-tag-group">{"".join(tags)}</div>'


def render_table_with_category_tags(
    df: pd.DataFrame,
    column_order: Sequence[str],
    rename_map: Optional[Dict[str, str]] = None,
    index_column: Optional[str] = None,
    table_classes: str = "takken-table",
) -> None:
    inject_ui_styles()
    rename_map = rename_map or {}
    if df.empty:
        st.markdown("<p>表示するデータがありません。</p>", unsafe_allow_html=True)
        return
    display = df.loc[:, column_order].copy()
    if "category" in display.columns:
        category_pos = list(display.columns).index("category")
        category_tags = display["category"].apply(render_category_tag)
        display = display.drop(columns=["category"])
        display.insert(category_pos, "_category_tag", category_tags)
    if index_column and index_column in display.columns:
        display = display.set_index(index_column)
        display.index.name = rename_map.get(index_column, index_column)
    rename_columns: Dict[str, str] = {}
    for original, new_name in rename_map.items():
        if original == index_column:
            continue
        if original in display.columns:
            rename_columns[original] = new_name
    if "_category_tag" in display.columns:
        rename_columns["_category_tag"] = rename_map.get("category", "分野")
    display = display.rename(columns=rename_columns)
    html_table = display.to_html(
        escape=False,
        classes=table_classes,
        border=0,
        na_rep="",
    )
    st.markdown(html_table, unsafe_allow_html=True)


def confidence_to_grade(is_correct: bool, confidence: int) -> int:
    confidence = max(0, min(100, confidence))
    if is_correct:
        if confidence >= 90:
            return 5
        if confidence >= 70:
            return 4
        if confidence >= 50:
            return 3
        return 2
    if confidence >= 70:
        return 1
    return 0


def get_offline_attempts_df() -> pd.DataFrame:
    records = st.session_state.get("offline_attempts", [])
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def persist_offline_attempts(df: pd.DataFrame) -> None:
    if df.empty:
        return
    OFFLINE_EXPORT_DIR.mkdir(exist_ok=True)
    csv_path = OFFLINE_EXPORT_DIR / "attempts_latest.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    parquet_path = OFFLINE_EXPORT_DIR / "attempts_latest.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        st.session_state["_offline_parquet_error"] = None
    except Exception as exc:
        st.session_state["_offline_parquet_error"] = str(exc)
        if parquet_path.exists():
            parquet_path.unlink()


def log_offline_attempt(entry: Dict[str, object]) -> None:
    attempts = st.session_state.setdefault("offline_attempts", [])
    attempts.append(entry)
    df = get_offline_attempts_df()
    persist_offline_attempts(df)


def render_offline_downloads(key_prefix: str) -> None:
    df = get_offline_attempts_df()
    if df.empty:
        return
    with st.expander("学習結果ダウンロード", expanded=False):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "CSVをダウンロード",
            data=csv_buffer.getvalue(),
            file_name="takken_learning_log.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )
        parquet_buffer = io.BytesIO()
        parquet_error = st.session_state.get("_offline_parquet_error")
        if parquet_error:
            st.warning(f"Parquetの自動保存に失敗しました: {parquet_error}")
        try:
            parquet_buffer.seek(0)
            df.to_parquet(parquet_buffer, index=False)
            st.download_button(
                "Parquetをダウンロード",
                data=parquet_buffer.getvalue(),
                file_name="takken_learning_log.parquet",
                mime="application/octet-stream",
                key=f"{key_prefix}_parquet",
            )
        except Exception as exc:
            st.warning(f"Parquetのダウンロード生成に失敗しました: {exc}")
        st.caption(f"ファイルは {OFFLINE_EXPORT_DIR.as_posix()} にも自動保存されます。")


def get_review_candidate_ids(db: DBManager) -> Set[str]:
    review_ids: Set[str] = set()
    attempts = db.get_attempt_stats()
    if not attempts.empty:
        attempts["created_at"] = pd.to_datetime(attempts["created_at"])
        last_attempts = (
            attempts.sort_values("created_at").groupby("question_id", as_index=False).tail(1)
        )
        review_ids.update(last_attempts[last_attempts["is_correct"] == 0]["question_id"].tolist())
        low_conf = int(st.session_state["settings"].get("review_low_confidence_threshold", 60))
        if "confidence" in last_attempts.columns:
            confidence_series = last_attempts["confidence"].fillna(101)
            review_ids.update(
                last_attempts[confidence_series <= low_conf]["question_id"].tolist()
            )
        days_threshold = int(st.session_state["settings"].get("review_elapsed_days", 7))
        cutoff = dt.datetime.now() - dt.timedelta(days=days_threshold)
        review_ids.update(
            last_attempts[last_attempts["created_at"] <= cutoff]["question_id"].tolist()
        )
    srs_due = db.get_due_srs()
    if not srs_due.empty:
        review_ids.update(srs_due["question_id"].tolist())
    return {str(qid) for qid in review_ids if pd.notna(qid)}


def parse_explanation_sections(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    if not text:
        return "", []
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    sections: List[Tuple[str, str]] = []
    summary = ""
    for line in lines:
        match = re.match(r"^【([^】]+)】(.*)$", line)
        if match:
            label = match.group(1).strip()
            content = match.group(2).strip()
        else:
            label = "補足"
            content = line.strip()
        sections.append((label, content))
        if not summary and label in ("要点", "結論") and content:
            summary = content
    if not summary and lines:
        summary = lines[0]
    summary = summary.strip()
    if len(summary) > 80:
        summary = summary[:77] + "…"
    return summary, sections


def normalize_law_reference_label(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    text = text.replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip("、,.;:：")
    match = LAW_NAME_PATTERN.search(text)
    candidate: Optional[str] = None
    if match:
        candidate = match.group("law")
    else:
        for keyword in LAW_NAME_KEYWORDS:
            if keyword in text:
                candidate = keyword
                break
        if candidate is None:
            for suffix in LAW_NAME_SUFFIXES:
                if text.endswith(suffix):
                    candidate = text
                    break
    if not candidate:
        return None
    candidate = candidate.replace("　", " ").strip()
    return candidate


def parse_reference_list(value: object) -> List[Dict[str, str]]:
    if isinstance(value, list):
        return [ref for ref in value if isinstance(ref, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [ref for ref in parsed if isinstance(ref, dict)]
    return []


def collect_law_reference_terms(
    row: pd.Series, analyzer: Optional[LawRevisionAnalyzer] = None
) -> List[str]:
    analyzer = analyzer or get_law_revision_analyzer()
    candidates: List[str] = []

    def append_candidate(value: object) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                append_candidate(item)
            return
        text = str(value).strip()
        if not text:
            return
        tokens = [tok.strip() for tok in re.split(r"[;／/,、]+", text) if tok.strip()]
        if not tokens:
            tokens = [text]
        for token in tokens:
            if token:
                candidates.append(token)
                parts = [part.strip() for part in re.split(r"\s+", token) if part.strip()]
                for part in parts:
                    if part != token:
                        candidates.append(part)

    tags_value = row.get("tags")
    append_candidate(tags_value)
    append_candidate(row.get("topic"))
    append_candidate(row.get("category"))
    append_candidate(row.get("law_name"))

    question_text = str(row.get("question", "") or "")
    append_candidate(question_text)
    if question_text:
        append_candidate(LAW_NAME_PATTERN.findall(question_text))

    if isinstance(tags_value, str):
        append_candidate(LAW_NAME_PATTERN.findall(tags_value))

    combined_parts: List[str] = []
    if question_text:
        combined_parts.append(question_text)
    if isinstance(tags_value, str):
        combined_parts.append(tags_value.replace(";", " "))
    combined_text = " ".join(part for part in combined_parts if part).strip()
    if combined_text:
        keywords = analyzer.extract_keywords(combined_text, limit=8)
        append_candidate(keywords)

    normalized: List[str] = []
    seen: Set[str] = set()
    for raw in candidates:
        normalized_label = normalize_law_reference_label(raw)
        if not normalized_label:
            continue
        if normalized_label in seen:
            continue
        seen.add(normalized_label)
        normalized.append(normalized_label)
    return normalized


def get_outline_insight(row: pd.Series) -> Dict[str, object]:
    cache: Dict[str, Dict[str, object]] = st.session_state.setdefault(OUTLINE_CACHE_KEY, {})
    question_id = str(row.get("id", "") or "")
    if question_id and question_id in cache:
        return cache[question_id]
    analyzer = get_law_revision_analyzer()
    question_text = str(row.get("question", "") or "")
    tags_text = str(row.get("tags", "") or "")
    combined = " ".join(
        part for part in [question_text, tags_text.replace(";", " ")] if part.strip()
    ).strip()
    summary = ""
    if combined:
        summary = analyzer.summarize(combined, max_sentences=2)
    if not summary:
        explanation = row.get("explanation")
        if pd.notna(explanation):
            explanation_summary, _ = parse_explanation_sections(str(explanation))
            summary = explanation_summary
    summary = (summary or "").strip()
    if len(summary) > 120:
        summary = summary[:117] + "…"
    law_terms = collect_law_reference_terms(row, analyzer=analyzer)
    references = [
        {
            "label": term,
            "url": LAW_REFERENCE_BASE_URL.format(query=quote_plus(term)),
        }
        for term in law_terms
    ]
    insight = {"summary": summary, "terms": law_terms, "references": references}
    if question_id:
        cache[question_id] = insight
    return insight


def render_explanation_content(row: pd.Series, db: Optional[DBManager] = None) -> None:
    explanation = row.get("explanation", "")
    explanation_summary, sections = parse_explanation_sections(explanation)
    outline = get_outline_insight(row)
    outline_summary = outline.get("summary") or explanation_summary
    references: List[Dict[str, str]] = outline.get("references", [])  # type: ignore[arg-type]

    if outline_summary or references:
        st.markdown("##### アウトラインサマリー")
        if outline_summary:
            st.write(outline_summary)
        else:
            st.caption("要約を生成できませんでした。")
        if references:
            st.markdown("###### 関連条文候補")
            for ref in references:
                label = ref.get("label")
                url = ref.get("url")
                if not label or not url:
                    continue
                st.markdown(f"- [{label}]({url})")

    saved_notes_df = pd.DataFrame()
    if db is not None:
        question_id = str(row.get("id", "") or "")
        label_value = str(row.get("label", "") or "").strip()
        if not label_value:
            year_display = format_year_value(row.get("year"))
            q_no_display = format_qno_value(row.get("q_no"))
            if year_display and q_no_display:
                label_value = f"{year_display} 問{q_no_display}"
            elif year_display:
                label_value = year_display
            elif q_no_display:
                label_value = f"問{q_no_display}"
            else:
                label_value = question_id
        tags_value = str(row.get("tags") or "") or None
        can_save = bool(outline_summary or references)
        if st.button(
            "アウトラインノートに保存",
            key=f"outline_save_{question_id}",
            disabled=not can_save,
            help="生成された要約と関連リンクをノートとして保存し、学習ログとあわせて振り返れます。"
            if can_save
            else "保存する内容がありません。",
        ):
            summary_to_save = outline_summary or explanation_summary or "要約なし"
            db.save_outline_note(
                question_id=question_id,
                summary=summary_to_save,
                references=references,
                question_label=label_value,
                tags=tags_value,
            )
            st.success("アウトラインノートに保存しました。")
        if question_id:
            saved_notes_df = db.fetch_outline_notes(question_id)
        if not saved_notes_df.empty:
            with st.expander("保存済みノート", expanded=False):
                for _, note in saved_notes_df.iterrows():
                    note_summary = str(note.get("summary", "") or "")
                    updated_at = note.get("updated_at")
                    timestamp = ""
                    if pd.notna(updated_at):
                        try:
                            timestamp = pd.to_datetime(updated_at).strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            timestamp = str(updated_at)
                    st.markdown(f"**{note_summary}**")
                    if timestamp:
                        st.caption(f"更新: {timestamp}")
                    note_refs = parse_reference_list(note.get("law_references"))
                    links = [
                        f"[{ref.get('label', '')}]({ref.get('url', '')})"
                        for ref in note_refs
                        if ref.get("label") and ref.get("url")
                    ]
                    if links:
                        st.caption(f"関連: {' / '.join(links)}")
                    note_tags = str(note.get("tags", "") or "").strip()
                    if note_tags:
                        st.caption(f"タグ: {note_tags}")

    if not explanation:
        st.write("解説が未登録です。『設定 ＞ データ入出力』から解答データを取り込みましょう。")
        return

    st.markdown(f"**要点版**：{explanation_summary}")
    with st.expander("詳細解説をひらく", expanded=False):
        for label, content in sections:
            if not content:
                continue
            if label == "ミニ図":
                st.markdown(f"**{label}**")
                st.markdown(content, unsafe_allow_html=True)
            else:
                st.markdown(f"- **{label}**：{content}")
        similar = compute_similarity(row["id"])
        if not similar.empty:
            st.markdown("#### 類似問題")
            st.dataframe(similar, use_container_width=True)
            questions_df = load_questions_df()
            similar_ids = [qid for qid in similar["id"] if pd.notna(qid)]
            if similar_ids:
                selected_similar_id = st.selectbox(
                    "類似問題をプレビュー",
                    similar_ids,
                    format_func=lambda x: format_question_label(questions_df, x),
                    key=f"similar_preview_{row['id']}",
                )
                preview_row = questions_df[questions_df["id"] == selected_similar_id]
                if preview_row.empty:
                    st.info("選択した問題がデータベースに見つかりません。")
                else:
                    render_question_preview(preview_row.iloc[0], db=db)


def estimate_theta(attempts: pd.DataFrame, df: pd.DataFrame) -> Optional[float]:
    if attempts.empty:
        return None
    merged = attempts.merge(
        df[["id", "difficulty"]], left_on="question_id", right_on="id", how="left"
    )
    merged = merged.dropna(subset=["difficulty"])
    if merged.empty:
        return None
    difficulties = (merged["difficulty"].astype(float) - 3.0) * 0.7
    responses = merged["is_correct"].astype(float)
    theta = 0.0
    for _ in range(10):
        logits = theta - difficulties
        probs = 1.0 / (1.0 + np.exp(-logits))
        gradient = np.sum(responses - probs)
        hessian = -np.sum(probs * (1 - probs))
        if abs(hessian) < 1e-6:
            break
        theta -= gradient / hessian
        if abs(gradient) < 1e-4:
            break
    return float(theta)


def recommend_adaptive_questions(
    df: pd.DataFrame,
    attempts: pd.DataFrame,
    theta: float,
    limit: int = 10,
    low_conf_threshold: int = 70,
) -> pd.DataFrame:
    candidates = df.copy()
    candidates["difficulty"] = candidates["difficulty"].fillna(DIFFICULTY_DEFAULT)
    candidates["difficulty_scaled"] = (candidates["difficulty"].astype(float) - 3.0) * 0.7
    candidates["priority"] = np.where(
        candidates["difficulty_scaled"] >= theta,
        candidates["difficulty_scaled"] - theta,
        (theta - candidates["difficulty_scaled"]) * 1.5,
    )
    if not attempts.empty:
        attempts["created_at"] = pd.to_datetime(attempts["created_at"])
        last_attempts = attempts.sort_values("created_at").groupby("question_id").tail(1)
        if "confidence" in last_attempts:
            confidence_series = last_attempts["confidence"].fillna(0)
        else:
            confidence_series = pd.Series(0, index=last_attempts.index)
        mastered_ids = last_attempts[
            (last_attempts["is_correct"] == 1)
            & (confidence_series >= low_conf_threshold)
        ]["question_id"].tolist()
        if mastered_ids:
            candidates = candidates[~candidates["id"].isin(mastered_ids)]
    ranked = candidates.sort_values(["priority", "difficulty"], ascending=[True, False])
    return ranked.head(limit)


def compute_tricky_vocab_heatmap(
    attempts: pd.DataFrame, df: pd.DataFrame, top_n: int = 12
) -> pd.DataFrame:
    wrong = attempts[attempts["is_correct"] == 0]
    if wrong.empty:
        return pd.DataFrame()
    merged = wrong.merge(
        df[["id", "question", "category", "tags"]],
        left_on="question_id",
        right_on="id",
        how="left",
    )
    records: List[Dict[str, object]] = []
    pattern = re.compile(r"[一-龠ぁ-んァ-ンA-Za-z0-9]{2,}")
    for _, row in merged.iterrows():
        text = f"{row.get('question', '')} {row.get('tags', '')}"
        words = {w for w in pattern.findall(str(text)) if len(w) >= 2}
        for word in list(words)[:20]:
            records.append({"word": word, "category": row.get("category", "未分類")})
    if not records:
        return pd.DataFrame()
    freq = pd.DataFrame(records)
    counts = freq.groupby(["word", "category"]).size().reset_index(name="count")
    totals = counts.groupby("word")["count"].sum().reset_index(name="total")
    top_words = totals.nlargest(top_n, "total")["word"]
    heatmap_df = counts[counts["word"].isin(top_words)]
    return heatmap_df


def compute_most_improved_topic(attempts: pd.DataFrame, df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if "topic" not in df.columns:
        return None
    merged = attempts.merge(
        df[["id", "topic"]], left_on="question_id", right_on="id", how="left"
    )
    if "topic" not in merged.columns:
        return None
    merged = merged.dropna(subset=["topic"])
    if merged.empty:
        return None
    improvements: List[Dict[str, object]] = []
    for topic, group in merged.groupby("topic"):
        if len(group) < 4:
            continue
        group = group.sort_values("created_at")
        window = max(1, len(group) // 3)
        early = group.head(window)["is_correct"].mean()
        late = group.tail(window)["is_correct"].mean()
        improvements.append(
            {
                "topic": topic,
                "delta": late - early,
                "early": early,
                "late": late,
                "attempts": len(group),
            }
        )
    if not improvements:
        return None
    best = max(improvements, key=lambda x: x["delta"])
    if best["delta"] <= 0:
        return None
    return best


def build_notion_summaries(attempts: pd.DataFrame, days: int = 7) -> List[Dict[str, object]]:
    if attempts.empty:
        return []
    working = attempts.copy()
    working["created_at"] = pd.to_datetime(working["created_at"])
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    working = working[working["created_at"] >= cutoff]
    if working.empty:
        return []
    working["is_correct"] = working["is_correct"].fillna(0).astype(int)
    summary = (
        working
        .groupby(working["created_at"].dt.date)
        .agg(
            attempts=("question_id", "count"),
            correct=("is_correct", "sum"),
            avg_seconds=("seconds", "mean"),
        )
        .reset_index()
        .rename(columns={"created_at": "date"})
    )
    summary["accuracy"] = summary.apply(
        lambda row: (row["correct"] / row["attempts"]) if row["attempts"] else 0.0,
        axis=1,
    )
    summary["avg_seconds"] = summary["avg_seconds"].fillna(0.0)
    results: List[Dict[str, object]] = []
    for row in summary.itertuples(index=False):
        date_value = row.date if hasattr(row, "date") else row[0]
        if isinstance(date_value, dt.datetime):
            date_value = date_value.date()
        results.append(
            {
                "date": date_value,
                "attempts": int(row.attempts),
                "accuracy": float(row.accuracy),
                "avg_seconds": float(row.avg_seconds),
            }
        )
    return results


def parse_external_attempt_logs(
    file: "UploadedFile", questions_df: pd.DataFrame
) -> Tuple[List[Dict[str, object]], List[str]]:
    suffix = Path(file.name).suffix.lower()
    raw_bytes = file.getvalue()
    errors: List[str] = []
    if not raw_bytes:
        return [], [f"{file.name}: ファイルが空です。"]
    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            return [], [f"{file.name}: UTF-8でデコードできませんでした。"]
    if suffix == ".csv":
        data_df = pd.read_csv(io.StringIO(decoded))
    elif suffix == ".json":
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            return [], [f"{file.name}: JSONの読み込みに失敗しました ({exc})"]
        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                payload = payload["records"]
            elif "data" in payload and isinstance(payload["data"], list):
                payload = payload["data"]
            else:
                payload = [payload]
        data_df = pd.DataFrame(payload)
    else:
        return [], [f"{file.name}: 対応していない形式です。CSVまたはJSONを使用してください。"]
    if data_df.empty:
        return [], [f"{file.name}: 取り込み可能なデータがありません。"]
    normalized_columns = {col.lower(): col for col in data_df.columns}
    if "seconds" not in normalized_columns:
        return [], [f"{file.name}: seconds 列が必要です。"]
    seconds_col = normalized_columns["seconds"]
    question_id_col = normalized_columns.get("question_id")
    timestamp_col = normalized_columns.get("timestamp") or normalized_columns.get("created_at")
    year_col = normalized_columns.get("year")
    qno_col = normalized_columns.get("q_no") or normalized_columns.get("qno")
    topic_col = normalized_columns.get("topic")
    mode_col = normalized_columns.get("mode")
    selected_col = normalized_columns.get("selected")
    correct_col = normalized_columns.get("is_correct")
    confidence_col = normalized_columns.get("confidence")
    grade_col = normalized_columns.get("grade")

    resolved: List[Dict[str, object]] = []
    question_index = questions_df.set_index("id") if "id" in questions_df.columns else pd.DataFrame()
    for idx, row in data_df.iterrows():
        question_id = None
        if question_id_col and pd.notna(row.get(question_id_col)):
            candidate = str(row.get(question_id_col)).strip()
            if not question_index.empty and candidate in question_index.index:
                question_id = candidate
        if not question_id and year_col and qno_col and pd.notna(row.get(year_col)) and pd.notna(row.get(qno_col)):
            try:
                year_val = int(row.get(year_col))
                qno_val = int(row.get(qno_col))
            except (TypeError, ValueError):
                year_val = qno_val = None
            if year_val is not None and qno_val is not None:
                matches = questions_df[
                    (questions_df["year"] == year_val) & (questions_df["q_no"] == qno_val)
                ]
                if not matches.empty:
                    question_id = matches.iloc[0]["id"]
        if not question_id and topic_col and pd.notna(row.get(topic_col)):
            topic_val = str(row.get(topic_col)).strip()
            matches = questions_df[questions_df["topic"] == topic_val]
            if not matches.empty:
                question_id = matches.iloc[0]["id"]
        if not question_id:
            errors.append(f"{file.name} 行{idx + 1}: 対応する問題IDを特定できませんでした。")
            continue
        try:
            seconds = int(float(row.get(seconds_col, 0)))
        except (TypeError, ValueError):
            errors.append(f"{file.name} 行{idx + 1}: seconds 列が数値ではありません。")
            continue
        timestamp_value = row.get(timestamp_col) if timestamp_col else None
        if pd.notna(timestamp_value):
            try:
                created_at = pd.to_datetime(timestamp_value)
            except Exception:
                created_at = pd.Timestamp.utcnow()
        else:
            created_at = pd.Timestamp.utcnow()
        payload: Dict[str, object] = {
            "question_id": question_id,
            "seconds": seconds,
            "mode": str(row.get(mode_col, "external_log")) if mode_col else "external_log",
            "created_at": created_at.to_pydatetime() if hasattr(created_at, "to_pydatetime") else created_at,
        }
        if selected_col and pd.notna(row.get(selected_col)):
            try:
                payload["selected"] = int(row.get(selected_col))
            except (TypeError, ValueError):
                payload["selected"] = None
        if correct_col and pd.notna(row.get(correct_col)):
            value = row.get(correct_col)
            try:
                payload["is_correct"] = int(bool(int(value)))
            except (TypeError, ValueError):
                payload["is_correct"] = None
        if confidence_col and pd.notna(row.get(confidence_col)):
            try:
                payload["confidence"] = int(float(row.get(confidence_col)))
            except (TypeError, ValueError):
                payload["confidence"] = None
        if grade_col and pd.notna(row.get(grade_col)):
            try:
                payload["grade"] = int(float(row.get(grade_col)))
            except (TypeError, ValueError):
                payload["grade"] = None
        resolved.append(payload)
    return resolved, errors


def register_keyboard_shortcuts(mapping: Dict[str, str]) -> None:
    if not mapping:
        return
    st_html(
        """
        <script>
        (function() {
            const mapping = %s;
            document.addEventListener('keydown', function(event) {
                const active = document.activeElement;
                if (active && ['input', 'textarea', 'select'].includes(active.tagName.toLowerCase())) {
                    return;
                }
                const key = event.key ? event.key.toLowerCase() : '';
                const label = mapping[key];
                if (!label) {
                    return;
                }
                const doc = window.parent ? window.parent.document : document;
                const buttons = doc.querySelectorAll('button');
                for (const btn of buttons) {
                    if (!btn.innerText) {
                        continue;
                    }
                    if (btn.innerText.trim().startsWith(label.trim())) {
                        event.preventDefault();
                        btn.click();
                        break;
                    }
                }
            }, true);
        })();
        </script>
        """ % json.dumps({k.lower(): v for k, v in mapping.items()}, ensure_ascii=False),
        height=0,
    )
def decode_uploaded_file(file: "UploadedFile") -> List[Tuple[str, pd.DataFrame]]:
    filename = file.name
    suffix = Path(filename).suffix.lower()
    dataframes = []
    bytes_data = file.getvalue()
    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(bytes_data)) as z:
            for inner in z.infolist():
                if inner.is_dir():
                    continue
                inner_suffix = Path(inner.filename).suffix.lower()
                with z.open(inner) as f:
                    df = read_tabular(f.read(), inner_suffix)
                    dataframes.append((inner.filename, df))
    else:
        df = read_tabular(bytes_data, suffix)
        dataframes.append((filename, df))
    return dataframes


def read_tabular(data: bytes, suffix: str) -> pd.DataFrame:
    encoding_options = ["utf-8", "cp932"]
    if suffix in [".csv", ".txt", ".tsv", ""]:
        for enc in encoding_options:
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(data))
    elif suffix in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(io.BytesIO(data))
    else:
        raise ValueError("サポートされていないファイル形式です")


def guess_dataset_kind(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    if {"choice1", "choice2", "choice3", "choice4"}.issubset(cols):
        return MAPPING_KIND_QUESTIONS
    if "correct_number" in cols or "correct_label" in cols or "correct_text" in cols:
        return MAPPING_KIND_ANSWERS
    return MAPPING_KIND_QUESTIONS


def describe_dataset_name(raw_name: str) -> Tuple[str, Optional[str]]:
    base_name = Path(raw_name).name
    stem = Path(base_name).stem or base_name
    suffix = Path(base_name).suffix.lower()
    normalized = stem.strip() or base_name
    lower = normalized.lower()
    name_map = {
        "questions": "設問データ",
        "answers": "正答データ",
        "predicted": "予想問題データ",
        "law_revision": "法改正予想問題データ",
    }
    if lower in name_map:
        return name_map[lower], base_name
    hint: Optional[str] = None
    if suffix in {".csv", ".tsv", ".txt"}:
        hint = "CSVデータ"
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        hint = "Excelシート"
    elif any(keyword in normalized for keyword in ["テーブル", "table", "Table"]):
        hint = "表データ"
    elif any(keyword in normalized for keyword in ["グラフ", "chart", "Chart"]):
        hint = "グラフデータ"
    if hint:
        return f"{normalized}（{hint}）", base_name
    return normalized, base_name if normalized != base_name else None


def store_uploaded_file(file: "UploadedFile", timestamp: str) -> Path:
    target_dir = UPLOAD_DIR / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / file.name
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path


def init_session_state() -> None:
    defaults = {
        "nav": "ホーム",
        "current_question": None,
        "attempt_start": None,
        "exam_session": None,
        "import_state": {},
        "settings": {
            "shuffle_choices": True,
            "theme": "セピア",
            "font_size": "標準",
            "timer": True,
            "sm2_initial_ease": 2.5,
            "auto_advance": False,
            "review_low_confidence_threshold": 60,
            "review_elapsed_days": 7,
            "integrations": {
                "google_calendar": {
                    "client_id": "",
                    "client_secret": "",
                    "redirect_uri": "",
                    "access_token": "",
                    "refresh_token": "",
                    "calendar_id": "primary",
                },
                "notion": {
                    "integration_token": "",
                    "database_id": "",
                    "notion_version": "2022-06-28",
                },
            },
        },
        "_nav_widget": "ホーム",
        "integration_status": {
            "google_calendar": {
                "last_synced": None,
                "message": "未同期",
                "success": False,
            },
            "notion": {
                "last_synced": None,
                "message": "未同期",
                "success": False,
            },
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="宅建10年ドリル", layout="wide")
    init_session_state()
    apply_user_preferences()
    engine = get_engine()
    db = DBManager(engine)
    db.initialize_from_csv()
    df = load_questions_df()

    sidebar = st.sidebar
    sidebar.title("宅建10年ドリル")
    if st.session_state.get("_nav_widget") != st.session_state.get("nav"):
        st.session_state["_nav_widget"] = st.session_state.get("nav", "ホーム")
    menu_options = ["ホーム", "学習", "模試", "統計", "設定"]
    current_nav = st.session_state.get("nav", "ホーム")
    if current_nav not in menu_options:
        current_nav = "ホーム"
        st.session_state["nav"] = current_nav
        st.session_state["_nav_widget"] = current_nav
    sidebar.radio(
        "メニュー",
        menu_options,
        index=menu_options.index(current_nav),
        key="_nav_widget",
        on_change=with_rerun(handle_nav_change),
    )
    nav = st.session_state.get("nav", "ホーム")
    sidebar.divider()
    with sidebar.expander("モード別の使い方ガイド", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- **ホーム**：進捗サマリーと最近のインポート履歴を確認できます。",
                    "- **学習**：演習プラン・特別対策・弱点ケアのタブから目的に応じて学習モードを選択します。",
                    "- **模試**：年度や出題方式を指定して本番同様の模試を開始します。",
                    "- **統計**：分野別の成績や時間分析を把握できます。",
                    "- **設定**：表示設定の調整と『設定 ＞ データ入出力』タブでのCSV/ZIP取り込みをまとめています。",
                ]
            )
        )

    if nav == "ホーム":
        render_home(db, df)
    elif nav == "学習":
        render_learning(db, df)
    elif nav == "模試":
        render_mock_exam(db, df)
    elif nav == "統計":
        render_stats(db, df)
    elif nav == "設定":
        render_settings(db)


def render_home(db: DBManager, df: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
        .home-dropzone-card {
            background: rgba(64, 138, 255, 0.08);
            border: 1px dashed rgba(64, 138, 255, 0.35);
            border-radius: 1rem;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.2rem;
        }
        .home-dropzone-card strong {
            display: block;
            font-size: 1.05rem;
            margin-bottom: 0.25rem;
        }
        .home-data-card {
            background: rgba(0, 0, 0, 0.03);
            border: 1px solid rgba(49, 51, 63, 0.08);
            border-radius: 0.9rem;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("ホーム")
    dropzone_files = st.file_uploader(
        "questions.csv / answers.csv をドラッグ＆ドロップ",
        type=["csv"],
        accept_multiple_files=True,
        key="home_dropzone",
        label_visibility="collapsed",
        help="ページ全体がドロップゾーンとして機能します。questions.csv と answers.csv をまとめて追加できます。",
    )
    st.markdown(
        """
        <div class="home-dropzone-card">
            <strong>CSVをドラッグ＆ドロップ</strong>
            ページ全体がドロップゾーンとして機能します。ここに追加した questions.csv / answers.csv は下のクイックインポートに自動で反映されます。
        </div>
        """,
        unsafe_allow_html=True,
    )

    attempts = db.get_attempt_stats()
    st.markdown("### サマリー")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("設問数", len(df))
    with col2:
        st.metric("学習履歴", len(attempts))
    with col3:
        coverage = attempts["year"].nunique() / max(df["year"].nunique(), 1) * 100 if not attempts.empty else 0
        st.metric("年度カバレッジ", f"{coverage:.0f}%")

    st.markdown(
        """
        <div class="home-data-card">
            <strong>データの取り込みについて</strong><br>
            上部のドロップゾーンに CSV を配置すると、このページの「データ入出力」で即座にクイックインポートできます。列マッピングや詳細設定は『設定 ＞ データ入出力』で従来通り調整できます。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### データ入出力")
    st.caption("questions.csv / answers.csv の読み込みと学習履歴のエクスポートをホームから直接操作できます。")
    render_quick_import_controls(
        db,
        key_prefix="home",
        heading="#### クイックインポート (questions.csv / answers.csv)",
        initial_files=dropzone_files or None,
    )
    render_history_export_controls(
        db,
        heading="#### 学習履歴のエクスポート",
    )

    st.markdown("### 最近のインポート")
    with db.engine.connect() as conn:
        logs = pd.read_sql(select(import_logs_table).order_by(import_logs_table.c.id.desc()).limit(5), conn)
    if logs.empty:
        st.write("インポート履歴がありません。")
    else:
        st.dataframe(logs)


def render_learning(db: DBManager, df: pd.DataFrame) -> None:
    st.title("学習")
    if df.empty:
        st.warning("設問データがありません。『設定 ＞ データ入出力』からアップロードしてください。")
        return
    primary_tabs = st.tabs(["演習プラン", "特別対策", "弱点ケア"])
    with primary_tabs[0]:
        plan_tabs = st.tabs(["本試験モード", "適応学習", "分野別ドリル", "年度別演習"])
        with plan_tabs[0]:
            render_full_exam_lane(db, df)
        with plan_tabs[1]:
            render_adaptive_lane(db, df)
        with plan_tabs[2]:
            render_subject_drill_lane(db, df)
        with plan_tabs[3]:
            render_year_drill_lane(db, df)
    with primary_tabs[1]:
        special_tabs = st.tabs(["法改正対策", "予想問題演習"])
        with special_tabs[0]:
            render_law_revision_lane(db, parent_nav="学習")
        with special_tabs[1]:
            render_predicted_lane(db, parent_nav="学習")
    with primary_tabs[2]:
        review_tabs = st.tabs(["弱点分析", "SRS復習"])
        with review_tabs[0]:
            render_weakness_lane(db, df)
        with review_tabs[1]:
            render_srs(db, parent_nav="学習")
    st.divider()
    render_outline_notes_overview(db, df)


def render_full_exam_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("本試験モード")
    st.caption("50問・120分の本試験同等環境で得点力と時間配分をチェックします。")
    if len(df) < 50:
        st.info("50問の出題には最低50問のデータが必要です。データを追加してください。")
        return
    attempts = db.get_attempt_stats()
    priority_df = build_question_priority(df, attempts)
    weight_map = dict(zip(priority_df["id"], priority_df["weight"]))
    if not attempts.empty:
        category_focus = (
            priority_df.groupby("category")["weight"].mean().sort_values(ascending=False)
        )
        if not category_focus.empty:
            highlights = [
                f"{category} (優先度 {score:.2f})"
                for category, score in category_focus.head(2).items()
            ]
            st.caption(
                "最近の弱点傾向を踏まえ、以下の分野が優先的に出題されます: "
                + "、".join(highlights)
            )
    session: Optional[ExamSession] = st.session_state.get("exam_session")
    error_key = "_full_exam_error"
    error_message = st.session_state.pop(error_key, None)
    if error_message:
        st.warning(error_message)

    def start_full_exam_session() -> None:
        questions = stratified_exam(df, weight_map if weight_map else None)
        if not questions:
            st.session_state[error_key] = "出題可能な問題が不足しています。データを確認してください。"
            return
        st.session_state.pop("exam_result_本試験モード", None)
        st.session_state["exam_session"] = ExamSession(
            id=None,
            name=f"本試験モード {dt.datetime.now():%Y%m%d-%H%M}",
            questions=questions,
            started_at=dt.datetime.now(),
            year_mode="層化ランダム50",
            mode="本試験モード",
        )

    if session is None or session.mode != "本試験モード":
        st.button(
            "50問模試を開始",
            key="start_full_exam",
            help="本試験と同じ50問・120分構成で一気に演習します。結果は統計に反映されます。",
            on_click=with_rerun(start_full_exam_session),
        )
    session = st.session_state.get("exam_session")
    if session and session.mode == "本試験モード":
        render_exam_session_body(db, df, session, key_prefix="main_exam")
    result = st.session_state.get("exam_result_本試験モード")
    if result:
        display_exam_result(result)


def render_outline_notes_overview(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("アウトラインノート")
    notes_df = db.fetch_outline_notes()
    if notes_df.empty:
        st.info("アウトラインノートはまだ保存されていません。解説画面の保存ボタンから作成できます。")
        return
    notes_df = notes_df.copy()
    notes_df["updated_at"] = pd.to_datetime(notes_df.get("updated_at"), errors="coerce")
    attempts = db.get_attempt_stats()
    if not attempts.empty and "question_id" in attempts.columns:
        attempt_counts = attempts.groupby("question_id").size().rename("attempts")
        notes_df = notes_df.merge(
            attempt_counts,
            left_on="question_id",
            right_index=True,
            how="left",
        )
    else:
        notes_df["attempts"] = 0
    notes_df["attempts"] = notes_df["attempts"].fillna(0).astype(int)
    meta_cols = ["id", "year", "q_no", "category", "topic"]
    available_meta = [col for col in meta_cols if col in df.columns]
    if available_meta:
        metadata = df[available_meta].copy()
        metadata = metadata.rename(columns={"id": "question_id"})
        notes_df = notes_df.merge(metadata, on="question_id", how="left")
    for missing in ["year", "q_no", "category", "topic"]:
        if missing not in notes_df.columns:
            notes_df[missing] = pd.NA
    notes_df["links_display"] = notes_df["law_references"].apply(
        lambda value: " / ".join(
            ref.get("label", "")
            for ref in parse_reference_list(value)
            if ref.get("label")
        )
    )
    notes_df = notes_df.sort_values("updated_at", ascending=False)
    notes_df["updated_at_display"] = notes_df["updated_at"].dt.strftime("%Y-%m-%d %H:%M")
    display_columns = {
        "question_id": "設問ID",
        "year": "年度",
        "q_no": "問番号",
        "category": "分野",
        "topic": "論点",
        "summary": "要約",
        "links_display": "関連リンク",
        "attempts": "学習回数",
        "updated_at_display": "更新日時",
    }
    display_df = notes_df[list(display_columns.keys())].rename(columns=display_columns)
    st.dataframe(display_df, use_container_width=True)
    st.caption("学習履歴の取り組み回数を併記しています。ノートとログの整合を確認できます。")
    export_df = notes_df.copy()
    export_df["law_references"] = export_df["law_references"].apply(
        lambda value: json.dumps(parse_reference_list(value), ensure_ascii=False)
    )
    export_df["updated_at"] = export_df["updated_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    buffer = io.StringIO()
    export_df.to_csv(buffer, index=False)
    st.download_button(
        "アウトラインノートをCSVエクスポート",
        data=buffer.getvalue(),
        file_name="outline_notes.csv",
        mime="text/csv",
    )


def render_adaptive_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("適応学習")
    st.caption("回答履歴から能力θを推定し、伸びしろの大きい難度を優先出題します。")
    attempts = db.get_attempt_stats()
    if attempts.empty:
        st.info("学習履歴がまだありません。本試験モードやドリルで取り組んでみましょう。")
        return
    theta = estimate_theta(attempts, df)
    if theta is None:
        st.info("推定に必要な難易度データが不足しています。問題に難易度を設定してください。")
        return
    metric_col, help_col = st.columns([1, 2])
    metric_col.metric("推定能力θ", f"{theta:.2f}")
    help_col.caption(
        "回答の正誤・難易度・最新の取り組み履歴を元に推定しています。"
        " 新しい回答を記録するたびに自動で更新されます。"
    )
    low_conf = int(st.session_state["settings"].get("review_low_confidence_threshold", 60))
    recommended = recommend_adaptive_questions(df, attempts, theta, low_conf_threshold=low_conf)
    if recommended.empty:
        st.info("おすすめできる問題がありません。条件を見直すか、新しい問題を追加してください。")
        return
    st.markdown("#### 推奨問題 (上位10件)")
    recommended_ids = recommended["id"].tolist()
    session_key = "adaptive_question_select"
    if not recommended_ids:
        st.warning("推奨問題の一覧を表示できませんでした。")
        return
    if session_key not in st.session_state or st.session_state[session_key] not in recommended_ids:
        st.session_state[session_key] = recommended_ids[0]
    selected_id = st.session_state[session_key]

    def select_adaptive_question(question_id: str) -> None:
        st.session_state[session_key] = question_id

    for _, rec in recommended.iterrows():
        qid = rec["id"]
        label = format_question_label(df, qid)
        difficulty_value = rec.get("difficulty")
        if pd.isna(difficulty_value):
            difficulty_display = "不明"
        elif isinstance(difficulty_value, (int, np.integer)):
            difficulty_display = str(int(difficulty_value))
        else:
            difficulty_display = f"{float(difficulty_value):.1f}"
        priority_value = rec.get("priority")
        priority_display = f"{priority_value:.2f}" if pd.notna(priority_value) else "N/A"
        button_label = f"{label}｜推奨度 {priority_display}｜難易度 {difficulty_display}"
        button_type = "primary" if qid == selected_id else "secondary"
        if st.button(
            button_label,
            key=f"adaptive_jump_{qid}",
            type=button_type,
            help="クリックすると下の詳細と出題画面がその問題に切り替わります。",
        ):
            select_adaptive_question(qid)
            selected_id = qid

    with st.expander("推奨問題の詳細一覧", expanded=False):
        display = recommended[["id", "year", "q_no", "category", "difficulty", "priority"]].copy()
        render_table_with_category_tags(
            display,
            column_order=["id", "category", "year", "q_no", "difficulty", "priority"],
            rename_map={
                "id": "問題ID",
                "category": "分野",
                "year": "年度",
                "q_no": "問番",
                "difficulty": "難易度",
                "priority": "推奨度",
            },
            index_column="id",
        )

    selected_id = st.selectbox(
        "取り組む問題",
        recommended["id"],
        format_func=lambda x: format_question_label(df, x),
        key="adaptive_question_select",
    )
    row = df[df["id"] == selected_id].iloc[0]
    render_question_interaction(db, row, attempt_mode="adaptive", key_prefix="adaptive")


def render_subject_drill_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("分野別ドリル")
    st.caption(
        "民法・借地借家法・都市計画法・建築基準法・税・鑑定評価・宅建業法といったテーマをピンポイントで鍛えます。"
        " 履歴に基づく優先度スコアを参照して弱点強化にも対応します。"
    )
    attempts = db.get_attempt_stats()
    priority_df = build_question_priority(df, attempts)
    priority_columns = priority_df[
        [
            "id",
            "weight",
            "accuracy",
            "last_confidence",
            "days_since_last_attempt",
            "attempts_count",
        ]
    ].rename(columns={"weight": "priority_score"})
    settings = st.session_state.get("settings", {})
    default_confidence = int(settings.get("review_low_confidence_threshold", 60))
    default_elapsed = int(settings.get("review_elapsed_days", 7))
    mode_options = ["手動選択"]
    if not attempts.empty:
        mode_options.append("弱点優先")
    mode = st.radio(
        "出題モード",
        mode_options,
        horizontal=True,
        key="subject_mode",
        help="履歴から算出した優先度をもとに手動選択または自動キューを切り替えます。",
    )
    with st.expander("出題条件", expanded=True):
        preset = st.selectbox(
            "クイックプリセット",
            list(SUBJECT_PRESETS.keys()),
            help="代表的な絞り込み条件をワンクリックで適用できます。",
            key="subject_preset",
        )
        def apply_subject_preset() -> None:
            config = SUBJECT_PRESETS[preset]
            st.session_state["subject_categories"] = config["categories"]
            st.session_state["subject_difficulty"] = config["difficulty"]
            st.session_state["subject_review_only"] = config["review_only"]
            st.session_state["subject_topics"] = config.get("topics", [])
            st.session_state["subject_keyword"] = ""

        st.button(
            "プリセットを適用",
            key="subject_apply_preset",
            on_click=with_rerun(apply_subject_preset),
        )
        categories = st.multiselect(
            "分野",
            CATEGORY_CHOICES,
            default=CATEGORY_CHOICES,
            key="subject_categories",
        )
        topic_options = sorted({t for t in df["topic"].dropna().unique() if str(t).strip()})
        selected_topics = st.multiselect(
            "テーマ",
            topic_options,
            default=[],
            key="subject_topics",
        )
        difficulties = st.slider(
            "難易度",
            1,
            5,
            (1, 5),
            key="subject_difficulty",
            help="1は易しい〜5は難しい問題です。",
        )
        keyword = st.text_input(
            "キーワードで絞り込み (問題文/タグ)",
            key="subject_keyword",
            help="語句を入力すると問題文とタグから部分一致で検索します。",
        )
        review_only = st.checkbox(
            "復習だけ表示 (誤答・低確信・経過日数)",
            value=st.session_state.get("subject_review_only", False),
            key="subject_review_only",
        )
    filtered = df[
        df["category"].isin(categories)
        & df["difficulty"].between(difficulties[0], difficulties[1])
    ]
    if selected_topics:
        filtered = filtered[filtered["topic"].isin(selected_topics)]
    if keyword:
        keyword_lower = keyword.lower()
        filtered = filtered[
            filtered["question"].str.lower().str.contains(keyword_lower)
            | filtered["tags"].fillna("").str.lower().str.contains(keyword_lower)
        ]
    if review_only:
        review_ids = get_review_candidate_ids(db)
        if not review_ids:
            st.info("復習対象の問題はありません。学習履歴を増やしてみましょう。")
            return
        filtered = filtered[filtered["id"].isin(review_ids)]
    if filtered.empty:
        st.warning("条件に合致する問題がありません。フィルタを調整してください。")
        return
    filtered = filtered.merge(priority_columns, on="id", how="left")
    filtered["priority_score"] = filtered["priority_score"].fillna(0.0)
    filtered["accuracy"] = filtered["accuracy"].fillna(0.5)
    filtered["last_confidence"] = filtered["last_confidence"].fillna(float(default_confidence))
    filtered["days_since_last_attempt"] = filtered["days_since_last_attempt"].fillna(float(default_elapsed))
    filtered["attempts_count"] = filtered["attempts_count"].fillna(0).astype(int)
    st.caption(
        f"現在の条件に合致する問題は {len(filtered)} 件です。"
        "優先度スコアは正答率・確信度・経過日数を組み合わせて算出しています。"
    )
    with st.expander("優先度付き一覧", expanded=False):
        display = filtered[[
            "id",
            "category",
            "difficulty",
            "priority_score",
            "accuracy",
            "last_confidence",
            "days_since_last_attempt",
            "attempts_count",
        ]].copy()
        display["accuracy"] = (display["accuracy"] * 100).round(0)
        display["last_confidence"] = display["last_confidence"].round(0)
        display["days_since_last_attempt"] = display["days_since_last_attempt"].round(1)
        render_table_with_category_tags(
            display,
            column_order=[
                "id",
                "category",
                "difficulty",
                "priority_score",
                "accuracy",
                "last_confidence",
                "days_since_last_attempt",
                "attempts_count",
            ],
            rename_map={
                "id": "問題ID",
                "category": "分野",
                "difficulty": "難易度",
                "priority_score": "優先度",
                "accuracy": "直近正答率(%)",
                "last_confidence": "直近確信度(%)",
                "days_since_last_attempt": "経過日数",
                "attempts_count": "挑戦回数",
            },
            index_column="id",
        )
    if mode == "弱点優先":
        prioritized = filtered.sort_values(
            ["priority_score", "accuracy"], ascending=[False, True]
        ).reset_index(drop=True)
        signature = (
            tuple(sorted(categories)),
            tuple(sorted(selected_topics)),
            tuple(difficulties),
            keyword,
            bool(review_only),
        )
        queue_key = "subject_priority_queue"
        signature_key = "subject_priority_signature"
        if st.session_state.get(signature_key) != signature:
            st.session_state[signature_key] = signature
            st.session_state[queue_key] = prioritized["id"].tolist()
        queue: List[str] = st.session_state.get(queue_key, [])
        if not queue:
            st.info("優先度条件に合致する問題がありません。フィルタを見直してください。")
            return
        current_id = queue[0]
        current_row = prioritized[prioritized["id"] == current_id].iloc[0]
        summary_parts = [f"優先度 {current_row['priority_score']:.2f}"]
        if not pd.isna(current_row.get("accuracy")):
            summary_parts.append(f"正答率 {current_row['accuracy'] * 100:.0f}%")
        if not pd.isna(current_row.get("last_confidence")):
            summary_parts.append(f"確信度 {current_row['last_confidence']:.0f}%")
        if not pd.isna(current_row.get("days_since_last_attempt")):
            summary_parts.append(
                f"経過 {current_row['days_since_last_attempt']:.0f}日"
            )
        st.info(" / ".join(summary_parts))

        def advance_priority_queue() -> None:
            queue_inner = st.session_state.get(queue_key, [])
            if queue_inner:
                queue_inner.pop(0)
            st.session_state[queue_key] = queue_inner
            safe_rerun()

        with st.expander("優先出題キュー", expanded=False):
            preview = prioritized.head(10)[["id", "priority_score", "accuracy"]]
            preview["accuracy"] = (preview["accuracy"] * 100).round(0)
            st.dataframe(
                preview.rename(
                    columns={
                        "id": "問題ID",
                        "priority_score": "優先度",
                        "accuracy": "正答率(%)",
                    }
                ).set_index("問題ID"),
                use_container_width=True,
            )
        current_row = filtered[filtered["id"] == current_id].iloc[0]
        render_question_interaction(
            db,
            current_row,
            attempt_mode="subject_drill",
            key_prefix="subject",
        )
        st.button(
            "次の優先問題に進む",
            key="subject_priority_next",
            help="キューから現在の問題を除外し、次の優先問題を表示します。",
            on_click=with_rerun(advance_priority_queue),
        )
        return

    def format_priority_label(question_id: str) -> str:
        label = format_question_label(filtered, question_id)
        row = filtered[filtered["id"] == question_id]
        if row.empty:
            return label
        score = row.iloc[0].get("priority_score")
        if pd.notna(score):
            return f"{label}｜優先度 {score:.2f}"
        return label

    question_id = st.selectbox(
        "出題問題",
        filtered["id"],
        format_func=format_priority_label,
        key="subject_question_select",
    )
    row = filtered[filtered["id"] == question_id].iloc[0]
    render_question_interaction(db, row, attempt_mode="subject_drill", key_prefix="subject")


def render_year_drill_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("年度別演習")
    st.caption("年度ごとの出題を通し演習し、本試験本番と同じ流れで知識を定着させます。")
    years = sorted(df["year"].unique(), reverse=True)
    if not years:
        st.info("年度情報が登録されていません。データを確認してください。")
        return
    selected_year = st.selectbox("年度", years, key="year_drill_year")
    subset = df[df["year"] == selected_year].sort_values("q_no")
    if subset.empty:
        st.warning("選択した年度の問題がありません。")
        return
    total = len(subset)
    progress_key = "year_drill_index"
    stored_year_key = "year_drill_current_year"
    if st.session_state.get(stored_year_key) != selected_year:
        st.session_state[stored_year_key] = selected_year
        st.session_state[progress_key] = 0
    index = st.session_state.get(progress_key, 0)
    index = max(0, min(index, total - 1))
    current_row = subset.iloc[index]
    st.progress((index + 1) / total)

    def go_prev() -> None:
        st.session_state[progress_key] = max(0, index - 1)

    def go_next() -> None:
        st.session_state[progress_key] = min(total - 1, index + 1)

    navigation = QuestionNavigation(
        has_prev=index > 0,
        has_next=index < total - 1,
        on_prev=go_prev,
        on_next=go_next,
        label=f"{index + 1}/{total} 問を学習中",
    )
    render_question_interaction(
        db,
        current_row,
        attempt_mode="year_drill",
        key_prefix="year",
        navigation=navigation,
    )


def render_weakness_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("弱点克服モード")
    st.caption("誤答・低正答率・時間超過が目立つ問題を優先的に出題し、得点の底上げを図ります。")
    attempts = db.get_attempt_stats()
    if attempts.empty:
        st.info("学習履歴がまだありません。本試験モードやドリルで取り組んでみましょう。")
        return
    summary = (
        attempts.groupby(["question_id", "category"])
        .agg(
            attempts_count=("is_correct", "count"),
            correct_count=("is_correct", "sum"),
            avg_seconds=("seconds", "mean"),
        )
        .reset_index()
    )
    summary["accuracy"] = summary["correct_count"] / summary["attempts_count"].replace(0, np.nan)
    summary["accuracy"] = summary["accuracy"].fillna(0)
    summary["avg_seconds"] = summary["avg_seconds"].fillna(0)
    summary["priority"] = (1 - summary["accuracy"]) * summary["attempts_count"] + np.where(
        summary["avg_seconds"] > 90,
        1,
        0,
    )
    merged = summary.merge(df[["id", "year", "q_no", "question"]], left_on="question_id", right_on="id", how="left")
    merged = merged.sort_values(["priority", "accuracy"], ascending=[False, True])
    st.markdown("#### 優先出題リスト")
    with st.expander("並び替え・フィルタ", expanded=False):
        category_options = sorted({str(cat) for cat in merged["category"].dropna()})
        selected_categories = st.multiselect(
            "分野",
            category_options,
            default=category_options,
            help="重点的に復習したい分野を選びます。",
            key="weakness_categories",
        )
        max_attempts = int(merged["attempts_count"].max()) if not merged.empty else 1
        min_attempts = int(merged["attempts_count"].min()) if not merged.empty else 0
        if min_attempts == max_attempts:
            attempts_threshold = max_attempts
            st.caption(f"挑戦回数フィルタ: {max_attempts} 回のみのデータです。")
        else:
            attempts_threshold = st.slider(
                "最低挑戦回数",
                min_attempts,
                max_attempts,
                min(min_attempts + 1, max_attempts),
                help="指定回数以上取り組んだ問題を対象にします。",
                key="weakness_attempts_threshold",
            )
        accuracy_ceiling = st.slider(
            "正答率の上限 (%)",
            0,
            100,
            70,
            step=5,
            help="この値より正答率が高い問題はリストから除外します。",
            key="weakness_accuracy_ceiling",
        )
        sort_option = st.selectbox(
            "並び順",
            ["優先度が高い順", "正答率が低い順", "挑戦回数が多い順", "年度が新しい順"],
            help="復習リストの並び替え基準を変更します。",
            key="weakness_sort",
        )
    filtered = merged.copy()
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if attempts_threshold:
        filtered = filtered[filtered["attempts_count"] >= attempts_threshold]
    filtered = filtered[filtered["accuracy"] * 100 <= accuracy_ceiling]
    if sort_option == "正答率が低い順":
        filtered = filtered.sort_values(["accuracy", "attempts_count"], ascending=[True, False])
    elif sort_option == "挑戦回数が多い順":
        filtered = filtered.sort_values(["attempts_count", "accuracy"], ascending=[False, True])
    elif sort_option == "年度が新しい順":
        filtered = filtered.sort_values(["year", "priority"], ascending=[False, False])
    else:
        filtered = filtered.sort_values(["priority", "accuracy"], ascending=[False, True])
    if filtered.empty:
        st.info("条件に合致する弱点候補がありません。フィルタ設定を見直してください。")
        return
    display_df = filtered.head(15)[
        [
            "question_id",
            "category",
            "year",
            "q_no",
            "accuracy",
            "attempts_count",
            "avg_seconds",
        ]
    ].copy()
    display_df["accuracy"] = (display_df["accuracy"].astype(float) * 100).round(0)
    display_df["avg_seconds"] = display_df["avg_seconds"].astype(float).round(1)
    render_table_with_category_tags(
        display_df,
        column_order=[
            "question_id",
            "category",
            "year",
            "q_no",
            "accuracy",
            "attempts_count",
            "avg_seconds",
        ],
        rename_map={
            "question_id": "問題ID",
            "category": "分野",
            "year": "年度",
            "q_no": "問",
            "accuracy": "正答率(%)",
            "attempts_count": "挑戦回数",
            "avg_seconds": "平均解答時間(秒)",
        },
        index_column="question_id",
    )
    candidates = filtered[~filtered["id"].isna()]
    if candidates.empty:
        st.info("弱点候補の問題を特定できませんでした。履歴を増やしましょう。")
        return
    selected_qid = st.selectbox(
        "復習する問題",
        candidates["id"],
        format_func=lambda x: format_question_label(df, x),
        key="weakness_question",
    )
    row = df[df["id"] == selected_qid].iloc[0]
    render_question_interaction(db, row, attempt_mode="weakness", key_prefix="weakness")


def render_law_revision_lane(db: DBManager, parent_nav: str = "学習") -> None:
    render_specialized_header(parent_nav, "法改正対策", "law_revision")
    st.subheader("法改正対策")
    law_df = db.load_law_revision_questions()
    sync_logs = db.load_law_revision_sync_logs(limit=5)
    if law_df.empty:
        st.info(
            "法改正予想問題データが登録されていません。『設定 ＞ データ入出力』から law_revision.csv を取り込みましょう。"
        )
        return
    st.caption(
        "最新の法改正ポイントを重点的に演習できます。正答が未設定の場合は自己採点してください。"
    )
    total_questions = len(law_df)
    pending_count = int((law_df.get("review_status") == "pending").sum())
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("登録数", total_questions)
    with summary_cols[1]:
        unique_laws = law_df["law_name"].replace("", pd.NA).dropna().nunique()
        st.metric("対象法令", int(unique_laws) if not pd.isna(unique_laws) else 0)
    with summary_cols[2]:
        recent_years = law_df["revision_year"].dropna()
        if recent_years.empty:
            st.metric("最新改正年度", "未設定")
        else:
            st.metric("最新改正年度", f"{int(recent_years.max())}年")
    with summary_cols[3]:
        st.metric("未レビュー", pending_count)
    if not sync_logs.empty:
        latest = sync_logs.iloc[0]
        status = latest.get("status", "-")
        timestamp = latest.get("fetched_at")
        status_text = f"{status}"
        if pd.notna(timestamp):
            status_text += f" / {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')}"
        st.caption(f"最終更新: {status_text}")
    else:
        st.caption("自動更新ログがありません。『データ入出力』で取得を開始できます。")
    with st.expander("自動更新状況", expanded=False):
        if sync_logs.empty:
            st.info("まだ自動取得履歴がありません。")
        else:
            display = sync_logs.copy()
            display["fetched_at"] = pd.to_datetime(display["fetched_at"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                display[["fetched_at", "source", "status", "revisions_detected", "questions_generated", "message"]],
                use_container_width=True,
            )
    with st.expander("出題条件", expanded=True):
        law_names = sorted(
            {
                str(name).strip()
                for name in law_df.get("law_name", pd.Series(dtype="object")).dropna()
                if str(name).strip()
            }
        )
        if law_names:
            selected_laws = st.multiselect(
                "対象法令",
                law_names,
                default=law_names,
                key="law_revision_laws",
            )
        else:
            selected_laws = []
        category_candidates = sorted(
            {
                str(cat).strip()
                for cat in law_df.get("category", pd.Series(dtype="object")).dropna()
                if str(cat).strip()
            }
        )
        if category_candidates:
            selected_categories = st.multiselect(
                "分野タグ",
                category_candidates,
                default=category_candidates,
                key="law_revision_categories",
            )
        else:
            selected_categories = []
        year_series = law_df.get("revision_year")
        include_unknown_year = st.checkbox(
            "改正年未設定の問題も含める",
            value=True,
            key="law_revision_include_unknown",
        )
        year_range = None
        if year_series is not None and year_series.dropna().size > 0:
            min_year = int(year_series.dropna().min())
            max_year = int(year_series.dropna().max())
            if min_year == max_year:
                default_range = (min_year, max_year)
            else:
                default_range = (max(min_year, max_year - 4), max_year)
            year_range = st.slider(
                "改正年度",
                min_year,
                max_year,
                default_range,
                key="law_revision_year_range",
            )
        status_options = ["すべて", "pending", "approved", "rejected"]
        review_status = st.selectbox(
            "レビュー状態",
            status_options,
            format_func=lambda value: {
                "すべて": "すべて",
                "pending": "要レビュー",
                "approved": "承認済み",
                "rejected": "差戻し",
            }.get(value, value),
            key="law_revision_review_status",
        )
        keyword = st.text_input(
            "キーワードフィルタ",
            value="",
            key="law_revision_keyword",
            help="問題文・タグ・法令名を部分一致で絞り込みます。",
        )
    filtered = law_df.copy()
    if selected_laws:
        filtered = filtered[filtered["law_name"].isin(selected_laws)]
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if year_range:
        start_year, end_year = year_range
        mask = filtered["revision_year"].between(start_year, end_year)
        if include_unknown_year:
            mask = mask | filtered["revision_year"].isna()
        filtered = filtered[mask]
    elif not include_unknown_year:
        filtered = filtered[filtered["revision_year"].notna()]
    if review_status != "すべて" and "review_status" in filtered.columns:
        filtered = filtered[filtered["review_status"].fillna("pending") == review_status]
    if keyword:
        keyword = keyword.strip()
        if keyword:
            contains_mask = (
                filtered.get("question", pd.Series(dtype="object")).fillna("").str.contains(keyword, case=False)
                | filtered.get("tags", pd.Series(dtype="object")).fillna("").str.contains(keyword, case=False)
                | filtered.get("law_name", pd.Series(dtype="object")).fillna("").str.contains(keyword, case=False)
                | filtered.get("topic", pd.Series(dtype="object")).fillna("").str.contains(keyword, case=False)
            )
            filtered = filtered[contains_mask]
    if filtered.empty:
        st.warning("条件に一致する法改正問題がありません。フィルタを緩和してください。")
        return
    with st.expander("レビュー・承認", expanded=False):
        pending_df = filtered[filtered.get("review_status") == "pending"]
        if pending_df.empty:
            st.info("未レビューの問題はありません。")
        else:
            review_selection = st.multiselect(
                "承認対象の問題",
                pending_df["id"],
                format_func=lambda qid: format_question_label(law_df, qid),
                key="law_revision_review_selection",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("選択した問題を承認", key="law_revision_approve"):
                    db.update_law_revision_review_status(review_selection, "approved")
                    st.success("承認しました。")
                    st.experimental_rerun()
            with col_b:
                if st.button("選択した問題を差戻し", key="law_revision_reject"):
                    db.update_law_revision_review_status(review_selection, "rejected")
                    st.warning("差戻しました。")
                    st.experimental_rerun()
    with st.expander("法改正問題一覧", expanded=False):
        preview_cols = [
            "law_name",
            "revision_year",
            "effective_date",
            "label",
            "category",
            "question",
        ]
        available_cols = [col for col in preview_cols if col in filtered.columns]
        st.dataframe(filtered[available_cols].head(20))
        st.caption("20件まで表示しています。CSVで詳細を確認できます。")
    max_questions = max(1, len(filtered))
    default_count = min(10, max_questions)
    col1, col2 = st.columns(2)
    with col1:
        question_count = st.slider(
            "出題数",
            1,
            max_questions,
            default_count,
            key="law_revision_question_count",
        )
    with col2:
        order_option = st.radio(
            "出題順",
            ["ランダム", "改正年が新しい順", "登録順"],
            key="law_revision_order",
            horizontal=True,
        )

    def start_law_revision_session() -> None:
        selection = filtered
        if order_option == "ランダム":
            selection = selection.sample(question_count)
        elif order_option == "改正年が新しい順":
            selection = selection.sort_values(
                ["revision_year", "created_at"], ascending=[False, False]
            ).head(question_count)
        else:
            selection = selection.head(question_count)
        selection = selection.reset_index(drop=True)
        st.session_state["law_revision_session"] = {
            "questions": selection.to_dict(orient="records"),
            "index": 0,
            "started_at": dt.datetime.now().isoformat(),
            "run_id": hashlib.sha256(f"lawrev|{time.time()}".encode("utf-8")).hexdigest()[:8],
        }

    st.button(
        "法改正問題を開始",
        key="law_revision_start",
        type="primary",
        on_click=with_rerun(start_law_revision_session),
    )

    session = st.session_state.get("law_revision_session")
    if not session:
        return
    questions = session.get("questions", [])
    if not questions:
        st.warning("セッションに問題がありません。再度開始してください。")
        return
    index = session.get("index", 0)
    total = len(questions)
    index = max(0, min(index, total - 1))
    st.progress((index + 1) / total)

    def set_index(new_index: int) -> None:
        current = st.session_state.get("law_revision_session", {})
        current["index"] = new_index
        st.session_state["law_revision_session"] = current

    row_series = pd.Series(questions[index])
    navigation = QuestionNavigation(
        has_prev=index > 0,
        has_next=index < total - 1,
        on_prev=with_rerun(set_index, max(0, index - 1)),
        on_next=with_rerun(set_index, min(total - 1, index + 1)),
        label=f"{index + 1}/{total} 問を学習中",
    )
    render_law_revision_metadata(row_series)
    render_question_interaction(
        db,
        row_series,
        attempt_mode="law_revision",
        key_prefix=f"law_revision_{session.get('run_id', 'session')}",
        navigation=navigation,
        enable_srs=False,
        log_attempts=False,
    )
    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        st.button(
            "前の問題",
            use_container_width=True,
            disabled=index <= 0,
            on_click=with_rerun(set_index, max(0, index - 1)),
            key="law_revision_prev_button",
        )
    with action_cols[1]:
        st.button(
            "次の問題",
            use_container_width=True,
            disabled=index >= total - 1,
            on_click=with_rerun(set_index, min(total - 1, index + 1)),
            key="law_revision_next_button",
        )
    with action_cols[2]:
        st.caption(f"演習中 {index + 1}/{total} 問")
    st.button(
        "セッションを終了",
        key="law_revision_end_session",
        on_click=with_rerun(lambda: st.session_state.pop("law_revision_session", None)),
    )


def render_predicted_lane(db: DBManager, parent_nav: str = "学習") -> None:
    render_specialized_header(parent_nav, "予想問題演習", "predicted")
    st.subheader("予想問題演習")
    predicted_df = db.load_predicted_questions()
    if predicted_df.empty:
        st.info("予想問題データが登録されていません。『設定 ＞ データ入出力』からCSVを取り込んでください。")
        return
    st.caption("アップロードした予想問題を使って直前対策の演習を行います。正答が未設定の場合は自己採点となります。")
    total_questions = len(predicted_df)
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("登録数", total_questions)
    with summary_cols[1]:
        available_correct = predicted_df["correct"].notna().sum()
        st.metric("正答付き", int(available_correct))
    with summary_cols[2]:
        categories = predicted_df["category"].replace("", pd.NA).dropna().nunique()
        st.metric("カテゴリ数", int(categories) if not pd.isna(categories) else 0)
    with st.expander("予想問題一覧", expanded=False):
        preview_cols = [col for col in ["label", "category", "topic", "source"] if col in predicted_df.columns]
        if preview_cols:
            st.dataframe(predicted_df[preview_cols + ["question"]].head(20))
        else:
            st.dataframe(predicted_df.head(20))
        st.caption("20件まで表示しています。詳細はCSVを編集してご確認ください。")
    max_questions = max(1, total_questions)
    default_count = min(10, max_questions)
    col1, col2 = st.columns(2)
    with col1:
        question_count = st.slider("出題数", 1, max_questions, default_count, key="predicted_question_count")
    with col2:
        order_option = st.radio("出題順", ["ランダム", "登録順"], key="predicted_order", horizontal=True)

    def start_predicted_session() -> None:
        if order_option == "ランダム":
            selection = predicted_df.sample(question_count).reset_index(drop=True)
        else:
            selection = predicted_df.head(question_count).reset_index(drop=True)
        st.session_state["predicted_session"] = {
            "questions": selection.to_dict(orient="records"),
            "index": 0,
            "started_at": dt.datetime.now().isoformat(),
            "run_id": hashlib.sha256(f"predicted|{time.time()}".encode("utf-8")).hexdigest()[:8],
        }

    st.button("予想問題を開始", key="predicted_start", type="primary", on_click=with_rerun(start_predicted_session))

    session = st.session_state.get("predicted_session")
    if not session:
        return
    questions = session.get("questions", [])
    if not questions:
        st.warning("セッションに問題がありません。再度開始してください。")
        return
    index = session.get("index", 0)
    total = len(questions)
    index = max(0, min(index, total - 1))
    st.progress((index + 1) / total)

    def set_index(new_index: int) -> None:
        current = st.session_state.get("predicted_session", {})
        current["index"] = new_index
        st.session_state["predicted_session"] = current

    row_series = pd.Series(questions[index])
    navigation = QuestionNavigation(
        has_prev=index > 0,
        has_next=index < total - 1,
        on_prev=with_rerun(set_index, max(0, index - 1)),
        on_next=with_rerun(set_index, min(total - 1, index + 1)),
        label=f"{index + 1}/{total} 問を学習中",
    )
    render_question_interaction(
        db,
        row_series,
        attempt_mode="predicted",
        key_prefix=f"predicted_{session.get('run_id', 'session')}",
        navigation=navigation,
        enable_srs=False,
        log_attempts=False,
    )
    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        disabled = index <= 0
        st.button(
            "前の問題",
            use_container_width=True,
            disabled=disabled,
            on_click=with_rerun(set_index, max(0, index - 1)),
            key="predicted_prev_button",
        )
    with action_cols[1]:
        disabled = index >= total - 1
        st.button(
            "次の問題",
            use_container_width=True,
            disabled=disabled,
            on_click=with_rerun(set_index, min(total - 1, index + 1)),
            key="predicted_next_button",
        )
    with action_cols[2]:
        st.caption(f"演習中 {index + 1}/{total} 問")
    st.button(
        "セッションを終了",
        key="predicted_end_session",
        type="secondary",
        on_click=with_rerun(lambda: st.session_state.pop("predicted_session", None)),
    )


def render_exam_session_body(
    db: DBManager,
    df: pd.DataFrame,
    session: ExamSession,
    key_prefix: str,
    pass_line: float = 0.7,
) -> None:
    st.subheader(session.name)
    if st.session_state["settings"].get("timer", True):
        elapsed = dt.datetime.now() - session.started_at
        remaining = max(0, 120 * 60 - int(elapsed.total_seconds()))
        minutes, seconds = divmod(remaining, 60)
        st.info(f"残り時間: {minutes:02d}:{seconds:02d}")
    responses: Dict[str, int] = {}
    choice_labels = ["①", "②", "③", "④"]
    for qid in session.questions:
        row_df = df[df["id"] == qid]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        st.markdown(f"### {row['year']}年 問{row['q_no']}")
        st.markdown(f"**{row['category']} / {row['topic']}**")
        render_law_reference(row, db=db)
        st.markdown(row["question"], unsafe_allow_html=True)
        options = [row.get(f"choice{i}", "") for i in range(1, 5)]
        option_map = {
            idx + 1: f"{choice_labels[idx]} {options[idx]}" if options[idx] else choice_labels[idx]
            for idx in range(4)
        }
        choice = st.radio(
            f"回答 ({qid})",
            list(option_map.keys()),
            format_func=lambda opt: option_map.get(opt, str(opt)),
            key=f"{key_prefix}_exam_{qid}",
            horizontal=True,
            index=None,
        )
        if choice is not None:
            responses[qid] = choice
    if st.button(
        "採点する",
        key=f"{key_prefix}_grade",
        help="現在の回答を保存し、正答率や分野別統計を表示します。",
    ):
        evaluate_exam_attempt(db, df, session, responses, pass_line)


def evaluate_exam_attempt(
    db: DBManager,
    df: pd.DataFrame,
    session: ExamSession,
    responses: Dict[str, int],
    pass_line: float,
) -> None:
    total_questions = len(session.questions)
    correct = 0
    per_category: Dict[str, Dict[str, int]] = {}
    wrong_choices: List[Dict[str, object]] = []
    attempt_records: List[Tuple[str, int, bool]] = []
    duration = max((dt.datetime.now() - session.started_at).total_seconds(), 1)
    avg_seconds = duration / max(len(responses), 1)
    for qid in session.questions:
        row_df = df[df["id"] == qid]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        correct_choice = int(row.get("correct") or 0)
        choice = responses.get(qid)
        is_correct = choice is not None and correct_choice == choice
        if is_correct:
            correct += 1
        category = row.get("category", "その他")
        stats = per_category.setdefault(category, {"total": 0, "correct": 0})
        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1
        attempt_records.append((qid, choice, is_correct))
        if not is_correct and correct_choice in range(1, 5):
            wrong_choices.append(
                {
                    "question": f"{row['year']}年 問{row['q_no']}",
                    "selected": choice,
                    "correct": correct_choice,
                    "category": category,
                }
            )
    finished_at = dt.datetime.now()
    exam_id = db.log_exam_result(
        {
            "name": session.name,
            "started_at": session.started_at,
            "finished_at": finished_at,
            "year_mode": session.year_mode,
            "score": correct,
        }
    )
    for qid, choice, is_correct in attempt_records:
        db.record_attempt(
            qid,
            choice,
            is_correct,
            seconds=int(avg_seconds),
            mode=session.mode,
            exam_id=exam_id,
            confidence=None,
            grade=None,
        )
    accuracy = correct / max(total_questions, 1)
    remaining_time = max(0, 120 * 60 - int(duration))
    answered = len(responses)
    unanswered = total_questions - answered
    expected_final = correct + unanswered * (correct / max(answered, 1)) if answered else 0
    result_payload = {
        "score": correct,
        "total": total_questions,
        "accuracy": accuracy,
        "pass_line": pass_line,
        "per_category": per_category,
        "wrong_choices": wrong_choices,
        "remaining_time": remaining_time,
        "expected_final": expected_final,
        "mode": session.mode,
        "exam_id": exam_id,
    }
    st.session_state[f"exam_result_{session.mode}"] = result_payload
    st.session_state["exam_session"] = None


def display_exam_result(result: Dict[str, object]) -> None:
    score = result["score"]
    total = result["total"]
    accuracy = result["accuracy"]
    pass_line = result["pass_line"]
    status = "✅ 合格ライン到達" if accuracy >= pass_line else "⚠️ 合格ライン未達"
    st.markdown(f"### 採点結果 — {status}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("得点", f"{score} / {total}")
    with col2:
        st.metric("正答率", f"{accuracy * 100:.1f}%")
    with col3:
        threshold = int(total * pass_line)
        st.metric("合格ライン", f"{threshold} 点")
    st.progress(min(accuracy / max(pass_line, 1e-6), 1.0))
    remaining_minutes, remaining_seconds = divmod(int(result["remaining_time"]), 60)
    st.metric(
        "残り時間 × 想定到達点",
        f"{remaining_minutes:02d}:{remaining_seconds:02d} / {result['expected_final']:.1f} 点",
    )
    if result["per_category"]:
        radar_df = pd.DataFrame(
            [
                {
                    "category": category,
                    "accuracy": stats["correct"] / max(stats["total"], 1),
                }
                for category, stats in result["per_category"].items()
            ]
        )
        if not radar_df.empty:
            import altair as alt

            radar_df = pd.concat([radar_df, radar_df.iloc[[0]]], ignore_index=True)
            chart = (
                alt.Chart(radar_df)
                .mark_line()
                .encode(
                    theta=alt.Theta("category", sort=None),
                    radius=alt.Radius("accuracy", scale=alt.Scale(domain=[0, 1])),
                )
                .properties(title="分野別スコアレーダー")
            )
            points = (
                alt.Chart(radar_df)
                .mark_point(size=80)
                .encode(
                    theta=alt.Theta("category", sort=None),
                    radius=alt.Radius("accuracy", scale=alt.Scale(domain=[0, 1])),
                )
            )
            st.altair_chart(chart + points, use_container_width=True)
    wrong_choices = result.get("wrong_choices", [])
    if wrong_choices:
        st.markdown("#### 誤答の代替正解肢傾向")
        wrong_df = pd.DataFrame(wrong_choices)
        option_map = {1: "①", 2: "②", 3: "③", 4: "④"}
        wrong_df["選択肢"] = wrong_df["selected"].map(option_map).fillna("未回答")
        wrong_df["正解肢"] = wrong_df["correct"].map({1: "①", 2: "②", 3: "③", 4: "④"})
        st.dataframe(
            wrong_df[["question", "category", "選択肢", "正解肢"]],
            use_container_width=True,
        )


def render_question_interaction(
    db: DBManager,
    row: pd.Series,
    attempt_mode: str,
    key_prefix: str,
    navigation: Optional[QuestionNavigation] = None,
    enable_srs: bool = True,
    log_attempts: bool = True,
) -> None:
    inject_ui_styles()
    last_question_key = f"{key_prefix}_last_question"
    feedback_key = f"{key_prefix}_feedback"
    selected_key = f"{key_prefix}_selected_{row['id']}"
    order_key = f"{key_prefix}_order_{row['id']}"
    explanation_key = f"{key_prefix}_explanation_{row['id']}"
    confidence_key = f"{key_prefix}_confidence_{row['id']}"
    help_state_key = f"{key_prefix}_help_visible"
    if st.session_state.get(last_question_key) != row["id"]:
        st.session_state[last_question_key] = row["id"]
        st.session_state.pop(feedback_key, None)
        st.session_state[selected_key] = None
        st.session_state[confidence_key] = 50
        st.session_state[order_key] = None
        st.session_state[explanation_key] = False
    feedback = st.session_state.get(feedback_key)
    if feedback and feedback.get("question_id") != row["id"]:
        feedback = None
    choices = [row.get(f"choice{i}", "") for i in range(1, 5)]
    base_order = list(range(4))
    if st.session_state["settings"].get("shuffle_choices", True):
        random.seed(f"{row['id']}_{attempt_mode}")
        random.shuffle(base_order)
    if st.session_state.get(order_key) is None:
        st.session_state[order_key] = base_order.copy()
    order = st.session_state.get(order_key, base_order)
    choice_labels = ["①", "②", "③", "④"]
    label_value = str(row.get("label", "")).strip()
    if label_value:
        header = label_value
    else:
        year_display = format_year_value(row.get("year"))
        q_no_display = format_qno_value(row.get("q_no"))
        if year_display and q_no_display:
            header = f"{year_display} 問{q_no_display}"
        elif year_display:
            header = year_display
        elif q_no_display:
            header = f"問{q_no_display}"
        else:
            header = "予想問題"
    st.markdown(f"### {header}")
    category_value = str(row.get("category", "") or "").strip()
    topic_value = str(row.get("topic", "") or "").strip()
    tag_html = render_category_topic_tags(category_value, topic_value)
    if tag_html:
        st.markdown(tag_html, unsafe_allow_html=True)
    render_law_reference(row, db=db)
    st.markdown(row["question"], unsafe_allow_html=True)
    selected_choice = st.session_state.get(selected_key)
    graded_selected_choice: Optional[int] = None
    correct_choice_idx: Optional[int] = None
    explanation_summary = ""
    explanation_value = row.get("explanation")
    if pd.notna(explanation_value) and str(explanation_value).strip():
        explanation_summary, _ = parse_explanation_sections(explanation_value)
    if feedback:
        graded_selected_choice = feedback.get("selected_choice")
        correct_choice = feedback.get("correct_choice")
        if isinstance(correct_choice, int):
            correct_choice_idx = correct_choice - 1
    display_selected_choice = (
        graded_selected_choice
        if graded_selected_choice is not None
        else selected_choice
    )
    is_graded = feedback is not None and feedback.get("question_id") == row["id"]
    button_labels: List[str] = []
    for idx in range(0, len(order), 2):
        cols = st.columns(2)
        for col_idx in range(2):
            pos = idx + col_idx
            if pos >= len(order):
                continue
            actual_idx = order[pos]
            label_text = choices[actual_idx]
            display_label = f"{choice_labels[actual_idx]} {label_text}".strip()
            button_labels.append(display_label or choice_labels[actual_idx])
            button_key = f"{key_prefix}_choice_{row['id']}_{actual_idx}"
            wrapper_classes = ["takken-choice-button"]
            if is_graded:
                wrapper_classes.append("is-graded")
                if actual_idx == correct_choice_idx:
                    wrapper_classes.append("is-correct")
                elif graded_selected_choice == actual_idx:
                    wrapper_classes.append("is-incorrect")
                else:
                    wrapper_classes.append("is-dimmed")
            if display_selected_choice == actual_idx and not is_graded:
                wrapper_classes.append("is-selected")
            button_type = (
                "primary" if display_selected_choice == actual_idx and not is_graded else "secondary"
            )
            with cols[col_idx]:
                st.markdown(
                    f'<div class="{" ".join(wrapper_classes)}">',
                    unsafe_allow_html=True,
                )
                if st.button(
                    display_label or choice_labels[actual_idx],
                    key=button_key,
                    use_container_width=True,
                    type=button_type,
                ):
                    if not is_graded:
                        st.session_state[selected_key] = actual_idx
                        selected_choice = actual_idx
                        safe_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
    st.caption("1〜4キーで選択肢を即答できます。E:解説 F:フラグ N/P:移動 H:ヘルプ R:SRSリセット")
    confidence_value = st.session_state.get(confidence_key)
    if confidence_value is None:
        confidence_value = 50
    else:
        confidence_value = int(confidence_value)
    confidence_value = st.slider(
        "確信度（ぜんぜん自信なし ↔ 完璧）",
        0,
        100,
        value=confidence_value,
        key=confidence_key,
    )
    show_explanation = st.session_state.get(explanation_key, False)
    flagged = row["id"] in set(st.session_state.get("review_flags", []))
    grade_label = "採点"
    explanation_label = "解説を隠す" if show_explanation else "解説を表示"
    flag_label = "フラグ解除" if flagged else "復習フラグ"
    help_label = "ヘルプ"
    auto_advance_enabled = st.session_state["settings"].get("auto_advance", False)
    grade_clicked = False
    needs_rerun_after_grade = False
    help_visible = st.session_state.get(help_state_key, False)
    action_buttons = [
        {
            "id": "grade",
            "label": grade_label,
            "key": f"{key_prefix}_grade_{row['id']}",
            "type": "primary",
        },
        {
            "id": "toggle_explanation",
            "label": explanation_label,
            "key": f"{key_prefix}_toggle_explanation_{row['id']}",
        },
        {
            "id": "flag",
            "label": flag_label,
            "key": f"{key_prefix}_flag_{row['id']}",
        },
        {
            "id": "help",
            "label": help_label,
            "key": f"{key_prefix}_help_{row['id']}",
        },
    ]
    if enable_srs:
        action_buttons.append(
            {
                "id": "srs_reset",
                "label": "SRSリセット",
                "key": f"{key_prefix}_srs_reset_{row['id']}",
            }
        )
    with st.container():
        st.markdown('<div class="takken-action-bar">', unsafe_allow_html=True)
        for action in action_buttons:
            st.markdown('<div class="takken-action-item">', unsafe_allow_html=True)
            button_kwargs = {
                "key": action["key"],
                "use_container_width": True,
            }
            if "type" in action:
                button_kwargs["type"] = action["type"]
            clicked = st.button(action["label"], **button_kwargs)
            if action["id"] == "grade" and clicked:
                grade_clicked = True
            elif action["id"] == "toggle_explanation" and clicked:
                show_explanation = not show_explanation
                st.session_state[explanation_key] = show_explanation
            elif action["id"] == "flag" and clicked:
                flags = set(st.session_state.get("review_flags", []))
                if flagged:
                    flags.discard(row["id"])
                else:
                    flags.add(row["id"])
                st.session_state["review_flags"] = list(flags)
            elif action["id"] == "help" and clicked:
                help_visible = not help_visible
                st.session_state[help_state_key] = help_visible
            elif action["id"] == "srs_reset" and clicked and enable_srs:
                db.upsert_srs(
                    row["id"],
                    {
                        "repetition": 0,
                        "interval": 1,
                        "ease": st.session_state["settings"].get("sm2_initial_ease", 2.5),
                        "due_date": dt.date.today(),
                        "last_grade": None,
                        "updated_at": dt.datetime.now(),
                    },
                )
                st.success("SRSを初期化しました。明日から復習に再投入されます。")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    flagged = row["id"] in set(st.session_state.get("review_flags", []))
    help_visible = st.session_state.get(help_state_key, help_visible)
    if auto_advance_enabled and navigation and navigation.has_next:
        st.caption("採点後0.8秒で次問に自動遷移します。")
    if flagged:
        st.caption("この問題は復習フラグが設定されています。")
    feedback = st.session_state.get(feedback_key)
    if grade_clicked:
        if selected_choice is None:
            st.warning("選択肢を選んでから採点してください。")
        else:
            correct_choice = row.get("correct")
            if pd.isna(correct_choice):
                st.warning("正答が未登録の問題です。解答データを取り込んでください。")
            else:
                correct_choice = int(correct_choice)
                is_correct = (selected_choice + 1) == correct_choice
                initial_ease = st.session_state["settings"].get("sm2_initial_ease", 2.5)
                grade_value = confidence_to_grade(is_correct, confidence_value)
                if enable_srs:
                    srs_row = db.fetch_srs(row["id"])
                    payload = sm2_update(srs_row, grade=grade_value, initial_ease=initial_ease)
                    db.upsert_srs(row["id"], payload)
                log_offline_attempt(
                    {
                        "timestamp": dt.datetime.now().isoformat(),
                        "question_id": row["id"],
                        "year": row.get("year"),
                        "q_no": row.get("q_no"),
                        "category": row.get("category"),
                        "topic": row.get("topic"),
                        "selected": selected_choice + 1,
                        "correct": correct_choice,
                        "is_correct": is_correct,
                        "mode": attempt_mode,
                        "confidence": confidence_value,
                        "srs_grade": grade_value,
                    }
                )
                st.session_state[feedback_key] = {
                    "is_correct": is_correct,
                    "correct_choice": correct_choice,
                    "question_id": row["id"],
                    "confidence": confidence_value,
                    "grade": grade_value,
                    "selected_choice": selected_choice,
                }
                feedback = st.session_state[feedback_key]
                if log_attempts:
                    db.record_attempt(
                        row["id"],
                        selected_choice + 1,
                        is_correct,
                        seconds=0,
                        mode=attempt_mode,
                        confidence=confidence_value,
                        grade=grade_value,
                    )
                if (
                    auto_advance_enabled
                    and navigation is not None
                    and navigation.has_next
                    and navigation.on_next is not None
                ):
                    time.sleep(0.8)
                    navigation.on_next()
                    safe_rerun()
                    needs_rerun_after_grade = False
                else:
                    needs_rerun_after_grade = True
    if grade_clicked and needs_rerun_after_grade:
        safe_rerun()
    if feedback and feedback.get("question_id") == row["id"]:
        correct_msg = choice_labels[feedback["correct_choice"] - 1]
        message = "正解です！" if feedback["is_correct"] else f"不正解。正答は {correct_msg}"
        (st.success if feedback["is_correct"] else st.error)(message)
        if explanation_summary:
            summary_html = (
                f'<div class="takken-feedback-summary">💡 <span>{html_module.escape(explanation_summary)}</span></div>'
            )
            st.markdown(summary_html, unsafe_allow_html=True)
        st.caption(
            f"確信度 {feedback.get('confidence', confidence_value)}% → 復習グレード {feedback.get('grade', '')}"
        )
    if show_explanation:
        st.markdown("#### 解説")
        render_explanation_content(row, db=db)
    if help_visible:
        st.info(
            """ショートカット一覧\n- 1〜4: 選択肢を選ぶ\n- E: 解説の表示/非表示\n- F: 復習フラグの切り替え\n- N/P: 次へ・前へ\n- H: このヘルプ"""
        )
    nav_prev_label = "前へ"
    nav_next_label = "次へ"
    if navigation:
        nav_cols = st.columns([1, 1, 2])
        prev_kwargs = {
            "use_container_width": True,
            "disabled": not navigation.has_prev,
            "key": f"{key_prefix}_prev_{row['id']}",
        }
        next_kwargs = {
            "use_container_width": True,
            "disabled": not navigation.has_next,
            "key": f"{key_prefix}_next_{row['id']}",
        }
        if navigation.on_prev:
            prev_kwargs["on_click"] = navigation.on_prev
        if navigation.on_next:
            next_kwargs["on_click"] = navigation.on_next
        with nav_cols[0]:
            st.button(nav_prev_label, **prev_kwargs)
        with nav_cols[1]:
            st.button(nav_next_label, **next_kwargs)
        with nav_cols[2]:
            if navigation.label:
                st.caption(navigation.label)
    render_offline_downloads(f"{key_prefix}_{row['id']}")
    shortcut_map: Dict[str, str] = {}
    for idx, label in enumerate(button_labels[:4]):
        shortcut_map[str(idx + 1)] = label
    shortcut_map["e"] = explanation_label
    shortcut_map["f"] = flag_label
    shortcut_map["h"] = help_label
    if enable_srs:
        shortcut_map["r"] = "SRSリセット"
    if navigation:
        shortcut_map["n"] = nav_next_label
        shortcut_map["p"] = nav_prev_label
    register_keyboard_shortcuts(shortcut_map)


def format_year_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        return f"{int(text)}年"
    return text


def format_qno_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        return str(int(text))
    return text


def format_question_label(df: pd.DataFrame, question_id: str) -> str:
    if df.empty:
        return str(question_id)
    matches = df[df["id"] == question_id]
    if matches.empty:
        return f"未登録の問題 ({question_id})"
    row = matches.iloc[0]
    year_value = row.get("year")
    if pd.isna(year_value):
        year_display = "?"
    elif isinstance(year_value, (int, np.integer)):
        year_display = int(year_value)
    elif isinstance(year_value, float):
        year_display = int(year_value)
    else:
        year_display = year_value
    q_no_value = row.get("q_no")
    if pd.isna(q_no_value):
        q_no_display = "?"
    elif isinstance(q_no_value, (int, np.integer)):
        q_no_display = int(q_no_value)
    elif isinstance(q_no_value, float):
        q_no_display = int(q_no_value)
    else:
        q_no_display = q_no_value
    category_value = row.get("category")
    if pd.isna(category_value):
        category_display = "不明"
    else:
        category_display = str(category_value).strip() or "不明"
    return f"{year_display}年 問{q_no_display} ({category_display})"


def build_law_reference_query(row: pd.Series) -> Optional[str]:
    insight = get_outline_insight(row)
    terms = insight.get("terms", [])
    if not terms:
        return None
    return " ".join(str(term) for term in terms if term)


def render_law_reference(row: pd.Series, db: Optional[DBManager] = None) -> None:
    insight = get_outline_insight(row)
    terms = insight.get("terms", [])
    query: Optional[str] = " ".join(terms) if terms else None
    caption_parts: List[str] = []
    if query:
        url = LAW_REFERENCE_BASE_URL.format(query=quote_plus(query))
        caption_parts.append(f"[条文検索]({url})")
    if caption_parts:
        st.caption(" ｜ ".join(caption_parts))


def render_law_revision_metadata(row: pd.Series) -> None:
    details: List[str] = []
    law_name = row.get("law_name")
    if pd.notna(law_name) and str(law_name).strip():
        details.append(f"**法令**: {str(law_name).strip()}")
    revision_year = row.get("revision_year")
    if pd.notna(revision_year) and str(revision_year).strip():
        try:
            details.append(f"**改正年度**: {int(revision_year)}年")
        except Exception:
            details.append(f"**改正年度**: {revision_year}")
    effective_date = row.get("effective_date")
    if pd.notna(effective_date) and str(effective_date).strip():
        details.append(f"**施行日**: {str(effective_date).strip()}")
    source = row.get("source")
    if pd.notna(source) and str(source).strip():
        details.append(f"**出典**: {str(source).strip()}")
    tags = row.get("tags")
    if pd.notna(tags) and str(tags).strip():
        details.append(f"**タグ**: {str(tags).strip()}")
    if details:
        st.caption(" ｜ ".join(details))


def render_question_preview(row: pd.Series, db: Optional[DBManager] = None) -> None:
    render_law_reference(row, db=db)
    question_text = row.get("question", "")
    if pd.isna(question_text):
        question_text = ""
    st.markdown(str(question_text), unsafe_allow_html=True)
    choice_labels = ["①", "②", "③", "④"]
    for idx, label in enumerate(choice_labels, start=1):
        choice_text = row.get(f"choice{idx}")
        if pd.isna(choice_text):
            continue
        choice_text = str(choice_text)
        if not choice_text.strip():
            continue
        st.markdown(f"{label} {choice_text}", unsafe_allow_html=True)


def render_mock_exam(db: DBManager, df: pd.DataFrame) -> None:
    st.title("模試")
    if df.empty:
        st.warning("設問データがありません。")
        return
    with st.form("mock_exam_form"):
        year_mode = st.selectbox(
            "出題方式",
            ["最新年度", "年度選択", "層化ランダム50"],
            help="最新年度の全問、任意年度のみ、または分野バランスを取った50問から選べます。",
        )
        if year_mode == "年度選択":
            selected_year = st.selectbox(
                "年度",
                sorted(df["year"].unique(), reverse=True),
                help="模試に使用する年度を選択します。",
            )
            subset = df[df["year"] == selected_year]
            questions = list(subset["id"])
        elif year_mode == "最新年度":
            latest_year = df["year"].max()
            subset = df[df["year"] == latest_year]
            questions = list(subset["id"])
        else:
            questions = stratified_exam(df)
        submit = st.form_submit_button("模試開始", help="選択した条件で模試を開始し、即座に試験画面へ移動します。")
    if submit:
        st.session_state.pop("exam_result_模試", None)
        st.session_state["exam_session"] = ExamSession(
            id=None,
            name=f"模試 {dt.datetime.now():%Y%m%d-%H%M}",
            questions=questions,
            started_at=dt.datetime.now(),
            year_mode=year_mode,
            mode="模試",
        )
    session: Optional[ExamSession] = st.session_state.get("exam_session")
    if session and session.mode == "模試":
        render_exam_session_body(db, df, session, key_prefix="mock")
    result = st.session_state.get("exam_result_模試")
    if result:
        display_exam_result(result)


def render_srs(db: DBManager, parent_nav: str = "学習") -> None:
    render_specialized_header(parent_nav, "弱点復習", "srs")
    st.subheader("弱点復習")
    due_df = db.get_due_srs(upcoming_days=1)
    sidebar_alert = st.sidebar.container()
    sidebar_alert.subheader("復習アラート")

    if due_df.empty:
        st.info("今日復習すべき問題はありません。")
        sidebar_alert.success("期限が迫る復習はありません。")
        return

    status_labels = {
        "overdue": "期限切れ",
        "due_today": "本日期限",
        "due_tomorrow": "明日期限",
        "upcoming": "近日期限",
        "unscheduled": "未設定",
        "scheduled": "予定あり",
    }

    status_counts = due_df["due_status"].value_counts().to_dict()
    overdue_count = int(status_counts.get("overdue", 0))
    today_count = int(status_counts.get("due_today", 0))
    tomorrow_count = int(status_counts.get("due_tomorrow", 0))

    if overdue_count:
        st.error(f"{overdue_count}件の復習が期限切れです。すぐに対応しましょう。")
        sidebar_alert.error(f"期限切れ: {overdue_count}件")
    if today_count:
        st.warning(f"本日期限の復習が{today_count}件あります。")
        sidebar_alert.warning(f"本日期限: {today_count}件")
    if tomorrow_count:
        st.info(f"明日が期限の復習が{tomorrow_count}件あります。")
        sidebar_alert.info(f"明日期限: {tomorrow_count}件")
    if not any([overdue_count, today_count, tomorrow_count]):
        sidebar_alert.success("期限が迫る復習はありません。")

    status_order = {"overdue": 0, "due_today": 1, "due_tomorrow": 2, "upcoming": 3, "unscheduled": 4, "scheduled": 5}

    def _normalize_due_date(value: Optional[dt.date]) -> dt.date:
        if value is None or pd.isna(value):
            return dt.date.max
        if isinstance(value, pd.Timestamp):
            return value.date()
        return value

    due_df = due_df.copy()
    due_df["_status_order"] = due_df["due_status"].map(lambda s: status_order.get(s, 99))
    due_df["_due_order"] = due_df["due_date"].apply(_normalize_due_date)
    due_df = due_df.sort_values(["_status_order", "_due_order"]).drop(columns=["_status_order", "_due_order"])

    for _, row in due_df.iterrows():
        question_title = str(row.get("question", ""))
        if len(question_title) > 40:
            question_title = f"{question_title[:40]}..."
        st.markdown(f"### {question_title}")
        due_date = row.get("due_date")
        due_display = due_date if pd.notna(due_date) else "未設定"
        status_label = status_labels.get(row.get("due_status"), "")
        st.write(f"分野: {row['category']} ｜ 期限: {due_display} ({status_label})")
        grade = st.slider(
            f"評価 ({row['question_id']})",
            0,
            5,
            3,
            help="5=完全に覚えた、0=全く覚えていない。評価に応じて次回復習日が変わります。",
        )
        if st.button(
            "評価を保存",
            key=f"srs_save_{row['question_id']}",
            help="SM-2アルゴリズムに基づき次回の出題タイミングを更新します。",
        ):
            initial_ease = st.session_state["settings"].get("sm2_initial_ease", 2.5)
            payload = sm2_update(row, grade, initial_ease=initial_ease)
            db.upsert_srs(row["question_id"], payload)
            st.success("SRSを更新しました")


def render_stats(db: DBManager, df: pd.DataFrame) -> None:
    st.title("分析ダッシュボード")
    attempts = db.get_attempt_stats()
    if attempts.empty:
        st.info("統計情報はまだありません。学習を開始しましょう。")
        return
    try:
        attempts["created_at"] = pd.to_datetime(attempts["created_at"])
        attempts["seconds"] = pd.to_numeric(attempts.get("seconds"), errors="coerce")
        attempts["confidence"] = pd.to_numeric(attempts.get("confidence"), errors="coerce")
    except Exception as exc:
        st.error(f"学習履歴の整形に失敗しました ({exc})")
        st.info("CSVを直接編集した場合は、日付や秒数の列が数値・日時形式になっているか確認してください。")
        return
    question_meta_cols = ["id", "question", "category", "topic", "tags", "difficulty", "year"]
    merged = attempts.merge(
        df[question_meta_cols],
        left_on="question_id",
        right_on="id",
        how="left",
        suffixes=("", "_question"),
    )
    for col in ["category", "topic"]:
        alt_col = f"{col}_question"
        if alt_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[alt_col])
            else:
                merged[col] = merged[alt_col]
            merged = merged.drop(columns=[alt_col])
    if merged.empty:
        st.warning("集計対象の設問が特定できませんでした。設問データが削除されていないか確認してください。")
        st.info("『設定 ＞ データ入出力』でquestions.csvを再度取り込み、設問IDと学習履歴の対応を復元できます。")
        return
    with st.expander("絞り込み", expanded=False):
        filter_cols = st.columns(3)
        category_options = sorted([c for c in merged["category"].dropna().unique()])
        year_options_series = merged.get("year")
        if year_options_series is not None:
            year_numeric = pd.to_numeric(year_options_series, errors="coerce")
            year_options = sorted({str(int(y)) for y in year_numeric.dropna().unique()}, reverse=True)
        else:
            year_options = []
        difficulty_series_full = pd.to_numeric(merged.get("difficulty"), errors="coerce")
        difficulty_options = sorted(difficulty_series_full.dropna().unique().tolist())
        selected_categories = filter_cols[0].multiselect(
            "分野",
            options=category_options,
            default=category_options,
            help="分析に含める分野を選択します。",
        )
        selected_years = filter_cols[1].multiselect(
            "年度",
            options=year_options,
            default=year_options,
            help="学習履歴を集計する年度を選びます。",
        )
        selected_difficulties = filter_cols[2].multiselect(
            "難易度",
            options=difficulty_options,
            default=difficulty_options,
            help="対象とする難易度帯を選択します。",
        )
    filtered = merged.copy()
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if selected_years:
        year_series = filtered.get("year")
        if year_series is not None:
            year_numeric = pd.to_numeric(year_series, errors="coerce")
            year_labels = year_numeric.apply(lambda x: str(int(x)) if pd.notna(x) else None)
            filtered = filtered[year_labels.isin(selected_years)]
        else:
            filtered = filtered.iloc[0:0]
    if selected_difficulties:
        diff_series = pd.to_numeric(filtered.get("difficulty"), errors="coerce")
        filtered = filtered[diff_series.isin(selected_difficulties)]
    if filtered.empty:
        st.info("データが不足しています")
        return
    accuracy_series = filtered["is_correct"].dropna()
    seconds_series = filtered["seconds"].dropna()
    confidence_series = filtered["confidence"].dropna()
    accuracy = accuracy_series.mean() if not accuracy_series.empty else np.nan
    avg_seconds = seconds_series.mean() if not seconds_series.empty else np.nan
    avg_confidence = confidence_series.mean() if not confidence_series.empty else np.nan
    st.subheader("サマリー")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("挑戦回数", f"{len(filtered)} 回")
    with col2:
        accuracy_text = f"{accuracy * 100:.1f}%" if not np.isnan(accuracy) else "--"
        st.metric("平均正答率", accuracy_text)
    with col3:
        st.metric("平均解答時間", f"{avg_seconds:.1f} 秒" if not np.isnan(avg_seconds) else "--")
    if not np.isnan(avg_confidence):
        st.caption(f"平均確信度: {avg_confidence:.1f}%")
    else:
        st.caption("平均確信度: -- (十分なデータがありません)")

    import altair as alt

    st.subheader("学習時間と挑戦回数の推移")
    freq = st.selectbox("集計粒度", ["日次", "週次"], index=0, help="学習時間と挑戦回数の推移を集計する粒度を切り替えます。")
    timeline = filtered.copy()
    if freq == "週次":
        timeline["period"] = timeline["created_at"].dt.to_period("W").dt.start_time
    else:
        timeline["period"] = timeline["created_at"].dt.normalize()
    timeline_grouped = (
        timeline.groupby("period")
        .agg(
            attempts_count=("is_correct", "count"),
            total_seconds=("seconds", "sum"),
        )
        .reset_index()
        .rename(columns={"period": "date"})
    )
    timeline_grouped = timeline_grouped.sort_values("date")
    timeline_grouped["学習時間 (分)"] = timeline_grouped["total_seconds"].fillna(0) / 60
    timeline_grouped["挑戦回数"] = timeline_grouped["attempts_count"].fillna(0)
    if timeline_grouped.empty:
        st.info("データが不足しています")
    else:
        try:
            time_base = alt.Chart(timeline_grouped).encode(
                x=alt.X("date:T", title="日付"),
                tooltip=[
                    alt.Tooltip("date:T", title="日付"),
                    alt.Tooltip("挑戦回数", format=","),
                    alt.Tooltip("学習時間 (分)", format=".1f"),
                ],
            )
            attempts_layer = time_base.mark_bar(opacity=0.5, color="#2563eb").encode(
                y=alt.Y("挑戦回数:Q", title="挑戦回数")
            )
            minutes_layer = time_base.mark_line(point=True, color="#f97316").encode(
                y=alt.Y("学習時間 (分):Q", title="学習時間 (分)", axis=alt.Axis(titleColor="#f97316"))
            )
            time_chart = alt.layer(attempts_layer, minutes_layer).resolve_scale(y="independent")
            st.altair_chart(time_chart, use_container_width=True)
        except Exception as exc:
            st.warning(f"学習時間の推移グラフを表示できませんでした ({exc})")
            st.caption("十分なデータが集まると自動で表示されます。")

    st.subheader("分野別分析")
    category_stats = (
        filtered.groupby("category")
        .agg(
            accuracy=("is_correct", "mean"),
            avg_seconds=("seconds", "mean"),
            attempts_count=("is_correct", "count"),
        )
        .reset_index()
    )
    if category_stats.empty:
        st.info("分野情報の十分なデータがありません。questions.csv の category 列を確認してください。")
    else:
        try:
            accuracy_chart = (
                alt.Chart(category_stats)
                .mark_bar()
                .encode(
                    x=alt.X("category", title="分野"),
                    y=alt.Y("accuracy", title="正答率", axis=alt.Axis(format="%")),
                    tooltip=["category", alt.Tooltip("accuracy", format=".2%"), "attempts_count"],
                )
                .properties(height=320)
            )
            st.altair_chart(accuracy_chart, use_container_width=True)
        except Exception as exc:
            st.warning(f"分野別正答率グラフを表示できませんでした ({exc})")
            st.caption("十分なデータが集まると自動で表示されます。")
        try:
            time_chart = (
                alt.Chart(category_stats)
                .mark_line(point=True)
                .encode(
                    x=alt.X("category", title="分野"),
                    y=alt.Y("avg_seconds", title="平均解答時間 (秒)", scale=alt.Scale(zero=False)),
                    tooltip=["category", alt.Tooltip("avg_seconds", format=".1f"), "attempts_count"],
                )
            )
            st.altair_chart(time_chart, use_container_width=True)
        except Exception as exc:
            st.warning(f"分野別時間グラフを表示できませんでした ({exc})")
            st.caption("十分なデータが集まると自動で表示されます。")

    st.subheader("正答率が低い論点")
    topic_stats = (
        filtered.dropna(subset=["topic"])
        .groupby(["category", "topic"])
        .agg(
            accuracy=("is_correct", "mean"),
            attempts_count=("is_correct", "count"),
        )
        .reset_index()
    )
    low_accuracy = (
        topic_stats[topic_stats["attempts_count"] >= 3]
        .sort_values("accuracy")
        .head(10)
        .copy()
    )
    if low_accuracy.empty:
        st.info("十分なデータがありません。学習を重ねて傾向を確認しましょう。")
    else:
        low_accuracy["正答率"] = (low_accuracy["accuracy"] * 100).round(1)
        display_cols = low_accuracy[["category", "topic", "attempts_count", "正答率"]]
        display_cols = display_cols.rename(columns={"category": "分野", "topic": "論点", "attempts_count": "挑戦回数"})
        st.dataframe(display_cols, use_container_width=True)
        if st.button("復習モードで重点復習", type="primary"):
            st.session_state["nav"] = "弱点復習"
            st.session_state["_nav_widget"] = "弱点復習"
            safe_rerun()

    st.subheader("確信度と正答の相関")
    valid_conf = filtered.dropna(subset=["confidence"])
    if valid_conf.empty:
        st.info("確信度データがまだ十分ではありません。学習時にスライダーで自己評価してみましょう。")
    else:
        corr = valid_conf["confidence"].corr(valid_conf["is_correct"])
        st.metric("相関係数", f"{corr:.2f}")
        try:
            scatter = (
                alt.Chart(valid_conf)
                .mark_circle(opacity=0.6)
                .encode(
                    x=alt.X("confidence", title="確信度 (%)"),
                    y=alt.Y("is_correct", title="正答 (1=正解)", scale=alt.Scale(domain=[-0.1, 1.1])),
                    color=alt.Color("category", legend=None),
                    tooltip=["category", "topic", "confidence", "is_correct", "seconds"],
                )
            )
            st.altair_chart(scatter, use_container_width=True)
        except Exception as exc:
            st.warning(f"相関散布図を表示できませんでした ({exc})")
            st.caption("十分なデータが集まると自動で表示されます。")

    st.subheader("ひっかけ語彙ヒートマップ")
    heatmap_df = compute_tricky_vocab_heatmap(filtered, df)
    if heatmap_df.empty:
        st.info("誤答語彙の十分なデータがありません。")
    else:
        try:
            word_order = (
                heatmap_df.groupby("word")["count"].sum().sort_values(ascending=False).index.tolist()
            )
            heatmap = (
                alt.Chart(heatmap_df)
                .mark_rect()
                .encode(
                    x=alt.X("category", title="分野"),
                    y=alt.Y("word", title="語彙", sort=word_order),
                    color=alt.Color("count", title="誤答回数", scale=alt.Scale(scheme="reds")),
                    tooltip=["word", "category", "count"],
                )
            )
            st.altair_chart(heatmap, use_container_width=True)
        except Exception as exc:
            st.warning(f"語彙ヒートマップを表示できませんでした ({exc})")
            st.caption("十分なデータが集まると自動で表示されます。")

    st.subheader("最も改善した論点")
    improvement = compute_most_improved_topic(filtered, df)
    if improvement:
        st.success(
            f"{improvement['topic']}：正答率が {(improvement['early'] * 100):.1f}% → {(improvement['late'] * 100):.1f}% (＋{improvement['delta'] * 100:.1f}ポイント)"
        )
    else:
        st.info("改善の傾向を示す論点はまだ検出されていません。継続して学習しましょう。")


def select_uploaded_file_by_name(
    files: Sequence["UploadedFile"], keyword: str
) -> Optional["UploadedFile"]:
    keyword_lower = keyword.lower()
    for file in files:
        if keyword_lower in file.name.lower():
            return file
    return None


def execute_quick_import(
    db: DBManager,
    questions_file: Optional["UploadedFile"],
    answers_file: Optional["UploadedFile"],
) -> None:
    quick_errors: List[str] = []
    questions_df: Optional[pd.DataFrame] = None
    answers_df: Optional[pd.DataFrame] = None
    if questions_file is None and answers_file is None:
        st.warning("questions.csv か answers.csv のいずれかを選択してください。")
        return

    if questions_file is not None:
        data = questions_file.getvalue()
        try:
            questions_df = pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            questions_df = pd.read_csv(io.BytesIO(data), encoding="cp932")
        quick_errors.extend(validate_question_records(questions_df))

    if answers_file is not None:
        data = answers_file.getvalue()
        try:
            answers_df = pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            answers_df = pd.read_csv(io.BytesIO(data), encoding="cp932")
        quick_errors.extend(validate_answer_records(answers_df))

    if quick_errors:
        for err in quick_errors:
            st.error(err)
        st.info("テンプレートの列構成と突合してください。『テンプレートをダウンロード』から最新のCSVサンプルを取得できます。")
        return

    policy = {"explanation": "overwrite", "tags": "merge"}
    merged_df: Optional[pd.DataFrame] = None
    rejects_q = pd.DataFrame()
    rejects_a = pd.DataFrame()
    conflicts = pd.DataFrame()
    normalization_failed = False

    if questions_df is not None:
        try:
            normalized_q = normalize_questions(questions_df)
        except Exception as exc:
            st.error(f"questions.csv の整形に失敗しました: {exc}")
            normalization_failed = True
            normalized_q = None
    else:
        normalized_q = None

    if answers_df is not None:
        try:
            normalized_a = normalize_answers(answers_df)
        except Exception as exc:
            st.error(f"answers.csv の整形に失敗しました: {exc}")
            normalization_failed = True
            normalized_a = None
    else:
        normalized_a = None

    if normalization_failed:
        st.warning("列名や値の形式を見直してから再度インポートしてください。")
        return

    if normalized_q is not None and normalized_a is not None:
        merged_df, rejects_q, rejects_a, conflicts = merge_questions_answers(
            normalized_q, normalized_a, policy=policy
        )
    elif normalized_q is not None:
        merged_df = normalized_q
    elif normalized_a is not None:
        existing = load_questions_df()
        if existing.empty:
            st.error("設問データが存在しません。answers.csv を取り込む前に questions.csv を読み込んでください。")
        else:
            merged_df, rejects_q, rejects_a, conflicts = merge_questions_answers(
                existing, normalized_a, policy=policy
            )

    if merged_df is None:
        return

    inserted, updated = db.upsert_questions(merged_df)
    rebuild_tfidf_cache()
    st.success(f"クイックインポートが完了しました。追加 {inserted} 件 / 更新 {updated} 件")

    if not rejects_q.empty or not rejects_a.empty:
        st.warning(
            f"取り込めなかったレコードがあります。questions: {len(rejects_q)} 件 / answers: {len(rejects_a)} 件"
        )
        with st.expander("取り込めなかった行の詳細", expanded=False):
            if not rejects_q.empty:
                st.markdown("**questions.csv**")
                st.dataframe(rejects_q.head(20))
            if not rejects_a.empty:
                st.markdown("**answers.csv**")
                st.dataframe(rejects_a.head(20))
            st.caption("理由列を参考にCSVの該当行を修正してください。全件はrejects_*.csvでダウンロードできます。")

    if not conflicts.empty:
        st.info(f"正答の衝突が {len(conflicts)} 件あり、上書きしました。")


def render_quick_import_controls(
    db: DBManager,
    *,
    key_prefix: str,
    heading: Optional[str] = None,
    initial_files: Optional[Sequence["UploadedFile"]] = None,
) -> None:
    if heading:
        st.markdown(heading)

    st.caption("questions.csv と answers.csv をまとめてドラッグ＆ドロップできます。必要な方だけでも取り込めます。")

    uploaded_files = st.file_uploader(
        "questions.csv / answers.csv をアップロード",
        type=["csv"],
        accept_multiple_files=True,
        key=f"{key_prefix}_quick_import_files",
        help="複数ファイルを同時に選択すると自動で候補に入ります。",
    )

    combined_files: List["UploadedFile"] = []
    seen_names: Set[str] = set()

    def add_file(file: "UploadedFile") -> None:
        if file.name not in seen_names:
            combined_files.append(file)
            seen_names.add(file.name)

    if initial_files:
        for file in initial_files:
            add_file(file)
        st.caption("ドロップゾーンに追加したファイルが候補として表示されています。")

    if uploaded_files:
        for file in uploaded_files:
            add_file(file)

    option_files = combined_files
    question_default = select_uploaded_file_by_name(option_files, "questions") or select_uploaded_file_by_name(option_files, "question")
    answer_default = select_uploaded_file_by_name(option_files, "answers") or select_uploaded_file_by_name(option_files, "answer")

    options = ["選択しない"] + [file.name for file in option_files]

    def get_default_index(default_file: Optional["UploadedFile"]) -> int:
        if default_file is None:
            return 0
        try:
            return option_files.index(default_file) + 1
        except ValueError:
            return 0

    question_index = get_default_index(question_default)
    answer_index = get_default_index(answer_default)

    if not option_files:
        st.caption("ファイルを選択するとここに一覧表示されます。上部のドロップゾーンか右のボタンから追加してください。")

    question_selection = st.selectbox(
        "questions.csv", options, index=question_index, key=f"{key_prefix}_quick_import_question"
    )
    answer_selection = st.selectbox(
        "answers.csv", options, index=answer_index, key=f"{key_prefix}_quick_import_answer"
    )

    def resolve_selection(selection: str) -> Optional["UploadedFile"]:
        if selection == "選択しない":
            return None
        for file in option_files:
            if file.name == selection:
                return file
        return None

    selected_questions = resolve_selection(question_selection)
    selected_answers = resolve_selection(answer_selection)

    if st.button("クイックインポート実行", key=f"{key_prefix}_quick_import_button"):
        execute_quick_import(db, selected_questions, selected_answers)


def render_history_export_controls(
    db: DBManager,
    *,
    heading: Optional[str] = None,
) -> None:
    if heading:
        st.markdown(heading)

    with db.engine.connect() as conn:
        attempts_df = pd.read_sql(select(attempts_table), conn)
        exams_df = pd.read_sql(select(exams_table), conn)

    if not attempts_df.empty:
        buffer = io.StringIO()
        attempts_df.to_csv(buffer, index=False)
        st.download_button("attempts.csv をダウンロード", buffer.getvalue(), file_name="attempts.csv", mime="text/csv")
    else:
        st.caption("attempts.csv：学習履歴はまだありません。学習モードで解答するとダウンロード可能になります。")

    if not exams_df.empty:
        buffer = io.StringIO()
        exams_df.to_csv(buffer, index=False)
        st.download_button("exams.csv をダウンロード", buffer.getvalue(), file_name="exams.csv", mime="text/csv")
    else:
        st.caption("exams.csv：模試の受験履歴はまだありません。模試モードで本試験を体験しましょう。")

    if DB_PATH.exists():
        st.download_button("SQLiteバックアップをダウンロード", DB_PATH.read_bytes(), file_name="takken.db")


def render_data_io(db: DBManager, parent_nav: str = "設定") -> None:
    render_specialized_header(parent_nav, "データ入出力", "data_io")
    st.subheader("データ入出力")
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    import_notifications = st.session_state.setdefault("import_notifications", [])
    if import_notifications:
        st.markdown("### インポート履歴 (このセッション)")
        history_df = pd.DataFrame(import_notifications)
        ordered_columns = [
            col
            for col in ["timestamp", "inserted", "updated", "rejected", "seconds"]
            if col in history_df.columns
        ]
        display_df = history_df[ordered_columns].rename(
            columns={
                "timestamp": "完了時刻",
                "inserted": "追加",
                "updated": "更新",
                "rejected": "リジェクト",
                "seconds": "処理秒数",
            }
        )
        st.dataframe(display_df, use_container_width=True)
    st.markdown("### テンプレートファイル")
    st.download_button(
        "テンプレートをダウンロード (ZIP)",
        data=get_template_archive(),
        file_name=f"takken_templates_{timestamp}.zip",
        mime="application/zip",
    )
    st.caption("設問・正答データのCSV/XLSXテンプレートが含まれます。必要に応じて編集してご利用ください。")
    if SCHEMA_GUIDE_PATH.exists():
        st.markdown(
            "📘 データ列の詳細仕様は下記のスキーマガイドで確認できます。テンプレート編集前にご覧ください。"
        )
        with st.expander("questions.csv / answers.csv / law_revision.csv のスキーマガイド"):
            st.markdown(SCHEMA_GUIDE_PATH.read_text(encoding="utf-8"))
    sample_cols = st.columns(4)
    with sample_cols[0]:
        st.download_button(
            "サンプル questions.csv",
            data=build_sample_questions_csv(),
            file_name="sample_questions.csv",
            mime="text/csv",
            help="Excelで開いて値を上書きすれば、そのまま取り込みできます。",
        )
    with sample_cols[1]:
        st.download_button(
            "サンプル answers.csv",
            data=build_sample_answers_csv(),
            file_name="sample_answers.csv",
            mime="text/csv",
            help="正答番号や解説の記入例です。コピーしてご利用ください。",
        )
    with sample_cols[2]:
        st.download_button(
            "サンプル predicted.csv",
            data=build_sample_predicted_csv(),
            file_name="sample_predicted.csv",
            mime="text/csv",
            help="予想問題の入力例です。ラベルや出典を記入して活用ください。",
        )
    with sample_cols[3]:
        st.download_button(
            "サンプル law_revision.csv",
            data=build_sample_law_revision_csv(),
            file_name="sample_law_revision.csv",
            mime="text/csv",
            help="最新の法改正論点を整理した問題サンプルです。改正年度や施行日を追記してご利用ください。",
        )
    st.markdown("### 外部サービス同期")
    integrations = st.session_state["settings"].setdefault("integrations", {})
    integration_status = st.session_state.setdefault(
        "integration_status",
        {
            "google_calendar": {"last_synced": None, "message": "未同期", "success": False},
            "notion": {"last_synced": None, "message": "未同期", "success": False},
        },
    )
    google_settings = integrations.get("google_calendar", {})
    google_status = integration_status.setdefault(
        "google_calendar", {"last_synced": None, "message": "未同期", "success": False}
    )
    notion_settings = integrations.get("notion", {})
    notion_status = integration_status.setdefault(
        "notion", {"last_synced": None, "message": "未同期", "success": False}
    )
    sync_cols = st.columns(2)
    with sync_cols[0]:
        st.markdown("#### Google Calendar")
        last_synced = google_status.get("last_synced")
        message = google_status.get("message", "未同期")
        if last_synced:
            st.caption(f"最終同期: {last_synced} / {message}")
        else:
            st.caption(f"状態: {message}")
        allow_resync = False
        if google_status.get("success"):
            allow_resync = st.checkbox("再同期を許可", key="google_calendar_resync")
        google_disabled = google_status.get("success") and not allow_resync
        if st.button("スケジュールをGoogle Calendarへ同期", disabled=google_disabled):
            due_df = db.get_due_srs()
            if due_df.empty:
                msg = "同期対象の学習スケジュールがありません。"
                google_status.update({"message": msg, "success": False})
                st.info(msg)
            else:
                credentials = OAuthCredentials(
                    client_id=google_settings.get("client_id", ""),
                    client_secret=google_settings.get("client_secret", ""),
                    redirect_uri=google_settings.get("redirect_uri", ""),
                    access_token=google_settings.get("access_token", ""),
                    refresh_token=google_settings.get("refresh_token", ""),
                )
                client = GoogleCalendarClient(
                    GoogleCalendarConfig(credentials=credentials, calendar_id=google_settings.get("calendar_id", "primary"))
                )
                try:
                    result = client.sync_study_schedule(due_df)
                except IntegrationConfigError as exc:
                    msg = str(exc)
                    google_status.update({"message": msg, "success": False})
                    st.error(msg)
                except IntegrationError as exc:
                    msg = str(exc)
                    google_status.update({"message": msg, "success": False})
                    st.error(msg)
                else:
                    now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    detail = f"イベント {result['created']} 件を同期しました。"
                    if result.get("failures"):
                        detail += f" 一部失敗 {len(result['failures'])} 件。"
                    google_status.update({"message": detail, "success": True, "last_synced": now_text})
                    st.success(detail)
    with sync_cols[1]:
        st.markdown("#### Notion")
        last_synced = notion_status.get("last_synced")
        message = notion_status.get("message", "未同期")
        if last_synced:
            st.caption(f"最終同期: {last_synced} / {message}")
        else:
            st.caption(f"状態: {message}")
        allow_resync = False
        if notion_status.get("success"):
            allow_resync = st.checkbox("再同期を許可", key="notion_resync")
        notion_disabled = notion_status.get("success") and not allow_resync
        notion_days = st.slider("送信対象日数", min_value=1, max_value=30, value=7, key="notion_sync_days")
        if st.button("学習ログをNotionデータベースへ送信", disabled=notion_disabled):
            attempts = db.get_attempt_stats()
            summaries = build_notion_summaries(attempts, days=notion_days)
            client = NotionClient(
                NotionConfig(
                    integration_token=notion_settings.get("integration_token", ""),
                    database_id=notion_settings.get("database_id", ""),
                    notion_version=notion_settings.get("notion_version", "2022-06-28"),
                )
            )
            if not summaries:
                msg = "送信対象の学習ログがありません。"
                notion_status.update({"message": msg, "success": False})
                st.info(msg)
            else:
                try:
                    result = client.sync_learning_log(summaries)
                except IntegrationConfigError as exc:
                    msg = str(exc)
                    notion_status.update({"message": msg, "success": False})
                    st.error(msg)
                except IntegrationError as exc:
                    msg = str(exc)
                    notion_status.update({"message": msg, "success": False})
                    st.error(msg)
                else:
                    now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    detail = f"{result['created']} 日分の学習ログを送信しました。"
                    if result.get("failures"):
                        detail += f" 送信失敗 {len(result['failures'])} 件。"
                    notion_status.update({"message": detail, "success": True, "last_synced": now_text})
                    st.success(detail)
    st.markdown("### 学習時間ログ取り込み")
    st.caption("スマートウォッチや外部記録ツールから出力したCSV/JSONをアップロードすると attempts に反映します。question_id、または年度と問番号で問題を特定します。")
    log_files = st.file_uploader(
        "学習ログファイルを選択",
        type=["csv", "json"],
        accept_multiple_files=True,
        key="external_log_uploader",
    )
    if log_files and st.button("学習ログを登録", key="external_log_apply"):
        questions_df = load_questions_df()
        total_inserted = 0
        parse_errors: List[str] = []
        for log_file in log_files:
            records, errors = parse_external_attempt_logs(log_file, questions_df)
            parse_errors.extend(errors)
            if records:
                total_inserted += db.bulk_insert_attempts(records)
        if total_inserted:
            st.success(f"{total_inserted}件の学習ログを追加しました。")
        if parse_errors:
            for err in parse_errors:
                st.warning(err)
    st.caption("サンプルCSVはExcelに貼り付けて使えるよう列幅を調整済みです。コピー&ペーストで手早く登録できます。")
    st.markdown("### 法改正自動更新")
    st.caption("不動産適正取引推進機構や専門予備校の公開フィードを HTTP 経由で取得し、自動で法改正問題を生成・登録します。")
    if st.button("最新の法改正フィードを取得", key="law_revision_sync_button"):
        service = get_law_update_service(db)
        with st.spinner("最新情報を取得しています..."):
            results = service.run(db)
        success_results = [r for r in results if r.status in {"success", "empty"}]
        if any(r.status == "success" for r in success_results):
            st.success("自動取得と問題生成が完了しました。『法改正対策』で確認してください。")
        elif success_results:
            st.info("フィードを取得しましたが新しい問題は生成されませんでした。")
        else:
            errors = [r for r in results if r.status == "error"]
            if errors:
                st.error("外部フィードの取得に失敗しました。ログを確認してください。")
    sync_history = db.load_law_revision_sync_logs(limit=10)
    if sync_history.empty:
        st.info("自動取得履歴はまだありません。上のボタンから同期を開始してください。")
    else:
        history_df = sync_history.copy()
        history_df["fetched_at"] = pd.to_datetime(history_df["fetched_at"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            history_df[["fetched_at", "source", "status", "revisions_detected", "questions_generated", "message"]],
            use_container_width=True,
        )
    render_quick_import_controls(
        db,
        key_prefix="settings",
        heading="### クイックインポート (questions.csv / answers.csv)",
    )

    st.markdown("### 予想問題インポート (predicted.csv)")
    predicted_file = st.file_uploader(
        "predicted.csv をアップロード",
        type=["csv"],
        key="predicted_file_uploader",
        help="予想問題データをCSV形式でアップロードします。選択肢や解説を含めたテンプレートに対応しています。",
    )
    if st.button("予想問題インポート実行", key="predicted_import_button"):
        if predicted_file is None:
            st.warning("predicted.csv を選択してください。")
        else:
            data = predicted_file.getvalue()
            try:
                predicted_df = pd.read_csv(io.BytesIO(data))
            except UnicodeDecodeError:
                predicted_df = pd.read_csv(io.BytesIO(data), encoding="cp932")
            errors = validate_predicted_records(predicted_df)
            if errors:
                for err in errors:
                    st.error(err)
                st.info("テンプレートの列構成に合わせて再度アップロードしてください。")
            else:
                try:
                    normalized_predicted = normalize_predicted_questions(predicted_df)
                except Exception as exc:
                    st.error(f"predicted.csv の整形に失敗しました: {exc}")
                else:
                    inserted, updated = db.upsert_predicted_questions(normalized_predicted)
                    st.success(f"予想問題データを取り込みました。追加 {inserted} 件 / 更新 {updated} 件")
                    st.session_state.pop("predicted_session", None)
                    if not normalized_predicted.empty:
                        st.dataframe(
                            normalized_predicted.head(10)[
                                [
                                    "label",
                                    "category",
                                    "topic",
                                    "source",
                                    "question",
                                    "correct",
                                ]
                            ],
                            use_container_width=True,
                        )
                        st.caption("取り込んだ予想問題の先頭10件を表示しています。")
    st.markdown("### 法改正予想問題インポート (law_revision.csv)")
    law_revision_file = st.file_uploader(
        "law_revision.csv をアップロード",
        type=["csv"],
        key="law_revision_file_uploader",
        help="ここ数年の法改正に関する予想問題データをCSVで読み込みます。改正年度や施行日も登録できます。",
    )
    if st.button("法改正予想問題インポート実行", key="law_revision_import_button"):
        if law_revision_file is None:
            st.warning("law_revision.csv を選択してください。")
        else:
            data = law_revision_file.getvalue()
            try:
                law_revision_df = pd.read_csv(io.BytesIO(data))
            except UnicodeDecodeError:
                law_revision_df = pd.read_csv(io.BytesIO(data), encoding="cp932")
            errors = validate_law_revision_records(law_revision_df)
            if errors:
                for err in errors:
                    st.error(err)
                st.info("テンプレートの列構成に合わせて再度アップロードしてください。")
            else:
                try:
                    normalized_law_revision = normalize_law_revision_questions(law_revision_df)
                except Exception as exc:
                    st.error(f"law_revision.csv の整形に失敗しました: {exc}")
                else:
                    inserted, updated = db.upsert_law_revision_questions(normalized_law_revision)
                    st.success(
                        f"法改正予想問題データを取り込みました。追加 {inserted} 件 / 更新 {updated} 件"
                    )
                    st.session_state.pop("law_revision_session", None)
                    if not normalized_law_revision.empty:
                        preview_cols = [
                            "label",
                            "law_name",
                            "revision_year",
                            "effective_date",
                            "question",
                            "correct",
                        ]
                        st.dataframe(
                            normalized_law_revision.head(10)[preview_cols],
                            use_container_width=True,
                        )
                        st.caption("取り込んだ法改正予想問題の先頭10件を表示しています。")
    st.markdown("### クイックエクスポート (questions.csv / answers.csv)")
    existing_questions = load_questions_df()
    if existing_questions.empty:
        st.info("エクスポート可能な設問データがありません。")
    else:
        question_cols = QUESTION_TEMPLATE_COLUMNS.copy()
        if "id" in existing_questions.columns and "id" not in question_cols:
            question_cols.append("id")
        q_export = existing_questions[question_cols]
        q_buffer = io.StringIO()
        q_export.to_csv(q_buffer, index=False)
        st.download_button(
            "questions.csv をダウンロード",
            q_buffer.getvalue(),
            file_name="questions.csv",
            mime="text/csv",
            key="export_questions_csv",
        )
        answers_export = build_answers_export(existing_questions)
        a_buffer = io.StringIO()
        answers_export.to_csv(a_buffer, index=False)
        st.download_button(
            "answers.csv をダウンロード",
            a_buffer.getvalue(),
            file_name="answers.csv",
            mime="text/csv",
            key="export_answers_csv",
        )
    st.markdown("### (1) ファイル選択")
    uploaded_files = st.file_uploader(
        "設問・解答ファイルを選択 (CSV/XLSX/ZIP)",
        type=["csv", "xlsx", "xls", "zip"],
        accept_multiple_files=True,
    )
    datasets = []
    file_summaries: List[Dict[str, object]] = []
    if uploaded_files:
        for file in uploaded_files:
            file_summary = {
                "file": file.name,
                "questions": 0,
                "answers": 0,
            }
            try:
                store_uploaded_file(file, timestamp)
                for name, df in decode_uploaded_file(file):
                    kind = guess_dataset_kind(df)
                    datasets.append({"name": name, "data": df, "kind": kind})
                    if kind == MAPPING_KIND_QUESTIONS:
                        file_summary["questions"] += len(df)
                    elif kind == MAPPING_KIND_ANSWERS:
                        file_summary["answers"] += len(df)
            except Exception as e:
                st.error(f"{file.name}: 読み込みに失敗しました ({e})")
            else:
                file_summaries.append(file_summary)
                st.caption(
                    f"{file_summary['file']}: 設問 {file_summary['questions']} 件 / 正答 {file_summary['answers']} 件"
                )
    if not datasets:
        st.info("ファイルをアップロードしてください。")
        return

    st.markdown("### (2) プレビュー & マッピング")
    mapping_profiles = db.fetch_mapping_profiles()
    profile_options = ["新規マッピング"] + (mapping_profiles["name"].tolist() if not mapping_profiles.empty else [])
    selected_profile = st.selectbox("マッピングテンプレート", profile_options)
    profile_mapping: Dict[str, Dict[str, str]] = {}
    if selected_profile != "新規マッピング" and not mapping_profiles.empty:
        profile_row = mapping_profiles[mapping_profiles["name"] == selected_profile].iloc[0]
        mapping_data = profile_row["mapping_json"]
        if isinstance(mapping_data, str):
            try:
                profile_mapping = json.loads(mapping_data)
            except json.JSONDecodeError:
                profile_mapping = {}
        else:
            profile_mapping = mapping_data

    normalized_question_frames = []
    normalized_answer_frames = []
    conflict_resolutions: List[Dict[str, object]] = []

    policy = {
        "explanation": st.selectbox("解説の取り扱い", ["overwrite", "append"], format_func=lambda x: "上書き" if x == "overwrite" else "追記"),
        "tags": st.selectbox("タグの取り扱い", ["merge", "overwrite"], format_func=lambda x: "結合" if x == "merge" else "上書き"),
    }

    for dataset in datasets:
        df = dataset["data"]
        display_name, original_name = describe_dataset_name(dataset["name"])
        st.subheader(display_name)
        if original_name and original_name != display_name:
            st.caption(f"元のファイル/シート名: {original_name}")
        st.dataframe(df.head())
        kind = st.selectbox(
            f"データ種別 ({display_name})",
            [MAPPING_KIND_QUESTIONS, MAPPING_KIND_ANSWERS],
            index=0 if dataset["kind"] == MAPPING_KIND_QUESTIONS else 1,
            format_func=lambda value: MAPPING_KIND_LABELS.get(value, value),
            help="アップロードした表が設問データか正答データかを選択してください。",
        )
        dataset["kind"] = kind
        columns = df.columns.tolist()
        lower_map = {col.lower(): col for col in columns}
        if kind == MAPPING_KIND_QUESTIONS:
            mapping_targets = {
                "year": "年度",
                "q_no": "問番",
                "category": "大分類",
                "topic": "小分類",
                "question": "問題文",
                "choice1": "選択肢1",
                "choice2": "選択肢2",
                "choice3": "選択肢3",
                "choice4": "選択肢4",
                "explanation": "解説",
                "difficulty": "難易度",
                "tags": "タグ",
                "id": "ID",
            }
        else:
            mapping_targets = {
                "year": "年度",
                "q_no": "問番",
                "correct_number": "正答番号",
                "correct_label": "正答ラベル",
                "correct_text": "正答テキスト",
                "explanation": "解説",
                "difficulty": "難易度",
                "tags": "タグ",
            }
        saved_mapping = profile_mapping.get(dataset["name"], {}) if profile_mapping else {}
        mapping = {}
        for key, label in mapping_targets.items():
            default_idx = -1
            if saved_mapping and key in saved_mapping and saved_mapping[key] in columns:
                default_idx = columns.index(saved_mapping[key])
            elif key in lower_map:
                default_idx = columns.index(lower_map[key])
            selected_col = st.selectbox(
                f"{label}",
                ["未設定"] + columns,
                index=default_idx + 1 if default_idx >= 0 else 0,
                key=f"map_{dataset['name']}_{key}",
            )
            if selected_col != "未設定":
                mapping[key] = selected_col
        try:
            if kind == MAPPING_KIND_QUESTIONS:
                normalized = normalize_questions(df, mapping=mapping)
                normalized_question_frames.append(normalized)
            else:
                normalized = normalize_answers(df, mapping=mapping)
                normalized_answer_frames.append(normalized)
            dataset["mapping"] = mapping
        except Exception as e:
            st.error(f"マッピングエラー: {e}")

    if st.checkbox("マッピングをテンプレート保存"):
        profile_name = st.text_input("テンプレート名")
        if profile_name and st.button("保存"):
            mapping_payload = {ds["name"]: ds.get("mapping", {}) for ds in datasets}
            db.save_mapping_profile(profile_name, "mixed", mapping_payload)
            st.success("マッピングテンプレートを保存しました")

    if not normalized_question_frames:
        st.warning("設問データがありません。")
        return

    merged_questions = pd.concat(normalized_question_frames).drop_duplicates(subset=["id"])
    merged_answers = pd.concat(normalized_answer_frames) if normalized_answer_frames else pd.DataFrame()

    if not merged_answers.empty:
        merged, rejects_q, rejects_a, conflicts = merge_questions_answers(merged_questions, merged_answers, policy)
    else:
        merged = merged_questions
        rejects_q = pd.DataFrame()
        rejects_a = pd.DataFrame()
        conflicts = pd.DataFrame()

    if not conflicts.empty:
        st.error("正答情報のコンフリクトがあります。解決方法を選択してください。")
        with st.form("conflict_resolution_form"):
            for _, conflict in conflicts.iterrows():
                st.write(f"{int(conflict['year'])}年 問{int(conflict['q_no'])}")
                action = st.selectbox(
                    f"処理方法 ({conflict['id']})",
                    ["既存を維持", "解答で上書き", "手動修正"],
                    key=f"conflict_action_{conflict['id']}",
                )
                manual_value = st.number_input(
                    f"手動正答番号 ({conflict['id']})",
                    min_value=1,
                    max_value=4,
                    value=int(conflict["existing"]) if pd.notna(conflict["existing"]) and int(conflict["existing"]) in [1, 2, 3, 4] else 1,
                    key=f"conflict_manual_{conflict['id']}",
                )
                conflict_resolutions.append(
                    {
                        "id": conflict["id"],
                        "action": action,
                        "manual": manual_value,
                        "incoming": conflict["incoming"],
                        "existing": conflict["existing"],
                    }
                )
            applied = st.form_submit_button("解決を適用")
        if not applied:
            st.stop()
        for resolution in conflict_resolutions:
            if resolution["action"] == "解答で上書き":
                merged.loc[merged["id"] == resolution["id"], "correct"] = resolution["incoming"]
            elif resolution["action"] == "手動修正":
                merged.loc[merged["id"] == resolution["id"], "correct"] = resolution["manual"]
        conflicts = pd.DataFrame()

    st.markdown("### (3) 正規化 & バリデーション結果")
    st.success(f"設問{len(merged)}件を取り込みます。")
    if not rejects_a.empty:
        buffer = io.StringIO()
        rejects_a.to_csv(buffer, index=False)
        st.download_button("rejects_answers.csv をダウンロード", buffer.getvalue(), file_name="rejects_answers.csv", mime="text/csv")
    if not rejects_q.empty:
        buffer = io.StringIO()
        rejects_q.to_csv(buffer, index=False)
        st.download_button("rejects_questions.csv をダウンロード", buffer.getvalue(), file_name="rejects_questions.csv", mime="text/csv")

    if st.button("(4) 統合 (UPSERT) 実行"):
        started = dt.datetime.now()
        progress = st.progress(0.0)

        def handle_progress(processed: int, total: int) -> None:
            if total <= 0:
                ratio = 1.0
            else:
                ratio = processed / total
            progress.progress(min(max(ratio, 0.0), 1.0))

        inserted, updated = db.bulk_upsert_questions(
            merged, batch_size=200, on_progress=handle_progress
        )
        finished = dt.datetime.now()
        seconds = (finished - started).total_seconds()
        policy_payload = {**policy, "conflict_resolutions": conflict_resolutions}
        db.log_import(
            {
                "started_at": started,
                "finished_at": finished,
                "files": len(uploaded_files),
                "inserted": inserted,
                "updated": updated,
                "rejected": len(rejects_a) + len(rejects_q),
                "conflicts": len(conflicts),
                "seconds": seconds,
                "policy": json.dumps(policy_payload, ensure_ascii=False),
            }
        )
        rebuild_tfidf_cache()
        progress.progress(1.0)
        rejected_total = len(rejects_a) + len(rejects_q)
        st.success(
            "インポートが完了しました。"
            f" 追加 {inserted} 件 / 更新 {updated} 件 / リジェクト {rejected_total} 件。"
            " TF-IDFを再構築しました。"
        )
        notification = {
            "timestamp": finished.strftime("%Y-%m-%d %H:%M:%S"),
            "inserted": inserted,
            "updated": updated,
            "rejected": rejected_total,
            "seconds": round(seconds, 2),
        }
        import_notifications.insert(0, notification)
        if len(import_notifications) > 20:
            del import_notifications[20:]
        st.toast(
            f"設問インポート完了: 追加 {inserted} 件 / 更新 {updated} 件 / リジェクト {rejected_total} 件",
            icon="✅",
        )
        if file_summaries:
            summary_rows = [
                {
                    "区分": "アップロード",
                    "ファイル": summary["file"],
                    "設問行数": summary["questions"],
                    "正答行数": summary["answers"],
                    "追加": "",
                    "更新": "",
                    "リジェクト": "",
                }
                for summary in file_summaries
            ]
        else:
            summary_rows = []
        summary_rows.append(
            {
                "区分": "DB反映",
                "ファイル": "結果",
                "設問行数": "",
                "正答行数": "",
                "追加": inserted,
                "更新": updated,
                "リジェクト": rejected_total,
            }
        )
        st.table(pd.DataFrame(summary_rows))

    render_history_export_controls(
        db,
        heading="### (5) 履歴エクスポート",
    )

    st.markdown("### (6) データ消去")
    with st.form("data_reset_form"):
        reset_attempts = st.checkbox("学習履歴 (attempts) を削除")
        reset_exams = st.checkbox("模試結果 (exams) を削除")
        reset_all = st.checkbox("全データを初期化 (設問含む)")
        confirmed = st.form_submit_button("削除を実行")
    if confirmed:
        with db.engine.begin() as conn:
            if reset_all:
                for table in [attempts_table, exams_table, srs_table, questions_table]:
                    conn.execute(table.delete())
            else:
                if reset_attempts:
                    conn.execute(attempts_table.delete())
                if reset_exams:
                    conn.execute(exams_table.delete())
        rebuild_tfidf_cache()
        st.success("選択したデータを削除しました。")

    st.markdown("### (7) テンプレートダウンロード")
    with open(DATA_DIR / "questions_sample.csv", "rb") as f:
        st.download_button("設問テンプレCSV", f, file_name="questions_template.csv")
    with open(DATA_DIR / "answers_sample.csv", "rb") as f:
        st.download_button("解答テンプレCSV", f, file_name="answers_template.csv")


def render_settings(db: DBManager) -> None:
    st.title("設定")
    tabs = st.tabs(["表示・操作設定", "データ入出力"])
    with tabs[0]:
        settings = st.session_state["settings"]
        st.info("学習体験を自分好みにカスタマイズできます。各項目の説明を参考に調整してください。")
        theme_options = ["ライト", "ダーク", "セピア"]
        current_theme = settings.get("theme", "セピア")
        theme_index = theme_options.index(current_theme) if current_theme in theme_options else 0
        settings["theme"] = st.selectbox(
            "テーマ",
            theme_options,
            index=theme_index,
            help="画面の配色を切り替えます。暗い環境ではダークテーマ、長文読解にはセピアテーマがおすすめです。",
        )
        size_options = list(FONT_SIZE_SCALE.keys())
        default_size = settings.get("font_size", "標準")
        size_index = (
            size_options.index(default_size)
            if default_size in size_options
            else size_options.index("標準")
        )
        settings["font_size"] = st.selectbox(
            "フォントサイズ",
            size_options,
            index=size_index,
            help="文字サイズを調整して読みやすさを最適化します。『大きい』は夜間学習や高解像度モニタ向きです。",
        )
        settings["shuffle_choices"] = st.checkbox(
            "選択肢をシャッフル",
            value=settings.get("shuffle_choices", True),
            help="毎回選択肢の順番をランダムに入れ替えて、位置記憶に頼らない訓練を行います。",
        )
        settings["timer"] = st.checkbox(
            "タイマーを表示",
            value=settings.get("timer", True),
            help="回答画面に経過時間を表示して本番同様の時間感覚を養います。",
        )
        sm2_key = "settings_sm2_initial_ease"
        current_sm2 = settings.get("sm2_initial_ease", 2.5)
        if st.session_state.get(sm2_key) != current_sm2:
            st.session_state[sm2_key] = current_sm2
        settings["sm2_initial_ease"] = st.slider(
            "SM-2初期ease",
            min_value=1.3,
            max_value=3.0,
            value=st.session_state[sm2_key],
            help="間隔反復アルゴリズムの初期難易度です。既定値2.5で迷ったらそのままにしましょう。",
            key=sm2_key,
        )
        settings["auto_advance"] = st.checkbox(
            "採点後に自動で次問へ進む (0.8秒遅延)",
            value=settings.get("auto_advance", False),
            help="正誤判定後に待機せず次の問題へ進みたい場合に有効化します。",
        )
        low_conf_key = "settings_review_low_confidence_threshold"
        current_low_conf = int(settings.get("review_low_confidence_threshold", 60))
        if st.session_state.get(low_conf_key) != current_low_conf:
            st.session_state[low_conf_key] = current_low_conf
        settings["review_low_confidence_threshold"] = st.slider(
            "低確信として扱う確信度 (%)",
            min_value=0,
            max_value=100,
            value=st.session_state[low_conf_key],
            help="自己評価の確信度がこの値未満なら復習対象に含めます。",
            key=low_conf_key,
        )
        elapsed_key = "settings_review_elapsed_days"
        current_elapsed = int(settings.get("review_elapsed_days", 7))
        if st.session_state.get(elapsed_key) != current_elapsed:
            st.session_state[elapsed_key] = current_elapsed
        settings["review_elapsed_days"] = st.slider(
            "復習抽出の経過日数しきい値",
            min_value=1,
            max_value=30,
            value=st.session_state[elapsed_key],
            help="最終挑戦からこの日数が経過した問題を復習候補に追加します。",
            key=elapsed_key,
        )
        integrations = settings.setdefault("integrations", {})
        st.markdown("#### 外部サービス連携設定")
        st.caption("Google Calendar や Notion 連携に必要なOAuth情報を入力してください。値はブラウザセッション内で保持されます。")
        with st.expander("Google Calendar 連携"):
            google_config = integrations.setdefault(
                "google_calendar",
                {
                    "client_id": "",
                    "client_secret": "",
                    "redirect_uri": "",
                    "access_token": "",
                    "refresh_token": "",
                    "calendar_id": "primary",
                },
            )
            google_config["client_id"] = st.text_input("Client ID", value=google_config.get("client_id", ""))
            google_config["client_secret"] = st.text_input(
                "Client Secret",
                value=google_config.get("client_secret", ""),
                type="password",
            )
            google_config["redirect_uri"] = st.text_input(
                "Redirect URI",
                value=google_config.get("redirect_uri", ""),
                help="OAuth同意画面で設定したリダイレクトURLを入力してください。",
            )
            google_config["access_token"] = st.text_input(
                "Access Token",
                value=google_config.get("access_token", ""),
                type="password",
                help="有効なアクセストークンを入力すると即時同期できます。",
            )
            google_config["refresh_token"] = st.text_input(
                "Refresh Token",
                value=google_config.get("refresh_token", ""),
                type="password",
            )
            google_config["calendar_id"] = st.text_input(
                "対象カレンダーID",
                value=google_config.get("calendar_id", "primary"),
                help="primary のままにするとメインカレンダーへ書き込みます。",
            )
        with st.expander("Notion 連携"):
            notion_config = integrations.setdefault(
                "notion",
                {
                    "integration_token": "",
                    "database_id": "",
                    "notion_version": "2022-06-28",
                },
            )
            notion_config["integration_token"] = st.text_input(
                "Integration Token",
                value=notion_config.get("integration_token", ""),
                type="password",
            )
            notion_config["database_id"] = st.text_input(
                "データベースID",
                value=notion_config.get("database_id", ""),
            )
            notion_config["notion_version"] = st.text_input(
                "Notion-Version",
                value=notion_config.get("notion_version", "2022-06-28"),
                help="Notion APIのバージョン文字列を必要に応じて変更してください。",
            )
        if st.button(
            "TF-IDFを再学習",
            help="検索精度が気になるときに再計算します。データ更新後の再実行がおすすめです。",
        ):
            rebuild_tfidf_cache()
            st.success("TF-IDFを再学習しました")
    with tabs[1]:
        render_data_io(db, parent_nav="設定")


if __name__ == "__main__":
    main()
