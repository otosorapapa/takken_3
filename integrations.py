"""External service integration clients for Takken Drill."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class IntegrationError(Exception):
    """Generic exception for integration failures."""


class IntegrationConfigError(IntegrationError):
    """Raised when integration is triggered without proper configuration."""


@dataclass
class OAuthCredentials:
    """OAuth credentials shared by Google Calendar and Notion integrations."""

    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    access_token: str = ""
    refresh_token: str = ""

    def is_configured(self) -> bool:
        return bool(self.access_token)


@dataclass
class GoogleCalendarConfig:
    """Configuration settings for Google Calendar sync."""

    credentials: OAuthCredentials
    calendar_id: str = "primary"


class GoogleCalendarClient:
    """Simple Google Calendar API client for creating study events."""

    EVENTS_ENDPOINT = "https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"

    def __init__(self, config: GoogleCalendarConfig) -> None:
        self.config = config

    def build_events(self, due_df: pd.DataFrame, limit: int = 20) -> List[Dict[str, object]]:
        if due_df.empty:
            raise IntegrationError("同期対象の学習スケジュールが見つかりませんでした。")
        working = due_df.copy()
        if "due_date" in working.columns:
            working["due_date"] = pd.to_datetime(working["due_date"]).dt.date
        else:
            working["due_date"] = dt.date.today()
        working = working.sort_values("due_date").head(limit)
        events: List[Dict[str, object]] = []
        for row in working.itertuples():
            due_date: dt.date = getattr(row, "due_date") or dt.date.today()
            question = getattr(row, "question", "復習セッション")
            category = getattr(row, "category", "宅建ドリル")
            question_id = getattr(row, "question_id", "")
            summary = f"宅建復習: {category}"
            description = question
            if question_id:
                description = f"問題ID: {question_id}\n\n{question}"
            event = {
                "summary": summary[:250],
                "description": description,
                "start": {"date": due_date.isoformat()},
                "end": {"date": (due_date + dt.timedelta(days=1)).isoformat()},
            }
            events.append(event)
        return events

    def sync_study_schedule(self, due_df: pd.DataFrame, limit: int = 20) -> Dict[str, object]:
        if not self.config.credentials.is_configured():
            raise IntegrationConfigError("Google Calendar のアクセストークンが設定されていません。")
        events = self.build_events(due_df, limit=limit)
        endpoint = self.EVENTS_ENDPOINT.format(calendar_id=self.config.calendar_id or "primary")
        headers = {
            "Authorization": f"Bearer {self.config.credentials.access_token}",
            "Content-Type": "application/json",
        }
        created = 0
        failures: List[str] = []
        for event in events:
            response = requests.post(endpoint, headers=headers, json=event, timeout=10)
            if response.status_code >= 400:
                error_detail = response.text
                failures.append(error_detail)
                logger.warning("Google Calendar API error %s: %s", response.status_code, error_detail)
                continue
            created += 1
        if created == 0 and failures:
            raise IntegrationError("Google Calendar への同期に失敗しました。")
        return {"created": created, "failures": failures, "events": events}


@dataclass
class NotionConfig:
    """Configuration parameters for Notion logging."""

    integration_token: str = ""
    database_id: str = ""
    notion_version: str = "2022-06-28"

    def is_configured(self) -> bool:
        return bool(self.integration_token and self.database_id)


class NotionClient:
    """Minimal Notion API client for pushing learning summaries."""

    PAGES_ENDPOINT = "https://api.notion.com/v1/pages"

    def __init__(self, config: NotionConfig) -> None:
        self.config = config

    def sync_learning_log(self, summaries: Iterable[Dict[str, object]]) -> Dict[str, object]:
        if not self.config.is_configured():
            raise IntegrationConfigError("Notion のデータベースIDまたはトークンが設定されていません。")
        headers = {
            "Authorization": f"Bearer {self.config.integration_token}",
            "Notion-Version": self.config.notion_version,
            "Content-Type": "application/json",
        }
        created = 0
        failures: List[str] = []
        for summary in summaries:
            payload = {
                "parent": {"database_id": self.config.database_id},
                "properties": {
                    "学習日": {
                        "date": {"start": summary["date"].isoformat()},
                    },
                    "挑戦数": {"number": int(summary["attempts"])},
                    "正答率": {"number": float(summary["accuracy"])},
                    "平均解答時間秒": {"number": float(summary["avg_seconds"])},
                },
            }
            response = requests.post(self.PAGES_ENDPOINT, headers=headers, json=payload, timeout=10)
            if response.status_code >= 400:
                error_detail = response.text
                failures.append(error_detail)
                logger.warning("Notion API error %s: %s", response.status_code, error_detail)
                continue
            created += 1
        if created == 0 and failures:
            raise IntegrationError("Notion への同期に失敗しました。")
        return {"created": created, "failures": failures}

