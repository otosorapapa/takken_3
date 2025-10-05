"""Lightweight fallback implementation of the ``ics`` interface used in the app."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Iterable, List

__all__ = ["Calendar", "Event"]


def _format_datetime(value: _dt.datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


@dataclass
class Event:
    """Minimal event representation compatible with the ``ics`` package."""

    name: str = ""
    begin: _dt.datetime | None = None
    end: _dt.datetime | None = None
    description: str = ""
    location: str | None = None

    def _to_lines(self) -> List[str]:
        lines = ["BEGIN:VEVENT"]
        if self.name:
            lines.append(f"SUMMARY:{_escape(self.name)}")
        if self.begin:
            lines.append(f"DTSTART:{_format_datetime(self.begin)}")
        if self.end:
            lines.append(f"DTEND:{_format_datetime(self.end)}")
        if self.description:
            lines.append(f"DESCRIPTION:{_escape(self.description)}")
        if self.location:
            lines.append(f"LOCATION:{_escape(self.location)}")
        lines.append("END:VEVENT")
        return lines


@dataclass
class _EventCollection:
    _events: List[Event] = field(default_factory=list)

    def add(self, event: Event) -> None:
        self._events.append(event)

    def __iter__(self) -> Iterable[Event]:
        return iter(self._events)


@dataclass
class Calendar:
    """Simple calendar container that mimics the public API of ``ics.Calendar``."""

    events: _EventCollection = field(default_factory=_EventCollection)
    prodid: str = "-//takken.app//Learning Planner//JP"
    version: str = "2.0"

    def serialize(self) -> str:
        lines = ["BEGIN:VCALENDAR", f"PRODID:{self.prodid}", f"VERSION:{self.version}"]
        for event in self.events:
            lines.extend(event._to_lines())
        lines.append("END:VCALENDAR")
        return "\r\n".join(lines) + "\r\n"

    def __str__(self) -> str:  # pragma: no cover - mirror upstream behaviour
        return self.serialize()
