import datetime as dt

import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import sm2_update


def test_sm2_initial_interval_sets_one_day():
    payload = sm2_update(None, grade=4, initial_ease=2.5)
    assert payload["interval"] == 1
    assert payload["repetition"] == 1
    assert payload["due_date"] == dt.date.today() + dt.timedelta(days=1)


def test_sm2_second_interval_sets_six_days():
    row = pd.Series({"repetition": 1, "interval": 1, "ease": 2.5})
    payload = sm2_update(row, grade=5)
    assert payload["interval"] == 6
    assert payload["repetition"] == 2
    assert payload["due_date"] == dt.date.today() + dt.timedelta(days=6)


def test_sm2_subsequent_intervals_scale_with_ease():
    row = pd.Series({"repetition": 2, "interval": 6, "ease": 2.6})
    payload = sm2_update(row, grade=4)
    expected_interval = int(round(6 * 2.6))
    assert payload["interval"] == expected_interval
    assert payload["due_date"] == dt.date.today() + dt.timedelta(days=expected_interval)


def test_sm2_reset_on_low_grade():
    row = pd.Series({"repetition": 3, "interval": 15, "ease": 2.8})
    payload = sm2_update(row, grade=2)
    assert payload["repetition"] == 0
    assert payload["interval"] == 1
    assert payload["due_date"] == dt.date.today() + dt.timedelta(days=1)
