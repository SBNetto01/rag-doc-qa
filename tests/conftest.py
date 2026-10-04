"""Shared fixtures: a dependency-free corpus and a per-test index directory."""

from __future__ import annotations

from pathlib import Path

import pytest

CORPUS = [
    (
        "handbook.md",
        (
            "# Employee Handbook\n\n"
            "## Payroll\n"
            "Salaries are paid on the 5th business day of every month.\n"
            "Overtime is compensated at 1.5x the hourly rate.\n\n"
            "## Benefits\n"
            "All employees receive health insurance and 30 days of paid leave per year.\n"
            "Meal vouchers are provided every working day.\n"
        ),
    ),
    (
        "onboarding.txt",
        (
            "Your first week at the company: onboarding sessions cover security,\n"
            "the code repository guidelines and who to contact in HR.\n"
            "Bring your laptop and badge on day one.\n"
        ),
    ),
]


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    return tmp_path / "index"


@pytest.fixture
def corpus() -> list[tuple[str, str]]:
    return [(name, text) for name, text in CORPUS]