from __future__ import annotations

from dataclasses import replace

import pytest

from agent_tui.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(workspace=tmp_path, api_key="test-credential", max_steps=4)


@pytest.fixture
def readonly_settings(settings):
    return replace(settings, read_only=True)
