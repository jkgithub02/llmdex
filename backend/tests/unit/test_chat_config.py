"""R9.5 / R9.7 - the two numbers that bound a chat run, and where they come from."""

import pytest

from app.core.config import ChatSettings, chat_settings


def test_defaults_are_usable_without_any_environment(monkeypatch):
    for key in ("LLMDEX_CHAT_REQUEST_LIMIT", "LLMDEX_CHAT_COMPACT_ABOVE_TOKENS"):
        monkeypatch.delenv(key, raising=False)

    settings = chat_settings()

    assert settings.request_limit == 8
    assert settings.compact_above_tokens == 24000
    assert settings.keep_recent == 4


def test_the_environment_overrides_them(monkeypatch):
    monkeypatch.setenv("LLMDEX_CHAT_REQUEST_LIMIT", "3")
    monkeypatch.setenv("LLMDEX_CHAT_COMPACT_ABOVE_TOKENS", "1000")

    settings = chat_settings()

    assert settings.request_limit == 3
    assert settings.compact_above_tokens == 1000


def test_a_request_limit_below_two_is_refused():
    """One request cannot call a tool and then use its result (R9.7)."""
    with pytest.raises(ValueError):
        ChatSettings(request_limit=1)
