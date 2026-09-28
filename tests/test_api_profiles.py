from pathlib import Path
import sys

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.trainers import api_profiles


def test_normalize_api_profile_rejects_empty_name():
    with pytest.raises(ValueError, match="at least one alphanumeric"):
        api_profiles.normalize_api_profile("___")


def test_resolve_api_profile_uses_explicit_profile_and_profile_base_url():
    env = {
        "CO_SCIENTIST_API_KEY_PRIMARY": "primary-secret",
        "CO_SCIENTIST_BASE_URL_PRIMARY": "https://primary.example/v1",
    }
    selection = api_profiles.resolve_api_profile_selection(
        api_profile="primary",
        base_url=None,
        environ=env,
    )

    assert selection.active_profile == "primary"
    assert selection.normalized_profile == "PRIMARY"
    assert selection.api_key_env_var == "CO_SCIENTIST_API_KEY_PRIMARY"
    assert selection.base_url_env_var == "CO_SCIENTIST_BASE_URL_PRIMARY"
    assert selection.resolved_base_url == "https://primary.example/v1"
    assert not selection.clear_tinker_base_url


def test_apply_api_profile_selection_sets_key_and_clears_stale_base_url():
    env = {
        "CO_SCIENTIST_API_KEY_BACKUP": "backup-secret",
        "TINKER_BASE_URL": "https://stale.example/v1",
    }

    selection = api_profiles.apply_api_profile_selection(
        api_profile="backup",
        base_url=None,
        environ=env,
    )

    assert selection.clear_tinker_base_url
    assert env["CO_SCIENTIST_ACTIVE_API_PROFILE"] == "backup"
    assert env["TINKER_API_KEY"] == "backup-secret"
    assert "TINKER_BASE_URL" not in env


def test_create_service_client_uses_selected_profile(monkeypatch):
    env = {
        "CO_SCIENTIST_API_KEY_ALT_API": "alt-secret",
        "CO_SCIENTIST_BASE_URL_ALT_API": "https://alt.example/v1",
    }
    captured: dict[str, str | None] = {}

    class DummyServiceClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(api_profiles.tinker, "ServiceClient", DummyServiceClient)

    client = api_profiles.create_service_client(
        api_profile="alt api",
        base_url=None,
        environ=env,
    )

    assert isinstance(client, DummyServiceClient)
    assert env["TINKER_API_KEY"] == "alt-secret"
    assert env["TINKER_BASE_URL"] == "https://alt.example/v1"
    assert captured == {"base_url": "https://alt.example/v1"}
