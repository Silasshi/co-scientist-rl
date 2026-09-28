import os
import re
from dataclasses import dataclass
from typing import MutableMapping

import tinker


ACTIVE_PROFILE_ENV_VAR = "CO_SCIENTIST_ACTIVE_API_PROFILE"
PROFILE_API_KEY_ENV_PREFIX = "CO_SCIENTIST_API_KEY_"
PROFILE_BASE_URL_ENV_PREFIX = "CO_SCIENTIST_BASE_URL_"
TINKER_API_KEY_ENV_VAR = "TINKER_API_KEY"
TINKER_BASE_URL_ENV_VAR = "TINKER_BASE_URL"


@dataclass(frozen=True)
class APIProfileSelection:
    active_profile: str | None
    normalized_profile: str | None
    api_key_env_var: str | None
    base_url_env_var: str | None
    resolved_base_url: str | None
    clear_tinker_base_url: bool


def normalize_api_profile(profile: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", profile.strip()).strip("_").upper()
    if not normalized:
        raise ValueError("API profile names must contain at least one alphanumeric character.")
    return normalized


def resolve_api_profile_selection(
    *,
    api_profile: str | None = None,
    base_url: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> APIProfileSelection:
    env = os.environ if environ is None else environ
    requested_profile = api_profile
    if requested_profile is None:
        requested_profile = env.get(ACTIVE_PROFILE_ENV_VAR)

    if requested_profile is None or not requested_profile.strip():
        return APIProfileSelection(
            active_profile=None,
            normalized_profile=None,
            api_key_env_var=None,
            base_url_env_var=None,
            resolved_base_url=base_url,
            clear_tinker_base_url=False,
        )

    normalized_profile = normalize_api_profile(requested_profile)
    api_key_env_var = f"{PROFILE_API_KEY_ENV_PREFIX}{normalized_profile}"
    profile_api_key = env.get(api_key_env_var)
    if not profile_api_key:
        raise ValueError(
            "Missing API key for profile "
            f"{requested_profile!r}. Expected environment variable {api_key_env_var}."
        )

    base_url_env_var = f"{PROFILE_BASE_URL_ENV_PREFIX}{normalized_profile}"
    profile_base_url = env.get(base_url_env_var)
    resolved_base_url = base_url if base_url is not None else profile_base_url

    return APIProfileSelection(
        active_profile=requested_profile,
        normalized_profile=normalized_profile,
        api_key_env_var=api_key_env_var,
        base_url_env_var=base_url_env_var,
        resolved_base_url=resolved_base_url,
        clear_tinker_base_url=resolved_base_url is None,
    )


def apply_api_profile_selection(
    *,
    api_profile: str | None = None,
    base_url: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> APIProfileSelection:
    env = os.environ if environ is None else environ
    selection = resolve_api_profile_selection(
        api_profile=api_profile,
        base_url=base_url,
        environ=env,
    )

    if selection.api_key_env_var is None:
        return selection

    env[TINKER_API_KEY_ENV_VAR] = env[selection.api_key_env_var]
    env[ACTIVE_PROFILE_ENV_VAR] = selection.active_profile or ""

    if selection.clear_tinker_base_url:
        env.pop(TINKER_BASE_URL_ENV_VAR, None)
    else:
        env[TINKER_BASE_URL_ENV_VAR] = selection.resolved_base_url or ""

    return selection


def create_service_client(
    *,
    base_url: str | None = None,
    api_profile: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> tinker.ServiceClient:
    selection = apply_api_profile_selection(
        api_profile=api_profile,
        base_url=base_url,
        environ=environ,
    )
    return tinker.ServiceClient(base_url=selection.resolved_base_url)
