#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source tools/use_api_profile.sh <profile>|--list"
  exit 1
fi

_co_scientist_profile_key_prefix="CO_SCIENTIST_API_KEY_"
_co_scientist_profile_base_url_prefix="CO_SCIENTIST_BASE_URL_"

_co_scientist_normalize_profile() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | sed -E 's/[^A-Z0-9]+/_/g; s/^_+//; s/_+$//'
}

_co_scientist_list_profiles() {
  local name
  while IFS='=' read -r name _; do
    [[ "$name" == ${_co_scientist_profile_key_prefix}* ]] || continue
    printf '%s\n' "${name#${_co_scientist_profile_key_prefix}}"
  done < <(env | sort)
}

if [[ $# -ne 1 ]]; then
  echo "Usage: source tools/use_api_profile.sh <profile>|--list"
  return 1
fi

if [[ "$1" == "--list" ]]; then
  _co_scientist_list_profiles
  return 0
fi

_co_scientist_profile_name="$1"
_co_scientist_profile_norm="$(_co_scientist_normalize_profile "$_co_scientist_profile_name")"

if [[ -z "$_co_scientist_profile_norm" ]]; then
  echo "Profile names must contain at least one alphanumeric character."
  return 1
fi

_co_scientist_key_var="${_co_scientist_profile_key_prefix}${_co_scientist_profile_norm}"
_co_scientist_base_url_var="${_co_scientist_profile_base_url_prefix}${_co_scientist_profile_norm}"

if [[ -z "${!_co_scientist_key_var:-}" ]]; then
  echo "Missing ${_co_scientist_key_var}. Export it before activating this profile."
  return 1
fi

export CO_SCIENTIST_ACTIVE_API_PROFILE="$_co_scientist_profile_name"
export TINKER_API_KEY="${!_co_scientist_key_var}"

if [[ -n "${!_co_scientist_base_url_var:-}" ]]; then
  export TINKER_BASE_URL="${!_co_scientist_base_url_var}"
  echo "Activated API profile '$_co_scientist_profile_name' with ${_co_scientist_key_var} and ${_co_scientist_base_url_var}."
else
  unset TINKER_BASE_URL
  echo "Activated API profile '$_co_scientist_profile_name' with ${_co_scientist_key_var}."
fi
