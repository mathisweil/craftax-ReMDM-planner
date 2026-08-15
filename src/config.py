"""Configuration loader for ReMDM-Craftax.

Loads YAML configs with deep-merge and CLI override support. Mirrors
``src/config.py`` in the sibling minihack repo: same helpers, same
semantics, same error types (`KeyError` for an unknown key, `TypeError`
for a value that cannot be read as the key's type).
"""

from __future__ import annotations

import contextlib

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*).

    Args:
        base: Base dictionary to merge into.
        override: Dictionary whose values take precedence.

    Returns:
        The merged dictionary (same object as *base*).
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def validate_keys(
    keys, allowed: set[str], source: str, valid_source: str = "configs/defaults.yaml"
) -> None:
    """Reject unknown config keys instead of silently ignoring them.

    Args:
        keys: Keys to check.
        allowed: The full set of valid config keys.
        source: Label for the error message (file path or '--override').
        valid_source: Where the caller's valid keys are defined.

    Raises:
        KeyError: If any key is not a known config key.
    """
    unknown = sorted(set(keys) - allowed)
    if unknown:
        raise KeyError(
            f"Unknown config key(s) {unknown} in {source}. "
            f"Valid keys are defined in {valid_source}."
        )


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    """Split ``KEY=VALUE`` CLI strings into a dict.

    Args:
        pairs: Raw ``--override`` arguments.

    Returns:
        Mapping of key to raw (uncast) string value.

    Raises:
        ValueError: If an argument is not of the form ``KEY=VALUE``.
    """
    overrides: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def cast_override(key: str, raw: str, current) -> object:
    """Cast a CLI override string to the type of the current config value.

    Args:
        key: Config key being overridden.
        raw: Raw string from the command line.
        current: Current (default or config-file) value, used for typing.

    Returns:
        Parsed Python value.

    Raises:
        TypeError: If the value cannot be interpreted as the key's type.
    """
    if isinstance(current, str):
        return raw

    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw

    if current is None or value is None:
        return value

    # YAML 1.1 reads '1e-4' as a string; accept scientific notation for
    # numeric keys.
    if (
        isinstance(current, (int, float))
        and not isinstance(current, bool)
        and isinstance(value, str)
    ):
        with contextlib.suppress(ValueError):
            value = float(value)

    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(f"'{key}' expects a boolean, got '{raw}'")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"'{key}' expects an integer, got '{raw}'")
        if isinstance(value, float):
            if not value.is_integer():
                raise TypeError(f"'{key}' expects an integer, got '{raw}'")
            value = int(value)
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"'{key}' expects a number, got '{raw}'")
        return float(value)
    if isinstance(current, list):
        if not isinstance(value, list):
            raise TypeError(f"'{key}' expects a list, got '{raw}'")
        return value
    return value
