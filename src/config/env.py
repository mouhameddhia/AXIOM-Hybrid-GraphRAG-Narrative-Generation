"""
Typed, validated readers for environment-backed configuration.

Every configuration value in this project enters the process through one of
these helpers. They exist so that configuration failures surface *at startup*
with an actionable message naming the offending variable, rather than as a
``TypeError`` deep inside a retriever hours later.

Design rules enforced here:
  * an unset variable and an empty/whitespace-only variable are treated
    identically (both mean "use the default");
  * every numeric reader supports inclusive bounds, so an out-of-range value
    is rejected rather than silently producing a degenerate pipeline
    (e.g. ``top_k = 0``);
  * no reader ever returns ``None`` unless the supplied default is ``None``.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypeVar

__all__ = [
    "ConfigError",
    "EnvSource",
    "env_bool",
    "env_float",
    "env_frozenset",
    "env_int",
    "env_path",
    "env_str",
    "env_str_float_map",
    "env_tuple",
]

T = TypeVar("T")

#: Read-only view over the process environment. Injecting a plain ``dict``
#: instead lets tests configure the system without mutating ``os.environ``.
EnvSource = Mapping[str, str]

_TRUE_TOKENS: Final[frozenset[str]] = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_TOKENS: Final[frozenset[str]] = frozenset({"0", "false", "no", "n", "off"})


class ConfigError(ValueError):
    """Raised when an environment variable holds a value the system cannot use."""


def _raw(env: EnvSource, key: str) -> str | None:
    """Return the trimmed value of ``key``, or ``None`` if unset or blank."""
    value = env.get(key)
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def env_str(env: EnvSource, key: str, default: str) -> str:
    """Read a string variable, falling back to ``default`` when unset/blank."""
    return _raw(env, key) or default


def env_int(
    env: EnvSource,
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read an integer variable and validate it against optional inclusive bounds."""
    raw = _raw(env, key)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    _check_bounds(key, value, minimum, maximum)
    return value


def env_float(
    env: EnvSource,
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read a float variable and validate it against optional inclusive bounds."""
    raw = _raw(env, key)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{key} must be a number, got {raw!r}") from exc
    _check_bounds(key, value, minimum, maximum)
    return value


def env_bool(env: EnvSource, key: str, default: bool) -> bool:
    """Read a boolean variable accepting the usual 1/true/yes/on spellings."""
    raw = _raw(env, key)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_TOKENS:
        return True
    if lowered in _FALSE_TOKENS:
        return False
    raise ConfigError(
        f"{key} must be a boolean "
        f"({'/'.join(sorted(_TRUE_TOKENS | _FALSE_TOKENS))}), got {raw!r}"
    )


def env_path(env: EnvSource, key: str, default: Path | str) -> Path:
    """Read a filesystem path variable. Existence is *not* checked here."""
    raw = _raw(env, key)
    return Path(raw).expanduser() if raw else Path(default).expanduser()


def env_tuple(env: EnvSource, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated list, dropping blank entries and preserving order."""
    raw = _raw(env, key)
    if raw is None:
        return default
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def env_frozenset(env: EnvSource, key: str, default: frozenset[str]) -> frozenset[str]:
    """Read a comma-separated list as an unordered, deduplicated set."""
    raw = _raw(env, key)
    if raw is None:
        return default
    return frozenset(token.strip() for token in raw.split(",") if token.strip())


def env_str_float_map(
    env: EnvSource,
    key: str,
    default: Mapping[str, float],
    *,
    minimum: float | None = None,
) -> dict[str, float]:
    """
    Read a JSON object of ``{"name": number}`` pairs.

    Used for per-stream fusion weights, where the *set* of streams is not known
    to this module -- any stream name the pipeline registers can be weighted
    without a code change.
    """
    raw = _raw(env, key)
    if raw is None:
        return dict(default)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{key} must be a JSON object, got {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{key} must be a JSON object, got {type(parsed).__name__}")

    weights: dict[str, float] = {}
    for name, value in parsed.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"{key}['{name}'] must be a number, got {value!r}")
        numeric = float(value)
        _check_bounds(f"{key}['{name}']", numeric, minimum, None)
        weights[str(name)] = numeric
    return weights


def _check_bounds(
    key: str, value: float, minimum: float | None, maximum: float | None
) -> None:
    """Raise ``ConfigError`` if ``value`` falls outside the inclusive bounds."""
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} must be <= {maximum}, got {value}")


def process_env() -> EnvSource:
    """Return the live process environment as the default configuration source."""
    return os.environ
