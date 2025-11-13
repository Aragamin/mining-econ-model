from __future__ import annotations

from typing import List, Optional


def _parse_float_tokens(raw: str, context: str) -> List[float]:
    tokens = [token.strip() for token in raw.split(",")]
    values = [token for token in tokens if token]
    if not values:
        raise ValueError(f"{context} must contain at least one numeric value.")
    try:
        return [float(token) for token in values]
    except ValueError as exc:  # pragma: no cover - defensive parsing guard
        raise ValueError(f"Invalid numeric value in {context}: {raw}") from exc


def parse_optional_float_list(raw: str | None, argument: str) -> Optional[List[float]]:
    """Parse a comma-separated float list, returning None when empty."""

    if raw is None or not raw.strip():
        return None
    return _parse_float_tokens(raw, f"--{argument}")


def parse_required_float_list(raw: str, label: str) -> List[float]:
    """
    Parse a comma-separated float list, raising ValueError when empty.

    Args:
        raw: User-provided string (e.g., from Streamlit text_input).
        label: Human-readable label for error reporting.
    """

    if raw is None or not raw.strip():
        raise ValueError(f"{label} cannot be empty.")
    return _parse_float_tokens(raw, label)


def parse_optional_str_list(raw: str | None) -> Optional[List[str]]:
    """Parse a comma-separated string list, returning None when empty."""

    if raw is None:
        return None
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values or None


def parse_required_str_list(raw: str, label: str) -> List[str]:
    """Parse a comma-separated string list, raising ValueError when empty."""

    values = parse_optional_str_list(raw)
    if not values:
        raise ValueError(f"{label} must contain at least one value.")
    return values
