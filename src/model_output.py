"""Small helpers shared by model-backed pipeline stages."""

import json
from typing import Any, Callable, Dict, List, Optional

from .llm import parse_json_output


def generate_json(
    result: Dict[str, Any],
    generate: Callable[..., str],
    *args: Any,
    allow_array: bool = False,
) -> Optional[Any]:
    """Run one model call, save its raw response, and parse JSON."""
    errors = result["errors"]
    try:
        raw_output = generate(*args)
        result["raw_output"] = raw_output
    except Exception as exc:
        errors.append(f"Model generation failed: {type(exc).__name__}: {exc}")
        return None

    try:
        return parse_json_output(raw_output, allow_top_level_array=allow_array)
    except (TypeError, ValueError) as exc:
        errors.append(
            f"Model output could not be parsed: {type(exc).__name__}: {exc}"
        )
        return None


def string_list(value: Any, field: str, errors: List[str]) -> List[str]:
    """Return unique, non-empty strings from a model field."""
    if not isinstance(value, list):
        errors.append(f"Model output field '{field}' must be a list.")
        return []

    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(
                f"Model output field '{field}' at index {index} must be a "
                "non-empty string."
            )
            continue
        item = item.strip()
        if item not in result:
            result.append(item)
    return result


def json_text(value: Any) -> str:
    """Serialize structured prompt input consistently."""
    return json.dumps(value, ensure_ascii=False, indent=2)
