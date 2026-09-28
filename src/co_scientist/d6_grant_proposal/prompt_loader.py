"""D6 prompt loader — single source of truth for D6 prompts on disk.

D6 externalizes every prompt template to `projects/d6_grant_proposal/prompts/`.
Edit a `.md` file and the next training run uses the new text. This loader
reads the file, strips an optional doc header, and (optionally) substitutes
template variables via `str.format`.

Why D6-only and not shared/: D5 and D4 keep prompts hardcoded in `.py`. This
break with convention is intentional for D6 (paper-artifact diffability).
Promote to `shared/` if a second direction adopts the pattern.

Doc header convention. Each `.md` template file looks like:

    <!-- optional source/role/vars metadata -->

    # Title

    **Role**: ...
    **Template variables**: `{var1}`, `{var2}`

    ---

    <body — what load_prompt returns>

`_strip_doc_header` returns everything after the first `---` line that
appears alone on its own line. If no such marker exists, the raw text is
returned unchanged. Trailing newlines are stripped so `.format(**kwargs)`
produces the same bytes as the previous hardcoded Python strings.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_ROOT = Path(__file__).resolve().parents[3] / "projects/d6_grant_proposal/prompts"


def _strip_doc_header(raw: str) -> str:
    """Return the template body — everything after the first `---` rule line.

    A `---` rule is a line whose stripped content is exactly three dashes.
    Blank lines immediately following the rule are skipped. The result is
    `.rstrip("\\n")`-ed so callers don't pick up file trailing newlines.
    """
    lines = raw.splitlines(keepends=True)
    body_start: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == "---":
            body_start = i + 1
            break
    if body_start is None:
        return raw.rstrip("\n")
    while body_start < len(lines) and lines[body_start].strip() == "":
        body_start += 1
    return "".join(lines[body_start:]).rstrip("\n")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load a prompt template by relative path under `prompts/` (e.g. 'generation/student.md').

    Cached for the process lifetime. Call `reload_prompts()` after editing a
    template in a long-lived process (notebook, dev loop).
    """
    path = PROMPTS_ROOT / name
    raw = path.read_text(encoding="utf-8")
    return _strip_doc_header(raw)


def render_prompt(name: str, **kwargs: object) -> str:
    """Load + `str.format(**kwargs)`. Use when the template has `{var}` placeholders."""
    return load_prompt(name).format(**kwargs)


def reload_prompts() -> None:
    """Clear the load cache. Use after editing a `.md` mid-session."""
    load_prompt.cache_clear()


__all__ = ["PROMPTS_ROOT", "load_prompt", "render_prompt", "reload_prompts"]
