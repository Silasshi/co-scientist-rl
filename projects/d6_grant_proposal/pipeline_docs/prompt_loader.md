# prompt_loader

**Source:** `src/co_scientist/d6_grant_proposal/prompt_loader.py` (~81 lines)
**Stage:** Infrastructure (cross-cutting)

## Purpose

Single source of truth for D6 prompt templates. Loads `.md` files from `projects/d6_grant_proposal/prompts/`, strips the optional doc header (everything before the first `---` separator line), caches the result via `lru_cache`, and exposes `str.format`-style rendering. D6 breaks D5/D4's "hardcoded prompt strings in `.py`" convention deliberately — prompts are paper artifacts and benefit from being first-class diff-able files.

## Key callables

| Callable | Purpose |
|---|---|
| `load_prompt(name: str) -> str` | Return raw template body for `<name>.md` (cached) |
| `render_prompt(name: str, **kwargs) -> str` | `load_prompt(name).format(**kwargs)` |
| `reload_prompts() -> None` | Clear `lru_cache` (use after editing a `.md` mid-session) |
| `_strip_doc_header(raw: str) -> str` | Internal: drop everything before first `^---$` line + trailing newlines |

## Template file format

```markdown
<!-- optional doc header: source, role, vars -->

# Title (not loaded)

**Role**: ...
**Template variables**: `{var1}`, `{var2}`

---

<actual template body — what load_prompt returns>
```

The first standalone `---` line is the separator. Trailing newlines on the body are stripped so `.format(**kwargs)` produces bytes byte-equal to the pre-refactor hardcoded strings.

## Naming

`name` is a relative path under `prompts/`, with or without `.md`. Examples: `generation/student`, `review/critic_instructions`, `audit/absolute_score`.

## Dependencies

Standard library only (`pathlib`, `functools.lru_cache`). No external deps.

## Capability

Pure infrastructure — does not train, evaluate, or score anything. Used by every other D6 module that emits prompts.

## Tests

`tests/test_d6_prompt_loader.py` (24 tests): loader mechanics, doc-header stripping, `lru_cache` behaviour, plus structural invariants on every active template (8 reference headings in generation, 8 critique children in review, 12 active signals in audit + pairwise, no top-level `.md` files outside role subfolders).
