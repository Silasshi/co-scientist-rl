"""Phase 2A-bis-6: Re-extract TTT-Discover source paper from LaTeX (clean).

Replaces `data/source_paper/v1.md` (PDF-extracted, full of artifacts) with
`data/source_paper/v2.md` (LaTeX-extracted, clean prose + equations).

Strategy:
- Strip preamble before \\begin{document}
- Strip macros.tex (handled by ignoring \\input{macros})
- Section/subsection conversion: \\section{X} -> ## X, \\subsection{X} -> ### X, etc.
- Equation environments preserved verbatim ($...$, \\begin{equation}...\\end{equation})
- Strip \\cite{*} entirely (or replace with [ref])
- Strip figure/table environments (we keep captions if reasonable)
- Strip latex commands like \\textit{X} -> _X_, \\textbf{X} -> **X**
- Preserve abstract content (between \\begin{abstract} and \\end{abstract})
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import chz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    latex_path: str = "/mnt/d/AI/Co-scientist/reference/arXiv-2601.16175v2/main.tex"
    out_path: str = "projects/d5_abstract_retrieve_refine/data/source_paper/v2.md"
    header_title: str = "TTT-Discover (Yuksekgonul et al., arxiv 2601.16175)"


def latex_to_markdown(text: str) -> str:
    """Convert LaTeX source to clean markdown."""
    # Find document body
    m = re.search(r"\\begin\{document\}", text)
    if m:
        text = text[m.end():]
    m = re.search(r"\\end\{document\}", text)
    if m:
        text = text[:m.start()]

    # Strip comments (% to end of line, unless escaped)
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)

    # Strip \input{macros} and similar
    text = re.sub(r"\\input\{[^}]*\}", "", text)
    text = re.sub(r"\\include\{[^}]*\}", "", text)

    # Strip \cite, \citep, \citet, \citealp etc. (keep nothing)
    text = re.sub(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\s*\{[^}]*\}", "", text)

    # Strip \ref, \label
    text = re.sub(r"\\ref\{[^}]*\}", "[?]", text)
    text = re.sub(r"\\eqref\{[^}]*\}", "[?]", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)

    # Sections — convert to markdown headings
    text = re.sub(r"\\section\*?\{([^}]+)\}", r"\n\n## \1\n\n", text)
    text = re.sub(r"\\subsection\*?\{([^}]+)\}", r"\n\n### \1\n\n", text)
    text = re.sub(r"\\subsubsection\*?\{([^}]+)\}", r"\n\n#### \1\n\n", text)
    text = re.sub(r"\\paragraph\{([^}]+)\}", r"\n\n**\1**\n\n", text)

    # Strip figure / table environments (keep captions only)
    def replace_env(env_name: str, replacement_fn):
        nonlocal text
        pattern = rf"\\begin\{{{env_name}\*?\}}(.*?)\\end\{{{env_name}\*?\}}"
        text_local = re.sub(pattern, replacement_fn, text, flags=re.DOTALL)
        return text_local

    # Figures — extract caption only
    def fig_replacer(m):
        body = m.group(1)
        cap_m = re.search(r"\\caption\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", body)
        if cap_m:
            return f"\n\n*[Figure: {cap_m.group(1).strip()}]*\n\n"
        return "\n\n*[Figure]*\n\n"
    text = re.sub(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", fig_replacer, text, flags=re.DOTALL)

    # Tables — keep caption
    def tab_replacer(m):
        body = m.group(1)
        cap_m = re.search(r"\\caption\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", body)
        if cap_m:
            return f"\n\n*[Table: {cap_m.group(1).strip()}]*\n\n"
        return "\n\n*[Table]*\n\n"
    text = re.sub(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tab_replacer, text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{tabular\*?\}(.*?)\\end\{tabular\*?\}", "[tabular content]", text, flags=re.DOTALL)

    # Strip wrapfigure
    text = re.sub(r"\\begin\{wrapfigure\*?\}(.*?)\\end\{wrapfigure\*?\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{minipage\*?\}(.*?)\\end\{minipage\*?\}", lambda m: m.group(1), text, flags=re.DOTALL)

    # Abstract: keep content
    text = re.sub(r"\\begin\{abstract\}", "\n\n## Abstract\n\n", text)
    text = re.sub(r"\\end\{abstract\}", "\n", text)

    # Equation environments: keep verbatim with $$ wrapping
    text = re.sub(
        r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
        lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
        text, flags=re.DOTALL,
    )
    text = re.sub(
        r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
        lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
        text, flags=re.DOTALL,
    )

    # Inline formatting
    text = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\texttt\{([^}]+)\}", r"`\1`", text)
    text = re.sub(r"\\url\{([^}]+)\}", r"<\1>", text)
    text = re.sub(r"\\href\{([^}]+)\}\{([^}]+)\}", r"[\2](\1)", text)

    # Lists: itemize / enumerate
    text = re.sub(r"\\begin\{itemize\}", "\n", text)
    text = re.sub(r"\\end\{itemize\}", "\n", text)
    text = re.sub(r"\\begin\{enumerate\}", "\n", text)
    text = re.sub(r"\\end\{enumerate\}", "\n", text)
    text = re.sub(r"\\item\s*", "\n- ", text)

    # Strip remaining LaTeX commands that survived (best-effort)
    # \cmd{x} → x  (when cmd is unknown)
    # but keep $...$ math
    # Conservative: only strip a small whitelist of common formatting
    for cmd in ["small", "large", "Large", "huge", "tiny", "footnotesize", "normalsize"]:
        text = re.sub(rf"\\{cmd}\b\s*", "", text)
    # Strip \noindent, \vspace, \hspace, \newline, \\, \xspace
    text = re.sub(r"\\noindent\b", "", text)
    text = re.sub(r"\\vspace\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\hspace\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\newline\b", "\n", text)
    text = re.sub(r"\\par\b", "\n\n", text)
    text = re.sub(r"\\xspace\b\s*", " ", text)
    # \Cref / \cref → [?] like \ref
    text = re.sub(r"\\[Cc]ref\{[^}]*\}", "[?]", text)
    # \footnote{X} → drop (often clutter)
    text = re.sub(r"\\footnote\{(?:[^{}]|\{[^{}]*\})*\}", "", text)

    # Whitespace cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def main(config: Config) -> None:
    latex_tex = Path(config.latex_path)
    if not latex_tex.is_absolute():
        latex_tex = REPO_ROOT / latex_tex
    out_path = Path(config.out_path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    if not latex_tex.exists():
        logger.error("LaTeX source not found: %s", latex_tex)
        sys.exit(1)
    raw = latex_tex.read_text()
    logger.info("Read main.tex: %d chars", len(raw))

    md_text = latex_to_markdown(raw)
    logger.info("Converted to markdown: %d chars", len(md_text))

    header = (
        f"# {config.header_title}\n\n"
        "*Extracted from LaTeX source `main.tex` by `extract_source_paper_v2.py` for D5 oracle "
        "build + plan critic privileged info.*\n\n"
        "---\n\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + md_text + "\n")
    logger.info("Wrote %s (%d chars)", out_path, len(header) + len(md_text))

    # Quick sanity sample (3 random non-empty paragraphs from middle 80%)
    paragraphs = [p.strip() for p in md_text.split("\n\n") if p.strip()]
    n = len(paragraphs)
    if n > 10:
        sample_indices = [n // 4, n // 2, 3 * n // 4]
        print("\n=== Spot-check 3 paragraphs ===\n")
        for i in sample_indices:
            print(f"-- [{i}/{n}] --")
            print(paragraphs[i][:400])
            print()


if __name__ == "__main__":
    main(chz.entrypoint(Config))
