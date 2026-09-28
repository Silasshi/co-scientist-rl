#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DESIDERATA_NAMES = {
    1: "Handles All Criteria",
    2: "Detailed / Specific",
    3: "No Overlooked Flaws",
    4: "Well Justified",
    5: "Cost / Effort Efficient",
    6: "No Ethical Issues",
    7: "Consistent With Plan",
}

THEME_PATTERNS: dict[str, tuple[str, ...]] = {
    "criteria_coverage": (
        r"\bcriteria\b",
        r"\bfails? to address\b",
        r"\bomits?\b",
        r"\bmissing\b",
        r"\balign(?:ment)?\b",
    ),
    "detail_specificity": (
        r"\bdetail(?:ed)?\b",
        r"\bspecific(?:ity)?\b",
        r"\bconcrete\b",
        r"\bvague\b",
        r"\bimplementation\b",
    ),
    "overlooked_flaws": (
        r"\bweakness(?:es)?\b",
        r"\bflaw(?:s)?\b",
        r"\brisk(?:s)?\b",
        r"\bfailure(?:s)?\b",
        r"\boverlook(?:ed)?\b",
    ),
    "rationale_justification": (
        r"\brationale\b",
        r"\bjustify\b",
        r"\bjustification\b",
        r"\bmotivat(?:e|ion)\b",
        r"\bwhy\b",
    ),
    "efficiency_complexity": (
        r"\befficient(?:cy)?\b",
        r"\bcost\b",
        r"\bcomplex(?:ity)?\b",
        r"\bcompute\b",
        r"\bresource(?:s)?\b",
    ),
    "ethics_safety": (
        r"\bethic(?:al|s)?\b",
        r"\bsafety\b",
        r"\bharm\b",
        r"\bnegative consequence(?:s)?\b",
    ),
    "consistency": (
        r"\bconsistent(?:cy)?\b",
        r"\bcontradict(?:ion|s)?\b",
        r"\bcoheren(?:t|ce)\b",
    ),
    "evaluation_benchmarks": (
        r"\bbenchmark(?:s)?\b",
        r"\bmetric(?:s)?\b",
        r"\bevaluat(?:e|ion)\b",
        r"\bbaseline(?:s)?\b",
        r"\bablation\b",
    ),
    "scalability": (
        r"\bscal(?:e|able|ability)\b",
        r"\blarge[- ]scale\b",
        r"\blarger model(?:s)?\b",
        r"\bmemory\b",
        r"\bdistributed\b",
    ),
    "hyperparameter_robustness": (
        r"\bhyperparameter(?:s)?\b",
        r"\brobust(?:ness)?\b",
        r"\bsensitiv(?:e|ity)\b",
        r"\btuning\b",
    ),
    "interpretability_control": (
        r"\binterpret(?:ability)?\b",
        r"\bcontrol(?:lability)?\b",
        r"\bintuitive\b",
        r"\btargeted intervention(?:s)?\b",
    ),
    "format_compliance": (
        r"\bformat\b",
        r"\btag(?:s)?\b",
        r"\bword count\b",
        r"\bcompliant\b",
    ),
}

COVERED_THEMES = {
    "criteria_coverage",
    "detail_specificity",
    "overlooked_flaws",
    "rationale_justification",
    "efficiency_complexity",
    "ethics_safety",
    "consistency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SDPO grader desiderata-review comments from logs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more run directories or jsonl log files.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "analysis/results/desiderata_comment_audit"),
        help="Directory where markdown/json/csv outputs are written.",
    )
    parser.add_argument(
        "--status",
        choices=("all", "valid"),
        default="valid",
        help="Filter training log rows by status when present.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on processed rows per file. 0 means no cap.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_suggestion_text(text: str) -> str:
    normalized = clean_text(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_tag_text(text: str, tag: str) -> str:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return ""
    return clean_text(match.group(1))


def extract_bullets(text: str) -> list[str]:
    raw = clean_text(text)
    if not raw:
        return []
    parts = re.split(r"(?:^|\s)-\s+", raw)
    bullets = [clean_text(part) for part in parts if clean_text(part)]
    if bullets:
        return bullets
    return [raw]


def parse_item_blocks(xml_text: str) -> list[dict]:
    item_blocks = re.findall(
        r"<item\b[^>]*>.*?</item>",
        xml_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    items: list[dict] = []
    for item_xml in item_blocks:
        item_num_match = re.search(
            r"<item\b[^>]*num\s*=\s*['\"]?(\d+)['\"]?",
            item_xml,
            flags=re.IGNORECASE,
        )
        item_num = int(item_num_match.group(1)) if item_num_match is not None else None
        criteria = extract_tag_text(item_xml, "criteria")
        summary = ""
        suggestion = ""
        review_match = re.search(
            r"<desiderata_review>\s*(.*?)\s*</desiderata_review>",
            item_xml,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if review_match is not None:
            review_body = review_match.group(1)
            summary = extract_tag_text(review_body, "summary")
            suggestion = extract_tag_text(review_body, "suggested_new_desiderata")
            if not suggestion:
                suggestion = extract_tag_text(review_body, "new_desiderata")
        suggestion = clean_text(suggestion)
        if suggestion.lower() in {"none", "n/a", "na", "no"}:
            suggestion = ""
        levels = [
            int(v)
            for v in re.findall(
                r"<level>\s*([0-3])\s*</level>",
                item_xml,
                flags=re.IGNORECASE,
            )
        ]
        if len(levels) < 7:
            levels = levels + [0] * (7 - len(levels))
        items.append(
            {
                "item_num": item_num,
                "criteria": criteria,
                "summary": clean_text(summary),
                "suggestion": suggestion,
                "levels": levels[:7],
            }
        )
    return items


def parse_sample_review(xml_text: str) -> tuple[list[str], list[str]]:
    review_match = re.search(
        r"<sample_review>\s*(.*?)\s*</sample_review>",
        xml_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if review_match is None:
        return [], []
    body = review_match.group(1)
    return (
        extract_bullets(extract_tag_text(body, "weaknesses")),
        extract_bullets(extract_tag_text(body, "potential_fixes")),
    )


def parse_global_review(xml_text: str) -> str:
    rubric_match = re.search(
        r"<rubric\b[^>]*>(.*)</rubric>",
        xml_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    search_scope = rubric_match.group(1) if rubric_match is not None else xml_text
    search_scope = re.sub(
        r"<item\b[^>]*>.*?</item>",
        " ",
        search_scope,
        flags=re.DOTALL | re.IGNORECASE,
    )
    reviews = re.findall(
        r"<desiderata_review>\s*(.*?)\s*</desiderata_review>",
        search_scope,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not reviews:
        return ""
    for review_body in reviews:
        summary = extract_tag_text(review_body, "summary")
        if summary:
            return summary
    return clean_text(reviews[-1])


def resolve_input_paths(raw_inputs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in raw_inputs:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_file():
            resolved.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Input path not found: {path}")

        train_log = path / "train" / "training_logs.jsonl"
        if train_log.exists():
            resolved.append(train_log)
            continue

        eval_logs = sorted((path / "evaluation").glob("eval_logs(*).jsonl"))
        if eval_logs:
            resolved.extend(eval_logs)
            continue

        raise FileNotFoundError(
            f"Could not resolve a log file from directory: {path}"
        )
    return resolved


def detect_themes(text: str) -> list[str]:
    lowered = clean_text(text).lower()
    if not lowered:
        return []
    matched: list[str] = []
    for theme, patterns in THEME_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            matched.append(theme)
    return matched


def verdict_from_signals(
    explicit_suggestion_count: int,
    total_items: int,
    suggestion_counter: Counter[str],
    theme_counter: Counter[str],
) -> str:
    suggestion_rate = (
        explicit_suggestion_count / float(total_items) if total_items else 0.0
    )
    unique_suggestions = len(suggestion_counter)
    max_repeat = suggestion_counter.most_common(1)[0][1] if suggestion_counter else 0
    covered_total = sum(theme_counter[t] for t in COVERED_THEMES)
    extra_themes = [t for t in theme_counter if t not in COVERED_THEMES]
    extra_total = sum(theme_counter[t] for t in extra_themes)

    if suggestion_counter and (
        max_repeat >= 5
        or (
            explicit_suggestion_count >= 10
            and unique_suggestions / float(explicit_suggestion_count) <= 0.7
        )
    ):
        return (
            "There is direct evidence that the grader wants desiderata revisions: "
            "similar new-desiderata suggestions recur often enough to inspect manually."
        )
    if explicit_suggestion_count >= 5 and max_repeat <= 2:
        return (
            "Explicit `suggested_new_desiderata` entries exist, but they are mostly one-off, "
            "rubric-specific suggestions rather than recurring evidence that the global seven "
            "desiderata need revision."
        )
    if extra_total > covered_total and extra_total >= 25:
        return (
            "There is indirect evidence of a coverage gap: comments frequently emphasize "
            "themes not named by the current seven desiderata. Review whether those themes "
            "should remain rubric-specific or become explicit desiderata."
        )
    return (
        "There is no strong evidence from grader comments alone that the seven desiderata "
        "need revision. Most comments appear to criticize plan quality in ways already "
        "captured by the existing desiderata."
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()
    log_paths = resolve_input_paths(args.inputs)
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "inputs": [str(p) for p in log_paths],
        "row_count": 0,
        "rows_with_grader_output": 0,
        "rows_with_sample_review": 0,
        "rows_with_global_review": 0,
        "item_count": 0,
        "explicit_suggestion_count": 0,
        "suggestions": {},
        "unique_suggestion_count": 0,
        "max_suggestion_repeat": 0,
        "themes": {},
        "desiderata_level_mean": {},
        "desiderata_low_rate": {},
    }

    suggestion_counter: Counter[str] = Counter()
    theme_counter: Counter[str] = Counter()
    desir_level_sum = [0.0] * 7
    desir_level_count = [0] * 7
    desir_low_count = [0] * 7
    theme_examples: dict[str, list[str]] = defaultdict(list)
    suggestion_examples: list[dict] = []
    all_suggestions: list[dict] = []
    sample_examples: list[dict] = []

    for log_path in log_paths:
        processed_for_file = 0
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if args.max_rows and processed_for_file >= args.max_rows:
                    break
                payload = json.loads(line)
                summary["row_count"] = int(summary["row_count"]) + 1
                grader_output = payload.get("grader_output") or ""
                if not grader_output:
                    continue
                if args.status == "valid" and "status" in payload and payload.get("status") != "valid":
                    continue

                processed_for_file += 1
                summary["rows_with_grader_output"] = int(summary["rows_with_grader_output"]) + 1

                weaknesses, fixes = parse_sample_review(grader_output)
                if weaknesses or fixes:
                    summary["rows_with_sample_review"] = int(summary["rows_with_sample_review"]) + 1
                    if len(sample_examples) < 8:
                        sample_examples.append(
                            {
                                "source_file": str(log_path),
                                "batch_idx": payload.get("batch_idx"),
                                "group_idx": payload.get("group_idx"),
                                "sample_idx": payload.get("sample_idx"),
                                "weaknesses": " | ".join(weaknesses[:3]),
                                "fixes": " | ".join(fixes[:3]),
                            }
                        )

                global_review = parse_global_review(grader_output)
                if global_review:
                    summary["rows_with_global_review"] = int(summary["rows_with_global_review"]) + 1
                    for theme in detect_themes(global_review):
                        theme_counter[theme] += 1
                        if len(theme_examples[theme]) < 5:
                            theme_examples[theme].append(global_review)

                items = parse_item_blocks(grader_output)
                summary["item_count"] = int(summary["item_count"]) + len(items)

                for item in items:
                    for idx, level in enumerate(item["levels"], start=1):
                        desir_level_sum[idx - 1] += float(level)
                        desir_level_count[idx - 1] += 1
                        if level <= 1:
                            desir_low_count[idx - 1] += 1

                    if item["summary"]:
                        for theme in detect_themes(item["summary"]):
                            theme_counter[theme] += 1
                            if len(theme_examples[theme]) < 5:
                                theme_examples[theme].append(item["summary"])

                    if item["suggestion"]:
                        summary["explicit_suggestion_count"] = int(summary["explicit_suggestion_count"]) + 1
                        suggestion_counter[item["suggestion"]] += 1
                        suggestion_row = {
                            "source_file": str(log_path),
                            "batch_idx": payload.get("batch_idx"),
                            "group_idx": payload.get("group_idx"),
                            "sample_idx": payload.get("sample_idx"),
                            "item_num": item.get("item_num"),
                            "criteria": item["criteria"],
                            "suggestion": item["suggestion"],
                            "suggestion_normalized": normalize_suggestion_text(item["suggestion"]),
                            "summary": item["summary"],
                            "d1_level": item["levels"][0],
                            "d2_level": item["levels"][1],
                            "d3_level": item["levels"][2],
                            "d4_level": item["levels"][3],
                            "d5_level": item["levels"][4],
                            "d6_level": item["levels"][5],
                            "d7_level": item["levels"][6],
                        }
                        all_suggestions.append(suggestion_row)
                        if len(suggestion_examples) < 20:
                            suggestion_examples.append(suggestion_row)

    total_items = int(summary["item_count"])
    for idx in range(1, 8):
        mean = (
            desir_level_sum[idx - 1] / float(desir_level_count[idx - 1])
            if desir_level_count[idx - 1]
            else None
        )
        low_rate = (
            desir_low_count[idx - 1] / float(desir_level_count[idx - 1])
            if desir_level_count[idx - 1]
            else None
        )
        summary["desiderata_level_mean"][f"D{idx}"] = mean
        summary["desiderata_low_rate"][f"D{idx}"] = low_rate

    summary["suggestions"] = dict(suggestion_counter.most_common(20))
    summary["unique_suggestion_count"] = len(suggestion_counter)
    summary["max_suggestion_repeat"] = (
        suggestion_counter.most_common(1)[0][1] if suggestion_counter else 0
    )
    summary["themes"] = dict(theme_counter.most_common())
    summary["verdict"] = verdict_from_signals(
        explicit_suggestion_count=int(summary["explicit_suggestion_count"]),
        total_items=total_items,
        suggestion_counter=suggestion_counter,
        theme_counter=theme_counter,
    )

    report_lines: list[str] = []
    report_lines.append("# Desiderata Comment Audit")
    report_lines.append("")
    report_lines.append("## Inputs")
    for path in log_paths:
        report_lines.append(f"- {path}")
    report_lines.append("")
    report_lines.append("## Coverage")
    report_lines.append(f"- Rows scanned: {summary['row_count']}")
    report_lines.append(f"- Rows with grader output used: {summary['rows_with_grader_output']}")
    report_lines.append(f"- Rows with sample review: {summary['rows_with_sample_review']}")
    report_lines.append(f"- Rows with global desiderata review: {summary['rows_with_global_review']}")
    report_lines.append(f"- Rubric items parsed: {summary['item_count']}")
    report_lines.append("")
    report_lines.append("## Verdict")
    report_lines.append(summary["verdict"])
    report_lines.append("")
    report_lines.append("## Direct Suggestion Evidence")
    report_lines.append(
        f"- Explicit new-desiderata suggestions: {summary['explicit_suggestion_count']}"
    )
    report_lines.append(
        f"- Unique suggestion texts: {summary['unique_suggestion_count']}"
    )
    report_lines.append(
        f"- Max repeat count for any single suggestion: {summary['max_suggestion_repeat']}"
    )
    if suggestion_counter:
        for suggestion, count in suggestion_counter.most_common(10):
            report_lines.append(f"- {count}x: {suggestion}")
    else:
        report_lines.append("- No explicit `suggested_new_desiderata` suggestions were found.")
    report_lines.append("")
    report_lines.append("## Low-Level Pattern By Desideratum")
    for idx in range(1, 8):
        mean = summary["desiderata_level_mean"][f"D{idx}"]
        low_rate = summary["desiderata_low_rate"][f"D{idx}"]
        mean_text = f"{mean:.3f}" if isinstance(mean, float) else "NA"
        low_text = f"{low_rate:.3f}" if isinstance(low_rate, float) else "NA"
        report_lines.append(
            f"- D{idx} `{DESIDERATA_NAMES[idx]}`: mean level={mean_text}, low-rate(level<=1)={low_text}"
        )
    report_lines.append("")
    report_lines.append("## Recurring Comment Themes")
    if theme_counter:
        for theme, count in theme_counter.most_common(12):
            report_lines.append(f"- {theme}: {count}")
            for example in theme_examples[theme][:2]:
                report_lines.append(f"  Example: {example}")
    else:
        report_lines.append("- No recurring comment themes detected.")

    write_json(out_dir / "summary.json", summary)
    write_csv(
        out_dir / "all_suggestions.csv",
        all_suggestions,
        [
            "source_file",
            "batch_idx",
            "group_idx",
            "sample_idx",
            "item_num",
            "criteria",
            "suggestion",
            "suggestion_normalized",
            "summary",
            "d1_level",
            "d2_level",
            "d3_level",
            "d4_level",
            "d5_level",
            "d6_level",
            "d7_level",
        ],
    )
    write_jsonl(out_dir / "all_suggestions.jsonl", all_suggestions)
    write_csv(
        out_dir / "explicit_suggestions.csv",
        suggestion_examples,
        [
            "source_file",
            "batch_idx",
            "group_idx",
            "sample_idx",
            "item_num",
            "criteria",
            "suggestion",
            "suggestion_normalized",
            "summary",
            "d1_level",
            "d2_level",
            "d3_level",
            "d4_level",
            "d5_level",
            "d6_level",
            "d7_level",
        ],
    )
    write_csv(
        out_dir / "suggestion_exact_counts.csv",
        [
            {"suggestion": suggestion, "count": count}
            for suggestion, count in suggestion_counter.most_common()
        ],
        ["suggestion", "count"],
    )
    normalized_counter: Counter[str] = Counter(
        row["suggestion_normalized"] for row in all_suggestions
    )
    normalized_examples: dict[str, str] = {}
    for row in all_suggestions:
        key = row["suggestion_normalized"]
        if key and key not in normalized_examples:
            normalized_examples[key] = row["suggestion"]
    write_csv(
        out_dir / "suggestion_normalized_counts.csv",
        [
            {
                "suggestion_normalized": key,
                "count": count,
                "example_suggestion": normalized_examples.get(key, ""),
            }
            for key, count in normalized_counter.most_common()
        ],
        ["suggestion_normalized", "count", "example_suggestion"],
    )
    write_csv(
        out_dir / "sample_review_examples.csv",
        sample_examples,
        [
            "source_file",
            "batch_idx",
            "group_idx",
            "sample_idx",
            "weaknesses",
            "fixes",
        ],
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(summary["verdict"])
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
