"""D5 Phase 4a — persist S2 forward-citation scan for TTT-Discover follow-ups.

Writes data/cross_goal/candidate_scan.jsonl with one row per citing paper.
Reproducibility purpose only — selection rule and locked picks are pre-registered
in projects/d5_abstract_retrieve_refine/DECISIONS.md.

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.cross_goal_candidate_scan \
        out_path=projects/d5_abstract_retrieve_refine/data/cross_goal/candidate_scan.jsonl \
        source_arxiv_id=2601.16175 \
        limit=200
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.shared.paper_retrieval import S2BackwardRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    out_path: str = "projects/d5_abstract_retrieve_refine/data/cross_goal/candidate_scan.jsonl"
    source_arxiv_id: str = "2601.16175"  # TTT-Discover
    limit: int = 200


def main(config: Config):
    out = (REPO_ROOT / config.out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    retriever = S2BackwardRetriever()
    refs = retriever.fetch_paper_incoming_citations(config.source_arxiv_id, limit=config.limit)
    logger.info("Persisting %d incoming citations of arxiv:%s -> %s",
                len(refs), config.source_arxiv_id, out)

    with open(out, "w") as f:
        for r in refs:
            f.write(json.dumps(r.to_dict()) + "\n")

    retriever.close()
    logger.info("Wrote %d rows to %s", len(refs), out)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
