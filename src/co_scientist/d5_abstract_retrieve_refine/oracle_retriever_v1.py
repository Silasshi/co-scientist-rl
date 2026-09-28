"""D5 Exp C — slim oracle retriever via local HF transformers + BAAI/bge-small-en-v1.5.

Parses slim.md into 87 items (Insights/Methodology/Theory/Math/Empirical/Failure_modes),
embeds each at startup using bge-small (CLS pooling + L2-norm), exposes
`retrieve(query, k)` returning concatenated top-K item bodies.

Used by `train_mu_v8_d5sdpo.py` when `config.use_rag=True` to swap full
oracle (10K words) for top-K retrieved items (~500-1000 words).

bge usage notes:
- bge models use CLS-token pooling (NOT mean pooling), then L2-normalize
- For S2P (short query → long passage) retrieval, prefix query with the
  instruction "Represent this sentence for searching relevant passages: "
  (per BGE model card; documents are NOT prefixed)
- Cosine similarity == dot product since both query+doc are L2-normalized

Standalone smoke test:
    PYTHONPATH=src python -c "
    from co_scientist.d5_abstract_retrieve_refine.oracle_retriever_v1 import OracleRetrieverV1
    r = OracleRetrieverV1.load_default()
    out = r.retrieve('test-time RL with risk-sensitive objective for discovery', k=5)
    print('--- top-5 ---'); print(out[:1500])
    "
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SLIM_PATH = (
    REPO_ROOT
    / "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"
)
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


_ITEM_HEADER_RE = re.compile(
    r"^### (Insights|Methodology|Theory|Math|Empirical|Failure_modes) (\d+):",
    re.MULTILINE,
)


@dataclass
class OracleItem:
    section: str  # e.g. "Math"
    number: int  # e.g. 5
    header: str  # full header line like "### Math 5: Risk-sensitive (exponential-utility) objective"
    body: str  # full block including header through next-section start


def parse_slim(slim_path: Path) -> list[OracleItem]:
    text = slim_path.read_text()
    matches = list(_ITEM_HEADER_RE.finditer(text))
    items: list[OracleItem] = []
    for i, m in enumerate(matches):
        section = m.group(1)
        number = int(m.group(2))
        # Header line ends at first newline after match.start
        header_end = text.find("\n", m.start())
        header = text[m.start() : header_end if header_end != -1 else m.end()]
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start() : block_end].rstrip()
        items.append(OracleItem(section=section, number=number, header=header, body=body))
    return items


class OracleRetrieverV1:
    """One-shot embed at construction, then per-query top-K cosine retrieval.

    Constructor is heavy (downloads bge-small ~33MB + embeds 87 items, <5s on CPU).
    Each retrieve() call embeds the query (one short forward pass) + matmul.
    """

    def __init__(
        self,
        items: list[OracleItem],
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_query_tokens: int = 128,
        max_doc_tokens: int = 512,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.items = items
        self.device = torch.device(device)
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens

        logger.info(
            "OracleRetrieverV1: loading %s on %s, embedding %d items",
            model_name,
            device,
            len(items),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Embed all items once
        bodies = [it.body for it in items]
        self.doc_embeddings = self._embed(bodies, max_length=max_doc_tokens, is_query=False)
        # shape (N, D)
        logger.info("OracleRetrieverV1 ready: %s", tuple(self.doc_embeddings.shape))

    @classmethod
    def load_default(cls, slim_path: Path | str | None = None, **kwargs) -> "OracleRetrieverV1":
        path = Path(slim_path) if slim_path else DEFAULT_SLIM_PATH
        items = parse_slim(path)
        if len(items) != 87:
            logger.warning("Expected 87 items in slim.md, got %d", len(items))
        return cls(items, **kwargs)

    @torch.no_grad()
    def _embed(self, texts: list[str], max_length: int, is_query: bool) -> torch.Tensor:
        if is_query:
            texts = [QUERY_INSTRUCTION + t for t in texts]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(self.device)
        outputs = self.model(**encoded)
        # bge convention: CLS-token (index 0) embedding, then L2-normalize
        cls_emb = outputs.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls_emb, p=2, dim=1)

    @torch.no_grad()
    def retrieve(self, query: str, k: int = 5, joiner: str = "\n\n") -> str:
        """Return concatenated body of top-K most-similar items (cosine sim)."""
        q_emb = self._embed([query], max_length=self.max_query_tokens, is_query=True)
        scores = (q_emb @ self.doc_embeddings.T).squeeze(0)  # (N,)
        topk = torch.topk(scores, k=min(k, len(self.items)))
        idxs = topk.indices.tolist()
        ranked = [self.items[i] for i in idxs]
        return joiner.join(it.body for it in ranked)

    @torch.no_grad()
    def retrieve_with_scores(
        self, query: str, k: int = 5
    ) -> list[tuple[OracleItem, float]]:
        q_emb = self._embed([query], max_length=self.max_query_tokens, is_query=True)
        scores = (q_emb @ self.doc_embeddings.T).squeeze(0)
        topk = torch.topk(scores, k=min(k, len(self.items)))
        return [(self.items[i], float(scores[i])) for i in topk.indices.tolist()]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = OracleRetrieverV1.load_default()
    print("\n--- smoke retrieve ---")
    test_query = "test-time RL with risk-sensitive objective for scientific discovery"
    out = r.retrieve_with_scores(test_query, k=5)
    for it, score in out:
        print(f"  {score:.4f}  {it.header}")
    print("\n--- joined retrieved text (top-5) ---")
    print(r.retrieve(test_query, k=5)[:2000])
