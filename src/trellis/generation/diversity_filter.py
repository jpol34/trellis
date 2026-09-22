"""Within-scenario-cell near-duplicate detection for a batch of generated transcripts.

Embeds each transcript with a pluggable `Embedder` and flags pairs of items in the same
scenario cell (category, call_reason, variability_tier) whose cosine distance falls below a
threshold. Cross-cell similarity is never checked — items in different cells (different call
reasons, different tiers) are expected to differ and aren't compared to each other; some
similarity *within* a cell is normal (shared category fields, shared tier tags) and only very
close pairs are worth a human look.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from trellis.generation.review import write_flagged

# `list[str] -> list[list[float]]`, one embedding vector per input text, same order.
Embedder = Callable[[list[str]], list[list[float]]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def tfidf_embedder(texts: list[str]) -> list[list[float]]:
    """Default `Embedder`: a small hand-rolled TF-IDF vectorization over a vocabulary shared
    across `texts`. Deterministic and network-free, with no dependency beyond the stdlib — good
    enough to catch near-verbatim duplicate transcripts, which is all this filter needs."""
    tokenized = [_tokenize(t) for t in texts]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    n_docs = len(texts)
    vocab = sorted(doc_freq)
    idf = {term: math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0 for term in vocab}

    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        length = max(len(tokens), 1)
        vectors.append([(tf[term] / length) * idf[term] for term in vocab])
    return vectors


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity; 0.0 for identical direction, 1.0 for orthogonal, up to 2.0 for
    opposite. Either vector being all-zero (e.g. two texts sharing no vocabulary at all) is
    treated as maximally distant rather than raising on the zero-norm division."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


@dataclass(frozen=True)
class ScenarioItem:
    item_id: str
    transcript: str
    category: str
    call_reason: str
    variability_tier: str

    @property
    def scenario_cell(self) -> tuple[str, str, str]:
        return (self.category, self.call_reason, self.variability_tier)


@dataclass(frozen=True)
class NearDuplicatePair:
    item_a: str
    item_b: str
    scenario_cell: tuple[str, str, str]
    distance: float

    @property
    def flag_id(self) -> str:
        a, b = sorted((self.item_a, self.item_b))
        return f"{'/'.join(self.scenario_cell)}::{a}::{b}"


def find_near_duplicates(
    items: list[ScenarioItem],
    *,
    embedder: Embedder = tfidf_embedder,
    threshold: float = 0.15,
) -> list[NearDuplicatePair]:
    """Compares every pair of items sharing a scenario cell and flags pairs whose cosine
    distance is below `threshold`. Cells with fewer than 2 items are skipped (nothing to
    compare)."""
    by_cell: dict[tuple[str, str, str], list[ScenarioItem]] = {}
    for item in items:
        by_cell.setdefault(item.scenario_cell, []).append(item)

    pairs: list[NearDuplicatePair] = []
    for cell, cell_items in by_cell.items():
        if len(cell_items) < 2:
            continue
        vectors = embedder([i.transcript for i in cell_items])
        for (i, vec_i), (j, vec_j) in combinations(enumerate(vectors), 2):
            distance = cosine_distance(vec_i, vec_j)
            if distance < threshold:
                pairs.append(
                    NearDuplicatePair(
                        item_a=cell_items[i].item_id,
                        item_b=cell_items[j].item_id,
                        scenario_cell=cell,
                        distance=distance,
                    )
                )
    return pairs


def write_diversity_review(pairs: list[NearDuplicatePair], path: Path) -> None:
    """Writes flagged near-duplicate pairs to a human-editable review file, preserving any
    already-reviewed entries from a prior run."""
    flags = [
        {
            "flag_id": pair.flag_id,
            "item_a": pair.item_a,
            "item_b": pair.item_b,
            "scenario_cell": list(pair.scenario_cell),
            "distance": pair.distance,
        }
        for pair in pairs
    ]
    write_flagged(flags, path)
