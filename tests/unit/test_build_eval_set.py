from __future__ import annotations

import ast
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from faker import Faker

from trellis.generation.build_eval_set import build_eval_set
from trellis.generation.corpus_common import VARIABILITY_TIERS
from trellis.generation.distractors import build_candidates
from trellis.generation.personas import Persona
from trellis.generation.values import generate_value
from trellis.schema.loader import CATEGORIES_DIR, REPO_ROOT, load_all_categories

CATEGORIES = {spec.category: spec for spec in load_all_categories(CATEGORIES_DIR)}
GENERATION_DIR = REPO_ROOT / "src" / "trellis" / "generation"


def _make_fake_generate_fn(seed: int = 0):
    faker = Faker()
    faker.seed_instance(seed)
    rng = random.Random(seed)
    counter = itertools.count()

    async def generate_fn(category, call_reason, variability_tier) -> dict:
        idx = next(counter)
        ground_truth = []
        for f in category.fields:
            value = generate_value(f, faker, rng)
            candidates, correct_index = build_candidates(value, f, faker, rng)
            ground_truth.append(
                {
                    "field": f.name,
                    "value": value,
                    "candidates": candidates,
                    "correct_index": correct_index,
                    "char_offset": None,
                }
            )
        transcript = f"Caller: hi, calling about {call_reason}, eval item {idx}. Agent: got it."
        persona = Persona(
            age_range="40-50", tone="curt", verbosity="terse", background="n/a",
            speech_quirks="none",
        )
        return {"transcript": transcript, "persona": persona, "ground_truth": ground_truth}

    return generate_fn


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_build_eval_set_produces_correct_total(tmp_path):
    n = await build_eval_set(total=100, out_dir=tmp_path, generate_fn=_make_fake_generate_fn())
    assert n == 100
    records = _read_jsonl(tmp_path / "eval.jsonl")
    assert len(records) == 100


async def test_build_eval_set_tier_split_is_approximately_17_17_16_per_category(tmp_path):
    await build_eval_set(total=100, out_dir=tmp_path, generate_fn=_make_fake_generate_fn())
    records = _read_jsonl(tmp_path / "eval.jsonl")

    per_category_tier = Counter((r["category"], r["variability_tier"]) for r in records)
    for category in CATEGORIES:
        per_tier = [per_category_tier[(category, tier)] for tier in VARIABILITY_TIERS]
        assert sum(per_tier) == 50
        assert sorted(per_tier) == [16, 17, 17]


async def test_build_eval_set_records_validate_against_schema_and_pass_quality_gates(tmp_path):
    from trellis.generation.corpus_common import record_to_quality_gate_item
    from trellis.generation.quality_gates import run_quality_gates

    await build_eval_set(total=20, out_dir=tmp_path, generate_fn=_make_fake_generate_fn())
    records = _read_jsonl(tmp_path / "eval.jsonl")
    assert len(records) == 20

    for r in records:
        assert r["category"] in CATEGORIES
        category = CATEGORIES[r["category"]]
        assert r["call_reason"] in category.scenarios.call_reasons
        assert r["variability_tier"] in VARIABILITY_TIERS
        expected_fields = {f.name for f in category.fields}
        assert {f["field"] for f in r["fields"]} == expected_fields
        for f in r["fields"]:
            assert f["value"] in f["candidates"]
            assert f["candidates"][f["correct_index"]] == f["value"]

    # Same fixture batch the eval set itself was built from, run through the real quality gates
    # (ticket #6) — must not raise.
    report = run_quality_gates(
        [record_to_quality_gate_item(r) for r in records],
        diversity_review_path=tmp_path / "qg" / "diversity.json",
        injection_review_path=tmp_path / "qg" / "injection.json",
        distractor_review_path=tmp_path / "qg" / "distractor.json",
    )
    assert report is not None


async def test_build_eval_set_reproducible_given_fixed_seed(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    await build_eval_set(total=30, out_dir=out_a, generate_fn=_make_fake_generate_fn(seed=5))
    await build_eval_set(total=30, out_dir=out_b, generate_fn=_make_fake_generate_fn(seed=5))

    assert _read_jsonl(out_a / "eval.jsonl") == _read_jsonl(out_b / "eval.jsonl")


async def test_build_eval_set_never_touches_training_corpus_output(tmp_path):
    training_dir = tmp_path / "training_corpus"
    eval_dir = tmp_path / "eval_set"
    training_dir.mkdir()
    (training_dir / "sentinel.txt").write_text("do not touch", encoding="utf-8")

    await build_eval_set(total=10, out_dir=eval_dir, generate_fn=_make_fake_generate_fn())

    assert (training_dir / "sentinel.txt").read_text(encoding="utf-8") == "do not touch"
    assert not any(p.name == "sentinel.txt" for p in eval_dir.rglob("*"))


def test_build_eval_set_module_does_not_import_build_training_corpus():
    tree = ast.parse((GENERATION_DIR / "build_eval_set.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("build_training_corpus" in name for name in imported)
