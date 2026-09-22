"""Mechanically enforces that `build_training_corpus.py` and `build_eval_set.py` never import
each other (they may both import any other shared module, e.g. `corpus_common`,
`quality_gates`, `transcript_gen`) — see their module docstrings for why."""

from __future__ import annotations

import ast

from trellis.schema.loader import REPO_ROOT

GENERATION_DIR = REPO_ROOT / "src" / "trellis" / "generation"
TRAINING_CORPUS_PATH = GENERATION_DIR / "build_training_corpus.py"
EVAL_SET_PATH = GENERATION_DIR / "build_eval_set.py"


def _imported_module_names(path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_training_corpus_and_eval_set_modules_exist():
    assert TRAINING_CORPUS_PATH.is_file()
    assert EVAL_SET_PATH.is_file()


def test_build_training_corpus_does_not_import_build_eval_set():
    imported = _imported_module_names(TRAINING_CORPUS_PATH)
    assert not any(name == "trellis.generation.build_eval_set" or name.endswith(".build_eval_set")
                    for name in imported), imported


def test_build_eval_set_does_not_import_build_training_corpus():
    imported = _imported_module_names(EVAL_SET_PATH)
    assert not any(
        name == "trellis.generation.build_training_corpus"
        or name.endswith(".build_training_corpus")
        for name in imported
    ), imported
