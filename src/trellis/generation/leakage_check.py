"""Two cheap, explainable heuristic checks for shortcut/leakage signals in generated items —
flag-for-human-review tools, not precise classifiers.

1. Injection-point cleanliness: the sentence carrying an injected ground-truth value shouldn't
   read like a fill-in-the-blank template ("My email is X.") when the tier calls for natural
   variability — a classifier could learn to key off that template shape instead of reading the
   surrounding dialogue.
2. Distractor-tell: within one field's candidate set, the real value shouldn't be surface-level
   distinguishable from its distractors (e.g. always the "cleanest" one) in a way a classifier
   could shortcut on without ever reasoning about the transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from trellis.generation.distractors import NOT_MENTIONED
from trellis.generation.review import write_flagged
from trellis.generation.transcript_gen import load_variability_tags

_TURN_PREFIX_RE = re.compile(r"^(caller|agent)\s*:\s*", re.I)
# Punctuation followed by whitespace or end-of-string only — a bare lookahead-free split would
# also cut mid-token on values containing '.', like an email's domain (jofisher23@gmail.com).
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")
_NATURAL_MARKERS = (
    "uh", "um", "you know", "i mean", "well,", "actually", "kind of", "sort of",
    "hmm", "er,", "let me", "sorry", "...",
)
_HEDGE_TYPO_RE = re.compile(r"\b(uh|um|kinda|sorta|maybe|approx|around|about|i think)\b", re.I)


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Splits `text` into (sentence, start_offset, end_offset) spans on '.', '!', '?'. Not a
    real sentence tokenizer — good enough to locate the sentence around a known char offset."""
    sentences: list[tuple[str, int, int]] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        end = m.end()
        sentences.append((text[start:end], start, end))
        start = end
    if start < len(text):
        sentences.append((text[start:], start, len(text)))
    return [(s, a, b) for s, a, b in sentences if s.strip()]


def _sentence_containing(
    sentences: list[tuple[str, int, int]], char_offset: int | None, value: str
) -> str | None:
    if char_offset is not None:
        for sentence, start, end in sentences:
            if start <= char_offset < end:
                return sentence
    value_l = value.lower()
    for sentence, _, _ in sentences:
        if value_l in sentence.lower():
            return sentence
    return None


def _normalize_sentence(sentence: str) -> str:
    s = _TURN_PREFIX_RE.sub("", sentence.strip())
    s = s.strip(" .!?")
    return re.sub(r"\s+", " ", s).lower()


def _generic_templates(field_label: str, value: str) -> set[str]:
    field_l = field_label.lower()
    value_l = value.lower()
    return {
        f"my {field_l} is {value_l}",
        f"the {field_l} is {value_l}",
        f"{field_l} is {value_l}",
        f"it's {value_l}",
        f"it is {value_l}",
        f"{value_l} is my {field_l}",
        f"my {field_l}: {value_l}",
        f"{field_l}: {value_l}",
    }


def _average_sentence_word_count(sentences: list[tuple[str, int, int]]) -> float:
    counts = [len(_TURN_PREFIX_RE.sub("", s).split()) for s, _, _ in sentences]
    return sum(counts) / len(counts) if counts else 0.0


@dataclass(frozen=True)
class InjectionCleanlinessFlag:
    item_id: str
    field: str
    sentence: str
    reason: str

    @property
    def flag_id(self) -> str:
        # Includes `reason` so a rerun that flags the same item/field for a *different* reason
        # is treated as a new, unreviewed entry rather than silently inheriting an old review
        # decision that was made about a different problem.
        return f"{self.item_id}::{self.field}::injection::{self.reason}"


def check_injection_cleanliness(
    item_id: str,
    transcript: str,
    field: str,
    value: str,
    char_offset: int | None,
    tier_tags: dict[str, list[str]],
) -> InjectionCleanlinessFlag | None:
    """Flags `field`'s injection sentence if it reads as a generic fill-in-the-blank template
    regardless of persona/tier, or — when the tier calls for natural variability — if it's
    markedly shorter than the transcript's average sentence and shows none of the natural-speech
    markers (fillers, hedges, self-corrections) a varied tier would produce."""
    sentences = _split_sentences(transcript)
    if not sentences:
        return None
    sentence = _sentence_containing(sentences, char_offset, value)
    if sentence is None:
        return None

    normalized = _normalize_sentence(sentence)
    field_label = field.replace("_", " ")
    if normalized in _generic_templates(field_label, value):
        return InjectionCleanlinessFlag(item_id, field, sentence.strip(), "templated_pattern")

    has_tags = bool(tier_tags.get("linguistic")) or bool(tier_tags.get("conversational"))
    if has_tags:
        avg_words = _average_sentence_word_count(sentences)
        word_count = len(_TURN_PREFIX_RE.sub("", sentence).split())
        is_short = avg_words > 0 and word_count > 0 and word_count < 0.6 * avg_words
        has_marker = any(marker in sentence.lower() for marker in _NATURAL_MARKERS)
        if is_short and not has_marker:
            return InjectionCleanlinessFlag(item_id, field, sentence.strip(), "short_and_formulaic")

    return None


def _surface_features(s: str) -> tuple[int, float, bool]:
    length = len(s)
    non_alnum = sum(1 for c in s if not c.isalnum() and not c.isspace())
    punct_density = non_alnum / max(length, 1)
    has_marker = bool(_HEDGE_TYPO_RE.search(s))
    return length, punct_density, has_marker


@dataclass(frozen=True)
class DistractorTellFlag:
    item_id: str
    field: str
    value: str
    reason: str

    @property
    def flag_id(self) -> str:
        # See InjectionCleanlinessFlag.flag_id — `reason` is part of the id for the same reason.
        return f"{self.item_id}::{self.field}::distractor_tell::{self.reason}"


# Strategies whose distractors are literal other members of the same finite pool the real value
# was drawn from (not a transformation of it) — the real value and its distractors are
# interchangeable samples from one set, so a hedge-marker mismatch between them is expected pool
# variety, not a generation-introduced tell. The marker checks below don't apply here.
_POOL_SOURCED_STRATEGIES = frozenset({"swap_sibling_value", "shift_window"})


def check_distractor_tell(
    item_id: str, field: str, value: str, candidates: list[str], distractor_strategy: str = ""
) -> DistractorTellFlag | None:
    """Flags a field's candidate set if the real value is surface-level distinguishable from
    every distractor: uniquely lacking (or uniquely carrying) a hedge/typo marker, having no
    punctuation where every distractor has some, or being a length outlier relative to the
    distractor set."""
    distractors = [c for c in candidates if c != value and c != NOT_MENTIONED]
    if not distractors:
        return None

    value_len, value_punct, value_marker = _surface_features(value)
    d_features = [_surface_features(d) for d in distractors]
    d_markers = [m for _, _, m in d_features]

    if distractor_strategy not in _POOL_SOURCED_STRATEGIES:
        if not value_marker and all(d_markers):
            return DistractorTellFlag(item_id, field, value, "real_value_missing_expected_marker")
        if value_marker and not any(d_markers):
            return DistractorTellFlag(item_id, field, value, "real_value_uniquely_marked")

    d_puncts = [p for _, p, _ in d_features]
    if value_punct == 0.0 and min(d_puncts) > 0.15:
        return DistractorTellFlag(item_id, field, value, "real_value_lacks_punctuation")

    d_lens = [ln for ln, _, _ in d_features]
    if value_len > 0 and (value_len < 0.5 * min(d_lens) or value_len > 2.0 * max(d_lens)):
        return DistractorTellFlag(item_id, field, value, "real_value_length_outlier")

    return None


@dataclass(frozen=True)
class LeakageCheckField:
    field: str
    value: str
    candidates: list[str]
    char_offset: int | None
    distractor_strategy: str = ""


@dataclass(frozen=True)
class LeakageCheckItem:
    item_id: str
    transcript: str
    variability_tier: str
    fields: list[LeakageCheckField] = dataclass_field(default_factory=list)


def run_leakage_check(
    items: list[LeakageCheckItem],
) -> tuple[list[InjectionCleanlinessFlag], list[DistractorTellFlag]]:
    """Runs both heuristic checks over every field of every item in the batch."""
    injection_flags: list[InjectionCleanlinessFlag] = []
    distractor_flags: list[DistractorTellFlag] = []
    for item in items:
        tier_tags = load_variability_tags(item.variability_tier)
        for f in item.fields:
            inj = check_injection_cleanliness(
                item.item_id, item.transcript, f.field, f.value, f.char_offset, tier_tags
            )
            if inj is not None:
                injection_flags.append(inj)
            dist = check_distractor_tell(
                item.item_id, f.field, f.value, f.candidates, f.distractor_strategy
            )
            if dist is not None:
                distractor_flags.append(dist)
    return injection_flags, distractor_flags


def write_injection_review(flags: list[InjectionCleanlinessFlag], path: Path) -> None:
    write_flagged(
        [
            {
                "flag_id": f.flag_id,
                "item_id": f.item_id,
                "field": f.field,
                "sentence": f.sentence,
                "reason": f.reason,
            }
            for f in flags
        ],
        path,
    )


def write_distractor_review(flags: list[DistractorTellFlag], path: Path) -> None:
    write_flagged(
        [
            {
                "flag_id": f.flag_id,
                "item_id": f.item_id,
                "field": f.field,
                "value": f.value,
                "reason": f.reason,
            }
            for f in flags
        ],
        path,
    )
