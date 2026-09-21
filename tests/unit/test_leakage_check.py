from __future__ import annotations

import json

from trellis.generation.leakage_check import (
    check_distractor_tell,
    check_injection_cleanliness,
    write_distractor_review,
    write_injection_review,
)

CLEAN_TAGS = {"linguistic": [], "conversational": []}
NOISY_TAGS = {
    "linguistic": ["typos", "speech_asr_errors", "paraphrase", "simplification", "verbosity"],
    "conversational": ["disfluency", "interruption", "self_correction", "turn_spread_disclosure"],
}


def test_templated_injection_sentence_is_flagged():
    transcript = "Caller: My email is jofisher23@gmail.com. Agent: Great, thanks for that."
    offset = transcript.index("jofisher23@gmail.com")
    flag = check_injection_cleanliness(
        "item-1", transcript, "email", "jofisher23@gmail.com", offset, CLEAN_TAGS
    )
    assert flag is not None
    assert flag.reason == "templated_pattern"


def test_naturally_varied_injection_sentence_is_not_flagged():
    transcript = (
        "Caller: So I've been meaning to call about the two bedroom for a couple weeks now, "
        "sorry it took me so long to get around to it. "
        "Agent: No worries at all, happy to help whenever. "
        "Caller: Yeah, um, so it's jofisher23@gmail.com, sorry let me spell that out for you, "
        "j-o-fisher-two-three at gmail. "
        "Agent: Got it, thanks."
    )
    offset = transcript.index("jofisher23@gmail.com")
    flag = check_injection_cleanliness(
        "item-1", transcript, "email", "jofisher23@gmail.com", offset, NOISY_TAGS
    )
    assert flag is None


def test_short_formulaic_sentence_flagged_when_tier_expects_variability():
    transcript = (
        "Caller: So I wanted to reach out today because I've been thinking about moving for a "
        "little while now and this place kept coming up in my search results, seemed worth a call. "
        "That's 555-201-4477. "
        "Agent: Great, thanks, let me pull that up for you right now."
    )
    offset = transcript.index("555-201-4477")
    flag = check_injection_cleanliness(
        "item-1", transcript, "phone", "555-201-4477", offset, NOISY_TAGS
    )
    assert flag is not None
    assert flag.reason == "short_and_formulaic"


def test_short_sentence_not_flagged_when_tier_is_clean():
    transcript = (
        "Caller: So I wanted to reach out today because I've been thinking about moving for a "
        "little while now and this place kept coming up in my search results, seemed worth a call. "
        "That's 555-201-4477. "
        "Agent: Great, thanks, let me pull that up for you right now."
    )
    offset = transcript.index("555-201-4477")
    flag = check_injection_cleanliness(
        "item-1", transcript, "phone", "555-201-4477", offset, CLEAN_TAGS
    )
    assert flag is None


def test_char_offset_none_falls_back_to_value_search():
    transcript = "Caller: My email is jofisher23@gmail.com. Agent: Great, thanks for that."
    flag = check_injection_cleanliness(
        "item-1", transcript, "email", "jofisher23@gmail.com", None, CLEAN_TAGS
    )
    assert flag is not None


def test_value_absent_from_transcript_returns_none():
    transcript = "Caller: I'd like to ask about pricing. Agent: Sure."
    flag = check_injection_cleanliness(
        "item-1", transcript, "email", "nobody@example.com", None, CLEAN_TAGS
    )
    assert flag is None


def test_distractor_tell_flags_real_value_missing_expected_marker():
    candidates = ["around 1800", "about 1900", "maybe 2000", "1950"]
    flag = check_distractor_tell("item-1", "budget", "1950", candidates)
    assert flag is not None
    assert flag.reason == "real_value_missing_expected_marker"


def test_distractor_tell_not_flagged_when_features_are_consistent():
    candidates = ["1950", "1975", "2010", "1600"]
    flag = check_distractor_tell("item-1", "budget", "1950", candidates)
    assert flag is None


def test_distractor_tell_ignores_not_mentioned_candidate():
    candidates = ["1950", "1975", "2010", "not mentioned"]
    flag = check_distractor_tell("item-1", "budget", "1950", candidates)
    assert flag is None


def test_write_injection_review_preserves_reviewed_entry(tmp_path):
    path = tmp_path / "injection_review.json"
    transcript = "Caller: My email is jofisher23@gmail.com. Agent: Great, thanks for that."
    offset = transcript.index("jofisher23@gmail.com")
    flag = check_injection_cleanliness(
        "item-1", transcript, "email", "jofisher23@gmail.com", offset, CLEAN_TAGS
    )
    write_injection_review([flag], path)

    entries = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["reviewed"] is None

    entries[0]["reviewed"] = False
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    other_transcript = "Caller: My email is someone@example.com. Agent: Great, thanks."
    other_offset = other_transcript.index("someone@example.com")
    other_flag = check_injection_cleanliness(
        "item-2", other_transcript, "email", "someone@example.com", other_offset, CLEAN_TAGS
    )
    write_injection_review([flag, other_flag], path)

    entries_after = json.loads(path.read_text(encoding="utf-8"))
    by_id = {e["item_id"]: e for e in entries_after}
    assert len(entries_after) == 2
    assert by_id["item-1"]["reviewed"] is False
    assert by_id["item-2"]["reviewed"] is None


def test_write_distractor_review_preserves_reviewed_entry(tmp_path):
    path = tmp_path / "distractor_review.json"
    flag = check_distractor_tell(
        "item-1", "budget", "1950", ["around 1800", "about 1900", "maybe 2000", "1950"]
    )
    write_distractor_review([flag], path)

    entries = json.loads(path.read_text(encoding="utf-8"))
    entries[0]["reviewed"] = True
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    write_distractor_review([flag], path)
    entries_after = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries_after) == 1
    assert entries_after[0]["reviewed"] is True
