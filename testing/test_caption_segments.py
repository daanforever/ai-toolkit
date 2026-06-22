import random

from toolkit.dataloader_mixins import (
    _get_unique_caption_permutations,
    _keep_caption_segments,
    _shuffle_caption_by_commas,
    _split_caption_segments,
)


def test_split_caption_segments_mixed_delimiters():
    assert _split_caption_segments("a, b; c. d") == ["a", "b", "c", "d"]


def test_keep_caption_segments():
    caption = "sks woman, beach, sunset"
    assert _keep_caption_segments(caption, 0) == ""
    assert _keep_caption_segments(caption, 1) == "sks woman"
    assert _keep_caption_segments(caption, 2) == "sks woman, beach"
    assert _keep_caption_segments(caption, 10) == "sks woman, beach, sunset"


def test_keep_caption_segments_semicolon_period():
    assert _keep_caption_segments("first; second. third", 1) == "first"
    assert _keep_caption_segments("first; second. third", 2) == "first, second"


def test_shuffle_caption_by_commas_keeps_first_segment():
    random.seed(42)
    result = _shuffle_caption_by_commas("fixed; b. c", keep_n=1)
    assert result.startswith("fixed, ")
    segments = _split_caption_segments(result)
    assert segments[0] == "fixed"
    assert sorted(segments[1:]) == ["b", "c"]


def test_shuffle_caption_by_commas_deterministic_with_seed():
    caption = "a, b, c, d"
    random.seed(0)
    first = _shuffle_caption_by_commas(caption, keep_n=1)
    random.seed(0)
    second = _shuffle_caption_by_commas(caption, keep_n=1)
    assert first == second
    assert first.startswith("a, ")


def test_get_unique_caption_permutations_fixed_prefix():
    caption = "keep; x. y"
    perms = _get_unique_caption_permutations(caption, max_permutations=6, keep_n=1)
    assert perms[0] == "keep, x, y"
    for p in perms:
        assert p.startswith("keep, ")
