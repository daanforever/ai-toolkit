import random
import re

import pytest
import yaml

from toolkit.config_modules import DatasetConfig
from toolkit.dataloader_mixins import (
    _get_unique_caption_permutations,
    _join_caption_segments,
    _keep_caption_segments,
    _shuffle_caption_by_commas,
    _split_caption_segments,
)

_LEGACY_SPLIT_RE = re.compile('|'.join(re.escape(s) for s in ['.', ',', ';']))
_LEGACY_JOIN = ', '


def test_split_caption_segments_default_period_space():
    assert _split_caption_segments("a, b; c. d") == ["a, b; c", "d"]
    assert _join_caption_segments(["a, b; c", "d"]) == "a, b; c. d"


def test_split_caption_segments_no_space_after_period():
    assert _split_caption_segments("Hello.World.Foo") == ["Hello.World.Foo"]


def test_split_caption_segments_trailing_period():
    assert _split_caption_segments("A. B.") == ["A", "B."]


def test_split_caption_segments_legacy_delimiters():
    assert _split_caption_segments("a, b; c. d", split_re=_LEGACY_SPLIT_RE) == ["a", "b", "c", "d"]


def test_join_caption_segments_default_and_custom():
    assert _join_caption_segments(["a", "b", "c"]) == "a. b. c"
    assert _join_caption_segments(["a", "b", "c"], join_str=_LEGACY_JOIN) == "a, b, c"


def test_keep_caption_segments_comma_caption_is_one_segment():
    caption = "sks woman, beach, sunset"
    assert _keep_caption_segments(caption, 0) == ""
    assert _keep_caption_segments(caption, 1) == caption
    assert _keep_caption_segments(caption, 2) == caption
    assert _keep_caption_segments(caption, 10) == caption


def test_keep_caption_segments_period_space():
    caption = "first; second. third"
    assert _keep_caption_segments(caption, 1) == "first; second"
    assert _keep_caption_segments(caption, 2) == "first; second. third"


def test_keep_caption_segments_legacy():
    assert _keep_caption_segments(
        "first; second. third", 1, split_re=_LEGACY_SPLIT_RE, join_str=_LEGACY_JOIN,
    ) == "first"
    assert _keep_caption_segments(
        "first; second. third", 2, split_re=_LEGACY_SPLIT_RE, join_str=_LEGACY_JOIN,
    ) == "first, second"


def test_shuffle_caption_by_commas_keeps_first_segment():
    random.seed(42)
    result = _shuffle_caption_by_commas("fixed. b. c", keep_n=1)
    assert result.startswith("fixed. ")
    segments = _split_caption_segments(result)
    assert segments[0] == "fixed"
    assert sorted(segments[1:]) == ["b", "c"]


def test_shuffle_caption_by_commas_deterministic_with_seed():
    caption = "a. b. c. d"
    random.seed(0)
    first = _shuffle_caption_by_commas(caption, keep_n=1)
    random.seed(0)
    second = _shuffle_caption_by_commas(caption, keep_n=1)
    assert first == second
    assert first.startswith("a. ")
    assert sorted(_split_caption_segments(first)) == ["a", "b", "c", "d"]


def test_shuffle_comma_only_caption_is_noop():
    caption = "a, b, c, d"
    assert _split_caption_segments(caption) == [caption]
    assert _shuffle_caption_by_commas(caption, keep_n=1) == caption


def test_get_unique_caption_permutations_fixed_prefix():
    caption = "keep. x. y"
    perms = _get_unique_caption_permutations(caption, max_permutations=6, keep_n=1)
    assert perms[0] == "keep. x. y"
    for p in perms:
        assert p.startswith("keep. ")
        assert sorted(_split_caption_segments(p)) == ["keep", "x", "y"]


def test_dataset_config_defaults():
    cfg = DatasetConfig()
    assert cfg.shuffle_tokens_split == ['. ']
    assert cfg.shuffle_tokens_join == '. '
    assert cfg.shuffle_tokens_split_re.pattern == r'\.\ '
    assert _split_caption_segments("a, b. c", split_re=cfg.shuffle_tokens_split_re) == ["a, b", "c"]


def test_dataset_config_from_yaml():
    raw = yaml.safe_load(
        "shuffle_tokens_split: ['. ']\n"
        "shuffle_tokens_join: '. '\n"
    )
    cfg = DatasetConfig(**raw)
    assert cfg.shuffle_tokens_split == ['. ']
    assert cfg.shuffle_tokens_join == '. '


def test_dataset_config_split_string_equals_list():
    cfg_str = DatasetConfig(shuffle_tokens_split='. ')
    cfg_list = DatasetConfig(shuffle_tokens_split=['. '])
    assert cfg_str.shuffle_tokens_split == cfg_list.shuffle_tokens_split
    assert cfg_str.shuffle_tokens_split_re.pattern == cfg_list.shuffle_tokens_split_re.pattern


def test_dataset_config_empty_split_raises():
    with pytest.raises(ValueError):
        DatasetConfig(shuffle_tokens_split=[])
    with pytest.raises(ValueError):
        DatasetConfig(shuffle_tokens_split='')


def test_dataset_config_legacy_split_join():
    cfg = DatasetConfig(
        shuffle_tokens_split=['.', ',', ';'],
        shuffle_tokens_join=', ',
    )
    segs = _split_caption_segments("a, b; c. d", split_re=cfg.shuffle_tokens_split_re)
    assert segs == ["a", "b", "c", "d"]
    assert _join_caption_segments(segs, join_str=cfg.shuffle_tokens_join) == "a, b, c, d"
