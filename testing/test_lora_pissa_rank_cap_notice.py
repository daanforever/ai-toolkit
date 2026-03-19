"""PiSSA rank cap: one UserWarning per network, not per layer."""

import warnings

from toolkit.lora_utils.pissa import emit_pissa_rank_cap_notice_once


class _DummyLoRANet:
    pass


def test_pissa_rank_cap_warning_emitted_once_per_network():
    net = _DummyLoRANet()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        emit_pissa_rank_cap_notice_once(net)
        emit_pissa_rank_cap_notice_once(net)
    pissa = [w for w in rec if "PiSSA" in str(w.message)]
    assert len(pissa) == 1
