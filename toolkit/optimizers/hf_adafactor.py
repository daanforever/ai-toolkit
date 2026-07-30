"""Stock HuggingFace Adafactor (transformers.optimization.Adafactor).

Use via ``get_optimizer(..., optimizer_type="hfadafactor")``.

LR contract (wired in ``toolkit.optimizer.get_optimizer``):
  - ``train.lr: 0`` or ``null`` → ``lr=None`` (paper relative schedule).
  - nonzero ``train.lr`` → float lr; ``relative_step`` defaults to False
    (HF forbids ``lr is not None`` with ``relative_step=True``).

For toolkit extras (stochastic rounding, fixed beta2, metrics), use local
``adafactor`` instead.
"""

from transformers.optimization import Adafactor as _HFAdafactor


class HFAdafactor(_HFAdafactor):
    """HuggingFace Adafactor with stock HF defaults (relative_step / scale_parameter True)."""

    pass
