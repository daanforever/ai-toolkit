"""Health (TensorBoard) and visual (CLIP/LPIPS) rubric for LoRA LR probes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_DEFAULT_CLIP_MODEL = "ViT-B-32"
_DEFAULT_CLIP_PRETRAINED = "laion2b_s34b_b79k"
_DEFAULT_LPIPS_DEAD = 0.04
_DEFAULT_LPIPS_BOOM = 0.45

# Lazy CLIP / LPIPS singletons (CPU). Tests monkeypatch the _metric_* helpers.
_clip_bundle: tuple[Any, Any, Any] | None = None  # model, preprocess, tokenizer
_lpips_model: Any | None = None
_preferred_clip_model: str = _DEFAULT_CLIP_MODEL
_preferred_clip_pretrained: str = _DEFAULT_CLIP_PRETRAINED


@dataclass
class HealthResult:
    ok: bool
    reason: str | None
    tags_seen: list[str] = field(default_factory=list)
    last_instability: float | None = None
    loss_tag: str | None = None
    loss_first_mean: float | None = None
    loss_last_mean: float | None = None
    last_lr: float | None = None
    last_effective_lr: float | None = None


@dataclass
class VisualResult:
    drop: bool
    reason: str | None
    score: float
    clip_i_s_r: float | None = None
    clip_i_s_m: float | None = None
    clip_i_m_r: float | None = None
    clip_t: float | None = None
    lpips_s_r: float | None = None
    lpips_s_m: float | None = None
    lpips_m_r: float | None = None


def _find_event_dir(log_dir: Path) -> Path | None:
    """Return directory containing the newest events file under log_dir."""
    if not log_dir.is_dir():
        return None
    event_files = list(log_dir.rglob("events.out.tfevents*"))
    if not event_files:
        return None
    latest = max(event_files, key=lambda p: p.stat().st_mtime)
    return latest.parent


def _scalar_series(ea: Any, tag: str) -> list[tuple[int, float]]:
    try:
        events = ea.Scalars(tag)
    except KeyError:
        return []
    return [(int(e.step), float(e.value)) for e in events]


def _last_scalar(ea: Any, tag: str) -> float | None:
    series = _scalar_series(ea, tag)
    if not series:
        return None
    return series[-1][1]


# Aux loss tags: checked for NaN/Inf but never used as primary for 8× ratio.
_AUX_LOSS_TAGS = frozenset({"loss/turbo_teacher"})


def _pick_primary_loss_tag(tags: Sequence[str]) -> str | None:
    """
    Prefer a primary training loss under loss/.

    Choice order: loss/loss, any loss/* ending with '/loss', first loss/* sorted,
    then bare 'loss' (BaseSDTrainProcess writes add_scalar('loss', ...)).
    Aux tags (e.g. loss/turbo_teacher) are skipped for the ratio check.
    """
    tag_set = set(tags)
    if "loss/loss" in tag_set:
        return "loss/loss"
    loss_star = sorted(
        t for t in tags if t.startswith("loss/") and t not in _AUX_LOSS_TAGS
    )
    for t in loss_star:
        if t.endswith("/loss"):
            return t
    if loss_star:
        return loss_star[0]
    if "loss" in tag_set:
        return "loss"
    return None


def _loss_tags_for_nan_check(tags: Sequence[str]) -> list[str]:
    out = [t for t in tags if t.startswith("loss/")]
    if "loss" in tags and "loss" not in out:
        out.append("loss")
    return out


def _window_means(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    k = max(1, int(n * 0.2))
    first = sum(values[:k]) / k
    last = sum(values[-k:]) / k
    return first, last


def health_from_tb(
    log_dir: Path,
    *,
    warmup_steps: int,
    instability_max: float,
) -> HealthResult:
    log_dir = Path(log_dir)
    event_dir = _find_event_dir(log_dir)
    if event_dir is None:
        return HealthResult(ok=False, reason="no_tb")

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return HealthResult(ok=False, reason="no_tb")

    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    tags = list(ea.Tags().get("scalars", []) or [])
    if not tags:
        return HealthResult(ok=False, reason="no_tb", tags_seen=tags)

    last_lr = _last_scalar(ea, "lr")
    last_effective_lr = _last_scalar(ea, "train/effective_lr")
    last_instability = _last_scalar(ea, "train/instability_score")
    loss_tag = _pick_primary_loss_tag(tags)

    base_kw: dict[str, Any] = {
        "tags_seen": tags,
        "last_instability": last_instability,
        "loss_tag": loss_tag,
        "last_lr": last_lr,
        "last_effective_lr": last_effective_lr,
    }

    # NaN/Inf in loss/*, train/grad_rms, train/update_rms
    nan_tags = _loss_tags_for_nan_check(tags) + [
        t for t in ("train/grad_rms", "train/update_rms") if t in tags
    ]
    for tag in nan_tags:
        for _step, value in _scalar_series(ea, tag):
            if not math.isfinite(value):
                return HealthResult(
                    ok=False,
                    reason=f"nan_inf:{tag}",
                    **base_kw,
                )

    # Post-warmup loss ratio (skip if no loss tag or no post-warmup points)
    if loss_tag is not None:
        post = [
            v for step, v in _scalar_series(ea, loss_tag) if step >= warmup_steps
        ]
        if post:
            loss_first_mean, loss_last_mean = _window_means(post)
            base_kw["loss_first_mean"] = loss_first_mean
            base_kw["loss_last_mean"] = loss_last_mean
            if loss_last_mean > 8.0 * loss_first_mean:
                return HealthResult(ok=False, reason="loss_ratio", **base_kw)

    # Instability (missing tag is not a fail)
    if last_instability is not None and last_instability > instability_max:
        return HealthResult(ok=False, reason="instability", **base_kw)

    return HealthResult(ok=True, reason=None, **base_kw)


def _get_clip_bundle(
    model_name: str | None = None,
    pretrained: str | None = None,
) -> tuple[Any, Any, Any]:
    global _clip_bundle
    if _clip_bundle is None:
        import open_clip

        name = model_name or _preferred_clip_model
        weights = pretrained or _preferred_clip_pretrained
        model, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained=weights
        )
        tokenizer = open_clip.get_tokenizer(name)
        model.eval()
        model.to("cpu")
        _clip_bundle = (model, preprocess, tokenizer)
    return _clip_bundle


def _get_lpips_model() -> Any:
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex")
        _lpips_model.eval()
        _lpips_model.to("cpu")
    return _lpips_model


def _load_rgb(path: Path | str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _embed_image(path: Path | str):
    import torch

    model, preprocess, _ = _get_clip_bundle()
    img = preprocess(_load_rgb(path)).unsqueeze(0)
    with torch.no_grad():
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()


def _embed_text(text: str):
    import torch

    model, _, tokenizer = _get_clip_bundle()
    tokens = tokenizer([text])
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()


def _cosine(a, b) -> float:
    return float((a * b).sum().item())


def _metric_clip_i(img_a: Path | str, img_b: Path | str) -> float:
    return _cosine(_embed_image(img_a), _embed_image(img_b))


def _metric_clip_t(img: Path | str, text: str) -> float:
    return _cosine(_embed_image(img), _embed_text(text))


def _metric_lpips(img_a: Path | str, img_b: Path | str) -> float:
    import numpy as np
    import torch

    def _prep(p: Path | str):
        img = _load_rgb(p).resize((256, 256))
        arr = np.asarray(img).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW in [0, 1]
        return (t * 2.0 - 1.0).unsqueeze(0)

    model = _get_lpips_model()
    with torch.no_grad():
        d = model(_prep(img_a), _prep(img_b))
    return float(d.squeeze().item())


def _metric_clip_i_vs_dataset(
    img: Path | str, dataset_images: Sequence[Path | str]
) -> float:
    import torch

    embeds = [_embed_image(p) for p in dataset_images]
    if not embeds:
        raise ValueError("dataset_images is empty")
    mean_emb = torch.stack(embeds, dim=0).mean(dim=0)
    mean_emb = mean_emb / mean_emb.norm().clamp(min=1e-12)
    return _cosine(_embed_image(img), mean_emb)


def visual_score(
    *,
    sample,
    reference,
    master,
    caption,
    stage_id,
    dataset_images,
    prompt,
    thresholds,
) -> VisualResult:
    global _preferred_clip_model, _preferred_clip_pretrained

    thr: Mapping[str, Any] = thresholds or {}
    lpips_dead = float(thr.get("lpips_dead", _DEFAULT_LPIPS_DEAD))
    lpips_boom = float(thr.get("lpips_boom", _DEFAULT_LPIPS_BOOM))
    _preferred_clip_model = str(thr.get("clip_model", _DEFAULT_CLIP_MODEL))
    _preferred_clip_pretrained = str(
        thr.get("clip_pretrained", _DEFAULT_CLIP_PRETRAINED)
    )

    stage = str(stage_id).lower()
    is_stage_a = stage == "a"

    lpips_s_m = _metric_lpips(sample, master)
    clip_i_s_m = _metric_clip_i(sample, master)

    if is_stage_a:
        clip_i_s_r = _metric_clip_i(sample, reference)
        clip_i_m_r = _metric_clip_i(master, reference)
        clip_t = _metric_clip_t(sample, caption)
        lpips_s_r = _metric_lpips(sample, reference)
        lpips_m_r = _metric_lpips(master, reference)
    else:
        clip_i_s_r = _metric_clip_i_vs_dataset(sample, dataset_images)
        clip_i_m_r = _metric_clip_i_vs_dataset(master, dataset_images)
        clip_t = _metric_clip_t(sample, prompt)
        lpips_s_r = None
        lpips_m_r = None

    # Gates (dataset-mean plays role of R for B/C)
    drop = False
    reason: str | None = None
    if lpips_s_m < lpips_dead and clip_i_s_r <= clip_i_m_r + 0.01:
        drop = True
        reason = "dead"
    elif lpips_s_m > lpips_boom and clip_i_s_r < clip_i_m_r:
        drop = True
        reason = "exploded"

    if is_stage_a:
        score = (
            0.45 * clip_i_s_r
            + 0.20 * clip_t
            + 0.25 * max(0.0, clip_i_s_r - clip_i_m_r)
            + 0.10 * max(0.0, (lpips_m_r or 0.0) - (lpips_s_r or 0.0))
        )
    else:
        score = 0.6 * clip_i_s_r + 0.4 * clip_t

    return VisualResult(
        drop=drop,
        reason=reason,
        score=float(score),
        clip_i_s_r=clip_i_s_r,
        clip_i_s_m=clip_i_s_m,
        clip_i_m_r=clip_i_m_r,
        clip_t=clip_t,
        lpips_s_r=lpips_s_r,
        lpips_s_m=lpips_s_m,
        lpips_m_r=lpips_m_r,
    )
