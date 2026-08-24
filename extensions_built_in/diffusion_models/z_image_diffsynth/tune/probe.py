"""Run one LoRA probe job in a fresh Python subprocess."""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

_ENV_FLAG = "ZIMAGE_TUNE_PROBE_SUBPROCESS"
_ENV_CONFIG = "ZIMAGE_TUNE_PROBE_CONFIG"
_ENV_STEP_TIMEOUT = "ZIMAGE_TUNE_STEP_TIMEOUT_S"
_SCRATCH_YAML = "_probe_job.yaml"
_STEP_S_RE = re.compile(r"^tune: step=(\d+) step_s=([0-9.]+)\s*$", re.MULTILINE)


@dataclass
class ProbeResult:
    exit_code: int
    training_folder: str
    log_dir: str
    save_root: str
    stdout: str
    stderr: str


def _repo_root() -> Path:
    # tune/probe.py → z_image_diffsynth → diffusion_models → extensions_built_in → repo
    return Path(__file__).resolve().parents[4]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} must be a mapping")
    return data


def _write_yaml(config: dict, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return dest.resolve()


def _meta(config: dict) -> tuple[str, str, str]:
    cfg = config["config"]
    name = str(cfg["name"])
    process0 = cfg["process"][0]
    training_folder = str(process0["training_folder"])
    log_dir = str(process0["log_dir"])
    return training_folder, log_dir, name


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True", "yes")


def median_step_s(log: str) -> float | None:
    """Median per-step seconds from `tune: step=N step_s=X.XXX` lines."""
    xs = [float(m.group(2)) for m in _STEP_S_RE.finditer(log)]
    if not xs:
        return None
    return float(statistics.median(xs))


def child_wall_timeout_s(
    load_budget_s: float,
    n_new_steps: int,
    step_timeout_s: float,
    sample_budget_s: float,
) -> float:
    return (
        float(load_budget_s)
        + int(n_new_steps) * float(step_timeout_s)
        + float(sample_budget_s)
    )


def step_timeout_exceeded(
    dt: float,
    *,
    is_first: bool,
    is_sample_or_save: bool,
    is_last: bool,
    limit_s: float,
) -> bool:
    if float(limit_s) <= 0:
        return False
    if is_first or is_sample_or_save or is_last:
        return False
    return float(dt) > float(limit_s)


def _venv_python() -> Path:
    root = _repo_root()
    if os.name == "nt":
        return root / "venv" / "Scripts" / "python.exe"
    return root / "venv" / "bin" / "python"


def is_venv_python(executable: str | None = None) -> bool:
    exe = Path(executable or sys.executable)
    venv = _venv_python()
    try:
        return os.path.normcase(os.path.realpath(str(exe))) == os.path.normcase(
            os.path.realpath(str(venv))
        )
    except OSError:
        return False


def cuda_gate_reason(
    *,
    device: str,
    cuda_available: bool,
    is_venv: bool,
) -> str | None:
    if not is_venv:
        return "not_venv"
    if str(device).strip().lower() != "cuda":
        return "not_cuda_device"
    if not cuda_available:
        return "cuda_unavailable"
    return None


def _step_timeout_s_from_env() -> float:
    raw = os.environ.get(_ENV_STEP_TIMEOUT, "2.0").strip()
    try:
        return float(raw)
    except ValueError:
        return 2.0


def _is_sample_or_save_step(proc) -> bool:
    completed = int(proc.step_num) - 1
    start = int(getattr(proc, "start_step", 0))
    if completed == start:
        return False
    save_every = getattr(getattr(proc, "save_config", None), "save_every", None)
    sample_every = getattr(getattr(proc, "sample_config", None), "sample_every", None)
    is_save = bool(save_every) and completed % int(save_every) == 0
    is_sample = bool(sample_every) and completed % int(sample_every) == 0
    disable = bool(
        getattr(getattr(proc, "train_config", None), "disable_sampling", False)
    )
    if disable:
        is_sample = False
    return bool(is_save or is_sample)


def _is_last_step(proc) -> bool:
    steps = getattr(getattr(proc, "train_config", None), "steps", None)
    if steps is None:
        return False
    return int(proc.step_num) >= int(steps)


def cuda_placement_reason(
    *,
    param_device_type: str,
    payload_device_type: str | None,
    mem_gb: float,
    min_mem_gb: float = 1.0,
) -> str | None:
    """Fail closed if frozen DiT (or its quantized payload) is not on CUDA at GB-scale."""
    if str(param_device_type) != "cuda":
        return "not_cuda_device"
    if payload_device_type is not None and str(payload_device_type) != "cuda":
        return "not_cuda_device"
    if float(mem_gb) < float(min_mem_gb):
        return "not_cuda_device"
    return None


def _payload_device_type(param) -> str | None:
    qdata = getattr(param, "qdata", None)
    if qdata is not None:
        return qdata.device.type
    data = getattr(param, "_data", None)
    if data is not None:
        return data.device.type
    return None


def _ensure_cuda_params(proc) -> None:
    import torch

    sd = getattr(proc, "sd", None)
    dit = getattr(sd, "_raw_dit", None) if sd is not None else None
    unet = getattr(sd, "unet", None) if sd is not None else None
    module = dit if dit is not None else unet
    if module is None:
        return
    first_frozen = getattr(sd, "_first_frozen_base_param", None)
    param = first_frozen(module) if callable(first_frozen) else None
    if param is None:
        try:
            param = next(
                p
                for n, p in module.named_parameters()
                if "lora" not in n.lower() and not p.requires_grad
            )
        except StopIteration:
            try:
                param = next(module.parameters())
            except StopIteration:
                return
    mem_gb = float(torch.cuda.memory_allocated()) / (1024 ** 3)
    payload_type = _payload_device_type(param)
    for n, p in module.named_parameters():
        if "lora" in n.lower() or p.requires_grad:
            continue
        pt = _payload_device_type(p)
        if pt is None:
            continue
        payload_type = pt
        if pt != "cuda":
            break
    print(
        f"\ntune: cuda_mem_gb={mem_gb:.2f} gpu={torch.cuda.get_device_name(0)} "
        f"param_device={param.device} payload_device={payload_type}",
        flush=True,
    )
    reason = cuda_placement_reason(
        param_device_type=param.device.type,
        payload_device_type=payload_type,
        mem_gb=mem_gb,
    )
    if reason:
        print(f"tune: {reason}", flush=True)
        sys.exit(1)


def _enforce_runtime_gates(job_path: str) -> None:
    cfg = _load_yaml(Path(job_path))
    device = str(cfg["config"]["process"][0].get("device", ""))
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        torch = None
        cuda_ok = False
    reason = cuda_gate_reason(
        device=device,
        cuda_available=cuda_ok,
        is_venv=is_venv_python(),
    )
    if reason:
        print(f"tune: {reason}", flush=True)
        sys.exit(1)
    assert torch is not None
    print(
        f"tune: gpu={torch.cuda.get_device_name(0)} torch_cuda={torch.version.cuda}",
        flush=True,
    )
    try:
        import bitsandbytes as bnb

        print(f"tune: bitsandbytes={getattr(bnb, '__file__', '?')}", flush=True)
    except Exception as exc:
        print(f"tune: bitsandbytes_missing {exc}", flush=True)


def _install_step_timer() -> None:
    """Print wall time between training steps; kill if a train step exceeds limit."""
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

    orig = BaseSDTrainProcess.end_step_hook
    if getattr(orig, "_zimage_tune_step_timer", False):
        return

    limit_s = _step_timeout_s_from_env()

    def timed(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        now = time.perf_counter()
        last = getattr(self, "_zimage_tune_step_last", None)
        self._zimage_tune_step_last = now
        if last is None:
            _ensure_cuda_params(self)
            return
        dt = now - last
        print(f"\ntune: step={int(self.step_num)} step_s={dt:.3f}", flush=True)
        if step_timeout_exceeded(
            dt,
            is_first=False,
            is_sample_or_save=_is_sample_or_save_step(self),
            is_last=_is_last_step(self),
            limit_s=limit_s,
        ):
            print(
                f"\ntune: step_timeout step={int(self.step_num)} "
                f"step_s={dt:.3f} limit={limit_s:.3f}",
                flush=True,
            )
            sys.exit(1)

    timed._zimage_tune_step_timer = True  # type: ignore[attr-defined]
    BaseSDTrainProcess.end_step_hook = timed  # type: ignore[method-assign]


def _run_child(
    cmd: list[str],
    env: dict[str, str],
    cwd: str,
    timeout_s: float | None = None,
) -> tuple[int, str]:
    """Run child, tee stdout+stderr to this process, return (exit_code, captured)."""
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    chunks: list[str] = []

    def _read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            chunks.append(line)

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        msg = "tune: child_wall_timeout\n"
        print(msg, end="", flush=True)
        chunks.append(msg)
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    reader.join(timeout=5)
    if timed_out:
        return 1, "".join(chunks)
    code = proc.poll()
    return int(code if code is not None else 1), "".join(chunks)


def _probe_subprocess_worker() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    _enforce_runtime_gates(os.environ[_ENV_CONFIG])
    _install_step_timer()
    from toolkit.job import run_job

    run_job(os.environ[_ENV_CONFIG])


def run_probe(
    config: dict | Path,
    *,
    python_exe: Path | None = None,
    timeout_s: float | None = None,
    step_timeout_s: float | None = None,
) -> ProbeResult:
    if isinstance(config, Path):
        job_config_path = config.resolve()
        cfg = _load_yaml(job_config_path)
    else:
        cfg = config
        training_folder, _, _ = _meta(cfg)
        job_config_path = _write_yaml(cfg, Path(training_folder) / _SCRATCH_YAML)

    training_folder, log_dir, name = _meta(cfg)
    save_root = os.path.join(training_folder, name)

    exe = str(python_exe) if python_exe is not None else sys.executable
    env = os.environ.copy()
    env.pop(_ENV_FLAG, None)
    env[_ENV_FLAG] = "1"
    env[_ENV_CONFIG] = str(job_config_path)
    env["PYTHONUNBUFFERED"] = "1"
    if step_timeout_s is not None:
        env[_ENV_STEP_TIMEOUT] = str(step_timeout_s)

    # File-path worker: parent package __init__ imports DiT; avoid -m package load.
    code, captured = _run_child(
        [exe, "-u", str(Path(__file__).resolve())],
        env,
        str(_repo_root()),
        timeout_s,
    )
    return ProbeResult(
        exit_code=code,
        training_folder=training_folder,
        log_dir=log_dir,
        save_root=save_root,
        stdout=captured,
        stderr="",
    )


if __name__ == "__main__":
    if _env_flag(_ENV_FLAG):
        _probe_subprocess_worker()
    else:
        sys.exit(2)
