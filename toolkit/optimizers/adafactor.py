import math
from typing import Iterable, List, Optional, Tuple
import torch
from toolkit.optimizers.optimizer_utils import copy_stochastic, stochastic_grad_accummulation
from toolkit.print import print_acc
from toolkit.util.debug import is_debug_enabled
from optimum.quanto import QBytesTensor
import random
from toolkit.lora_utils.stagnation_detector import StagnationDetector


class Adafactor(torch.optim.Optimizer):
    """
    Adafactor implementation with stochastic rounding accumulation and stochastic rounding on apply.
    Modified from transformers Adafactor implementation to support stochastic rounding accumulation and apply.

    AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://arxiv.org/abs/1804.04235 Note that
    this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
    `warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to `1e-4`):
            When `relative_step=True`, acts as maximum learning rate cap (upper bound). When `relative_step=False`, the manual learning rate.
        eps (`Tuple[float, float]`, *optional*, defaults to `(1e-30, 0.001)`):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (`float`, *optional*, defaults to 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (`float`, *optional*, defaults to -0.8):
            Deprecated. Previously used to compute beta2 as `1.0 + decay_rate`.
            Now beta2 is specified directly. Kept for backward compatibility.
        beta2 (`float`, *optional*, defaults to 0.99):
            Coefficient used to compute running averages of square
            (second moment, like in Adam). Suggested values: 0.99 (default), 0.999.
        rms_max_decay_rate (`float`, *optional*, defaults to `0.97`):
            Decay rate for running max of update RMS used in activity normalization.
            Applied each step: ``update_rms_max = max(update_rms_max * rms_max_decay_rate, update_rms)``.
            The same decay is used for group-level gradient RMS running max (``grad_rms_max`` on each
            param group) for consistency.
            When scale_parameter=True and relative_step=True, also used for the group-level running max
            of parameter RMS (``rms_max`` on each param group), which normalizes each parameter's scale
            to (0, 1] for LR (useful for mixed-scale groups e.g. LoRA). Group-level metrics
            (``rms_ema``, ``update_rms``, ``lr_mean``, ``gns``, ``step_efficiency``, ``dynamic_gain``, etc.)
            live on ``param_groups`` rather than in per-parameter ``state``.
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
            (first moment, like in Adam). If not None, enables momentum.
            Suggested values: 0.9 (default), 0.95 or 0.99 for smoother updates.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `False`):
            If True, learning rate is scaled by root mean square.
            Scaling is stronger when update magnitude is large (to protect small parameters).
        relative_step (`bool`, *optional*, defaults to `False`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            When True, the group learning rate `lr` is approached smoothly: one interpolation segment per change of
            `lr`, progress tracked once per group per step (see `_global_lr`). Works with both `relative_step=True` and
            manual mode (`relative_step=False`). If `lr` changes during a segment (e.g. via `set_lr`), a new segment
            starts from the current interpolated level toward the new `lr` (up or down). Runtime toggling of
            `warmup_init` is not supported.
        warmup_steps (`int`, *optional*, defaults to `100`):
            When `warmup_init=True`, number of optimizer steps to interpolate each warmup segment from its start
            toward `lr`. Progress is advanced once per group per `step()` (see `_warmup_update_group`).
        min_lr (`float`, *optional*, defaults to `1e-6`):
            Term in the relative-step learning-rate factor when `relative_step=True`: ``relative`` includes
            ``(1 + min_lr * ratio)`` (see `_get_lr`).
        factored (`bool | None`, *optional*, defaults to `None`):
            If True, use factored second-moment (row/col) for all parameters. If False, use full second-moment.
            If None, auto-detect: use factored for parameters with 2+ dimensions (current default behavior).
        emergency_brake (`float | None`, *optional*, defaults to `None`):
            When set, enables an adaptive "brake" on the internal relative LR and uses this value as the minimum
            floor for both `brake` and `soft_brake`. `None` disables the mechanism.

    This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

    Recommended T5 finetuning settings (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

        - Training without LR warmup or clip_threshold is not recommended.

           - use scheduled LR warm-up to fixed LR
           - use clip_threshold=1.0 (https://arxiv.org/abs/1804.04235)
        - Disable relative updates
        - Use scale_parameter=False
        - Additional optimizer operations like gradient clipping should not be used alongside Adafactor

    Example:

    ```python
    Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    ```

    Others reported the following combination to work well:

    ```python
    Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    ```

    When using `lr=None` with [`Trainer`] you will most likely need to use [`~optimization.AdafactorSchedule`]
    scheduler as following:

    ```python
    from transformers.optimization import Adafactor, AdafactorSchedule

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)
    trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
    ```

    Usage:

    ```python
    # replace AdamW with Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    ```"""

    def __init__(
        self,
        params,
        lr=1e-4,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        beta2=0.99,
        rms_max_decay_rate=0.97,
        weight_decay=0.0,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        min_lr=1e-6,
        warmup_steps: int = 100,
        do_parameter_swapping=False,
        parameter_swapping_factor=0.1,
        stochastic_accumulation=True,
        stochastic_rounding=True,
        factored=None,
        emergency_brake: float | None = None,
        saddle_point_window: int = 100,
        saddle_point_threshold: float = 0.001,
        saddle_point_step: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "beta1": beta1,
            "beta2": beta2,
            "rms_max_decay_rate": rms_max_decay_rate,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
            "warmup_steps": warmup_steps,
            "min_lr": min_lr,
            "factored": factored,
            "emergency_brake": emergency_brake,
            "instability_score": 0.0,  # cumulative instability tracking for soft brake
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            group["scale_parameter"] = scale_parameter
            group["relative_step"] = relative_step
            group["warmup_init"] = warmup_init
            group["warmup_steps"] = warmup_steps
            group["warmup_active"] = False
            group["factored"] = factored
            group["emergency_brake"] = emergency_brake

        # Create stagnation detector for RMS(parameter) based heuristic.
        self._saddle_point_detector = StagnationDetector(
            window_size=saddle_point_window,
            threshold=saddle_point_threshold,
            epsilon=float(eps[0]),
        )
        # Applied to saddle_point_boost each step when stagnant (add) or not (decay toward 1.0).
        self._saddle_point_step = float(saddle_point_step)
        self._saddle_point_boost = 1.0

        # Store config reapplied after load_state_dict (checkpoint param_groups may omit keys).
        self._lr = lr
        self._min_lr = min_lr
        self._eps = eps
        self._clip_threshold = clip_threshold
        self._rms_max_decay_rate = rms_max_decay_rate
        self._weight_decay = weight_decay
        self._scale_parameter = scale_parameter
        self._relative_step = relative_step
        self._warmup_init = warmup_init
        self._warmup_steps = warmup_steps
        self._beta1 = beta1
        self._beta2 = beta2
        self._factored = factored
        self._emergency_brake = emergency_brake
        self.is_stochastic_rounding_accumulation = False

        # setup stochastic grad accum hooks
        if stochastic_accumulation:
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad and param.dtype != torch.float32:
                        self.is_stochastic_rounding_accumulation = True
                        param.register_post_accumulate_grad_hook(
                            stochastic_grad_accummulation
                        )
    
        self.do_parameter_swapping = do_parameter_swapping
        self.parameter_swapping_factor = parameter_swapping_factor
        self._total_parameter_size = 0
        # count total parameters
        for group in self.param_groups:
            for param in group['params']:
                self._total_parameter_size += torch.numel(param)
        # pretty print total parameters with comma separation
        print(f"Total training parameters: {self._total_parameter_size:,}")
        
        # needs to be enabled to count parameters
        if self.do_parameter_swapping:
            self.enable_parameter_swapping(self.parameter_swapping_factor)

    def set_lr(self, value: float) -> None:
        """Update lr at runtime (e.g. from UI)."""
        self._lr = value
        for group in self.param_groups:
            group["lr"] = value
        if is_debug_enabled():
            print_acc(f"Adafactor: applied runtime lr={value}")

    def set_min_lr(self, value: float) -> None:
        """Update min_lr at runtime (e.g. from UI)."""
        self._min_lr = value
        for group in self.param_groups:
            group["min_lr"] = value
        if is_debug_enabled():
            print_acc(f"Adafactor: applied runtime min_lr={value}")

    def set_weight_decay(self, value: float) -> None:
        """Update weight_decay at runtime (e.g. from UI)."""
        for group in self.param_groups:
            group["weight_decay"] = value
        if is_debug_enabled():
            print_acc(f"Adafactor: applied runtime weight_decay={value}")

    def set_emergency_brake(self, value: float | None) -> None:
        """Update emergency_brake at runtime (e.g. from UI). None disables."""
        self._emergency_brake = value
        for group in self.param_groups:
            group["emergency_brake"] = value
        if is_debug_enabled():
            print_acc(f"Adafactor: applied runtime emergency_brake={value}")

    def set_beta1(self, value: float | None) -> None:
        """Update beta1 at runtime (e.g. from UI). None disables momentum."""
        self._beta1 = value
        for group in self.param_groups:
            group["beta1"] = value
        if is_debug_enabled():
            print_acc(f"Adafactor: applied runtime beta1={value}")

    def set_beta2(self, value: float) -> None:
        """Update beta2 at runtime (e.g. from UI)."""
        self._beta2 = value
        for group in self.param_groups:
            group["beta2"] = value
        if is_debug_enabled():
            print_acc(f"Adafactor: applied runtime beta2={value}")

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

        # saddle_point_boost is not checkpointed; always start fresh after load.
        self._saddle_point_boost = 1.0

        # Reapply current run config (checkpoint param_groups may lack keys e.g. clip_threshold).
        for group in self.param_groups:
            group["lr"] = self._lr
            group["min_lr"] = self._min_lr
            group["eps"] = self._eps
            group["clip_threshold"] = self._clip_threshold
            group["rms_max_decay_rate"] = self._rms_max_decay_rate
            group["weight_decay"] = self._weight_decay
            group["scale_parameter"] = self._scale_parameter
            group["relative_step"] = self._relative_step
            group["warmup_init"] = self._warmup_init
            group["warmup_steps"] = self._warmup_steps
            group["beta1"] = self._beta1
            group["beta2"] = self._beta2
            group["factored"] = self._factored
            group["emergency_brake"] = self._emergency_brake
            group["instability_score"] = group.get("instability_score", 0.0)

        self._migrate_optimizer_state_buffers()

    def _migrate_optimizer_state_buffers(self):
        """Add or reset momentum buffers after load when optimizer_params changed."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    continue

                factored, use_first_moment = self._get_options(group, p.shape)
                ref = p.data if p.dtype == torch.float32 else p.data.float()
                device = ref.device
                dtype = ref.dtype

                if use_first_moment:
                    if (
                        "exp_avg" not in state
                        or state["exp_avg"].shape != ref.shape
                    ):
                        state["exp_avg"] = torch.zeros_like(ref)
                else:
                    state.pop("exp_avg", None)

                if factored:
                    row_shape = p.shape[:-1]
                    col_shape = p.shape[:-2] + p.shape[-1:]
                    if (
                        "exp_avg_sq_row" not in state
                        or state["exp_avg_sq_row"].shape != row_shape
                    ):
                        state["exp_avg_sq_row"] = torch.zeros(
                            row_shape, device=device, dtype=dtype
                        )
                    if (
                        "exp_avg_sq_col" not in state
                        or state["exp_avg_sq_col"].shape != col_shape
                    ):
                        state["exp_avg_sq_col"] = torch.zeros(
                            col_shape, device=device, dtype=dtype
                        )
                    state.pop("exp_avg_sq", None)
                else:
                    if (
                        "exp_avg_sq" not in state
                        or state["exp_avg_sq"].shape != ref.shape
                    ):
                        state["exp_avg_sq"] = torch.zeros_like(ref)
                    state.pop("exp_avg_sq_row", None)
                    state.pop("exp_avg_sq_col", None)

    def enable_parameter_swapping(self, parameter_swapping_factor=0.1):
        self.do_parameter_swapping = True
        self.parameter_swapping_factor = parameter_swapping_factor
        # call it an initial time
        self.swap_parameters()
                    
    def swap_parameters(self):
        all_params = []
        # deactivate all parameters
        for group in self.param_groups:
            for param in group['params']:
                param.requires_grad_(False)
                # remove any grad
                param.grad = None
                all_params.append(param)
        # shuffle all parameters
        random.shuffle(all_params)
        
        # keep activating parameters until we are going to go over the target parameters
        target_parameters = max(1, int(self._total_parameter_size * self.parameter_swapping_factor))
        total_parameters = 0
        for param in all_params:
            param.requires_grad_(True)
            total_parameters += torch.numel(param)
            if total_parameters >= target_parameters:
                break

    def set_warmup_steps(self, value: int) -> None:
        """Update warmup_steps at runtime (e.g. from UI)."""
        self._warmup_steps = value
        for group in self.param_groups:
            group["warmup_steps"] = value
        if is_debug_enabled():
            print_acc(f"Adafactor: applied runtime warmup_steps={value}")

    def _global_lr(self) -> None:
        """Once per optimizer step: group-level warmup before any per-parameter _get_lr."""
        groups = self.param_groups
        if not groups or not groups[0].get("warmup_init"):
            return
        for group in groups:
            self._warmup_update_group(group)

    @staticmethod
    def scheduled_lr_changed(new_lr: float, old_lr: float) -> bool:
        """True if scheduled group lr should be treated as changed (``math.isclose`` with fixed tolerances)."""
        return not math.isclose(new_lr, old_lr, rel_tol=1e-7, abs_tol=1e-12)

    @staticmethod
    def stop_warmup(param_group):
        param_group["warmup_active"] = False
        for k in ("warmup_progress", "warmup_delta", "warmup_start", "warmup_lr"):
            param_group.pop(k, None)

        if is_debug_enabled():
            print_acc(f"Adafactor: warmup stopped")

    def _warmup_update_group(self, group) -> None:
        """Advance group warmup by one optimizer step.

        Starts a new segment when ``group["lr"]`` or ``warmup_steps`` changes.
        Segment: linear ramp stored in ``warmup_lr``; start level is prior ``warmup_lr`` if
        present, else ``group["lr"] * eps[1]``. After ``warmup_steps`` updates, calls ``stop_warmup``.
        """
        lr_target = group["lr"]
        lr_target_old = group.get("warmup_target", 0.0)
        warmup_steps = group["warmup_steps"]
        warmup_steps_old = group.get("warmup_steps_old", 0)

        if self.scheduled_lr_changed(lr_target, lr_target_old) or warmup_steps != warmup_steps_old:
            if "warmup_lr_previous" in group:
                lr_start = group["warmup_lr_previous"]
            else:
                lr_start = lr_target * group["eps"][1]

            group["warmup_active"]    = True
            group["warmup_start"]     = lr_start
            group["warmup_target"]    = lr_target
            group["warmup_progress"]  = 0
            group["warmup_delta"]     = (lr_target - lr_start) / warmup_steps
            group["warmup_steps_old"] = warmup_steps

            if is_debug_enabled():
                direction = "up" if lr_target > lr_target_old else "down"
                print_acc(
                    f"Adafactor: base_lr changed ({lr_start:.2e} -> {lr_target:.2e}, {direction}), starting warmup"
                )

        if group.get("warmup_active", False):
            warmup_start    = group["warmup_start"]
            warmup_progress = group["warmup_progress"]
            warmup_delta    = group["warmup_delta"]
            warmup_steps    = group["warmup_steps"]

            group["warmup_lr"] = warmup_start + warmup_progress * warmup_delta
            group["warmup_progress"]  += 1

            if group["warmup_progress"] >= warmup_steps:
                self.stop_warmup(group)

        group["warmup_lr_previous"] = group.get("warmup_lr", group["lr"])

    def _get_lr(self, param_group, param_state):
        """
        Compute per-parameter learning rate.

        Manual mode (relative_step=False):
          Group lr before scale/relative may be warmup_lr when warmup_init=True; see _global_lr.
          If scale_parameter=True, multiplies by max(eps1, param_rms).

        Adaptive mode (relative_step=True):
          Same group-lr handling; additional relative LR factors from gradients and group statistics.

        Returns:
            float: learning rate for this parameter
        """
        # Extract LR config parameters
        if "warmup_lr" in param_group:
            base_lr = param_group["warmup_lr"]
        else:
            base_lr = param_group["lr"]

        min_lr     = param_group["min_lr"]          # Minimum learning rate
        eps0       = param_group["eps"][0]          # Small constant for numerical stability (division guard)
        eps1       = param_group["eps"][1]          # Parameter scale regularization constant
        param_rms  = param_state["RMS"].item()      # Current parameter RMS magnitude
        scale      = 1.0                            # Default scale for LR
        relative   = 1.0                            # Default relative for LR

        if param_group["scale_parameter"]:
            # Scale LR by parameter magnitude for better adaptation to parameter scale
            scale = max(eps1, param_rms)

        if param_group["relative_step"]:
            # Adaptive LR mode: compute LR from gradient and parameter statistics
            grad_rms      = param_state["grad_rms"].item()        # Current gradient RMS
            # Running max of parameter RMS over the group (decayed each step, then max with each p).
            group_param_rms_max = param_group.get("rms_max", torch.tensor(eps1)).item()

            brake = 1.0
            soft_brake = 1.0

            emergency_brake = param_group.get("emergency_brake", None)
            if param_group["scale_parameter"] and emergency_brake is not None:
                emergency_brake = float(emergency_brake)
                # Instant Brake: multiplicative factor based on current directional consistency
                # Prefer fresh per-parameter dir_consistency; fallback to group mean (when beta1=None)
                dc = param_state.get("dir_consistency")
                if dc is not None:
                    dir_val = dc.item() if isinstance(dc, torch.Tensor) else float(dc)
                else:
                    dir_val = param_group.get("dir_consistency_mean") or 0.0

                brake = max(emergency_brake, min(1 + dir_val, 1.0))

                # Soft Brake: exponential damping based on cumulative instability
                # exp(-score) smoothly reduces LR: score=0 → 1.0, score=2 → 0.135, score=4 → 0.018
                instability_score = param_group.get("instability_score") or 0.0
                soft_brake = math.exp(-instability_score)
                soft_brake = max(emergency_brake, soft_brake)

            # Ratio of parameter RMS to group RMS max
            ratio = max(
                eps0,
                (group_param_rms_max - param_rms) / (group_param_rms_max + eps0),
            )

            saddle_point_mult = max(1.0, float(self._saddle_point_boost))
            relative = (1 + min_lr * ratio) * brake * soft_brake * saddle_point_mult

        new_lr = base_lr * scale * relative

        # Update-to-Weight Ratio safeguard: prevent updates > 10% of parameter magnitude
        if param_group.get("emergency_brake", None) is not None:
            if param_rms > eps1:
                # Cap LR to limit update magnitude relative to parameter scale
                # Ratio > 0.1 (10% per step) typically indicates overshoot
                max_allowed_lr = max(base_lr/10, param_rms * 0.1)
                if new_lr > max_allowed_lr:
                    new_lr = max_allowed_lr

        return new_lr

    def _update_beta1_from_dynamic_gain(self, group, global_mean_dynamic_gain: Optional[float]) -> None:
        """
        Update beta1 for a group based on global mean dynamic gain across all parameter groups.
        Scaling factor is normalized by number of groups to prevent excessive cumulative updates
        when multiple parameter groups are present.
        """
        if global_mean_dynamic_gain is None:
            return
        limited = max(0.0, min(2.0, global_mean_dynamic_gain))
        delta = 0.01 * (limited - 1.0)
        group["beta1"] += delta
        group["beta1"] = max(0.1, min(0.99, group["beta1"]))

    def _update_beta2_from_gns(self, group) -> None:
        """
        Softly adjust group["beta2"] toward a GNS-based target (only when relative_step=True).
        Low GNS (< 4) -> target 0.88; high GNS (> 10) -> target 0.99; else 0.9.
        """
        target_beta2 = 0.9
        current_gns = self._group_scalar_item(group, "gns", 0.0)

        if current_gns < 4.0:
            target_beta2 = 0.888
        elif current_gns > 10.0:
            target_beta2 = 0.999

        group["beta2"] = group["beta2"] + 0.01 * (target_beta2 - group["beta2"])

    @staticmethod
    def _get_options(param_group, param_shape):
        factored_setting = param_group.get("factored", None)
        if factored_setting is None:
            factored = len(param_shape) >= 2
        else:
            factored = factored_setting
        # Enable first moment (exp_avg) if beta1 is set OR emergency_brake is enabled
        use_first_moment = param_group["beta1"] is not None or param_group.get("emergency_brake", None) is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _get_group_scalars(
        self,
        group,
        state_key,
        default=0.0,
        reduction='mean',
        params: Optional[Iterable[torch.nn.Parameter]] = None,
    ):
        """
        Collect per-parameter scalars from state for a group, reduce in tensor space, return float.
        Used for unified metric aggregation (RMS, update_rms, update_rms_max) so get_* use the same
        path. Only params that have state_key in state are included (same as get_update_rms/get_update_rms_max).
        If ``params`` is given, only those parameters are considered (same reduction pattern).
        """
        values = []
        weights = []
        device = None
        param_iter = group["params"] if params is None else params
        for p in param_iter:
            if p not in self.state or state_key not in self.state[p]:
                continue
            val = self.state[p][state_key]
            v_t = torch.as_tensor(val, device=p.device, dtype=torch.float32)
            if device is None:
                device = v_t.device
            values.append(v_t.to(device))
            weights.append(torch.tensor(p.numel(), device=device, dtype=torch.float32))
        if not values:
            return None
        v_stacked = torch.stack(values)
        w_stacked = torch.stack(weights)
        if reduction == 'max':
            return v_stacked.max().item()
        weighted_sum = torch.sum(v_stacked * w_stacked)
        total_weight = torch.sum(w_stacked)
        return (weighted_sum / (total_weight + 1e-12)).item()

    def _get_global_scalar(self, state_key: str) -> Optional[float]:
        """
        Weighted mean over all groups and parameters for a given scalar state_key.
        Weights are parameter sizes (p.numel()) to mirror _get_group_scalars behavior.
        """
        values = []
        weights = []
        device = None
        for group in self.param_groups:
            for p in group["params"]:
                if p not in self.state or state_key not in self.state[p]:
                    continue
                val = self.state[p][state_key]
                v_t = torch.as_tensor(val, device=p.device, dtype=torch.float32)
                if device is None:
                    device = v_t.device
                values.append(v_t.to(device))
                weights.append(torch.tensor(p.numel(), device=device, dtype=torch.float32))
        if not values:
            return None
        v_stacked = torch.stack(values)
        w_stacked = torch.stack(weights)
        weighted_sum = torch.sum(v_stacked * w_stacked)
        total_weight = torch.sum(w_stacked)
        return (weighted_sum / (total_weight + 1e-12)).item()

    def _scalars_per_group_to_avg(self, per_group_list: List[float]) -> float:
        """Unified average over groups for get_avg_*; uses tensor reduction for consistency."""
        if len(per_group_list) == 0:
            return 0.0
        return torch.tensor(per_group_list, dtype=torch.float64).mean().item()

    def _mean_group_rms_ema_for_saddle(self) -> float:
        """Mean of group-level ``rms_ema`` across groups (for stagnation detector)."""
        vals = []
        for g in self.param_groups:
            t = g.get("rms_ema")
            if t is None:
                continue
            vals.append(t.item() if isinstance(t, torch.Tensor) else float(t))
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    @staticmethod
    def _group_scalar_item(group, key: str, default: float = 0.0) -> float:
        t = group.get(key)
        if t is None:
            return default
        return t.item() if isinstance(t, torch.Tensor) else float(t)

    @staticmethod
    def _group_running_max_update(group, key: str, candidate: torch.Tensor) -> None:
        """In-place: group[key] = max(group[key], candidate), with device alignment."""
        if key not in group:
            group[key] = candidate.clone().detach()
            return
        current = group[key]
        if isinstance(current, torch.Tensor) and current.device != candidate.device:
            current = current.to(candidate.device)
            group[key] = current
        group[key] = torch.maximum(current, candidate)

    def _finalize_group_step_metrics(
        self,
        group,
        metrics: List[Tuple[torch.nn.Parameter, torch.Tensor, float, Optional[torch.Tensor]]],
    ) -> None:
        """Aggregate per-param ``state`` (``RMS``, ``grad_rms``) and per-step rows into group scalars."""
        params_list = [p for p, _, _, _ in metrics if self.state.get(p) is not None]
        if not params_list:
            return

        ref_device = params_list[0].device
        total_numel = sum(p.numel() for p in params_list)

        avg_rms = self._get_group_scalars(
            group, "RMS", default=0.0, reduction='mean', params=params_list
        )
        avg_gr = self._get_group_scalars(
            group, "grad_rms", default=0.0, reduction='mean', params=params_list
        )
        if avg_rms is None:
            avg_rms = 0.0
        if avg_gr is None:
            avg_gr = 0.0

        # Weighted mean of ur (same tensor pattern as _get_group_scalars; one .item() at end).
        ur_values = []
        ur_weights = []
        device = None
        for p, ur, _, _ in metrics:
            if self.state.get(p) is None:
                continue
            v_t = torch.as_tensor(ur, device=p.device, dtype=torch.float32).reshape(())
            if device is None:
                device = v_t.device
            ur_values.append(v_t.to(device))
            ur_weights.append(torch.tensor(p.numel(), device=device, dtype=torch.float32))
        if not ur_values:
            return
        ur_stacked = torch.stack(ur_values)
        w_stacked = torch.stack(ur_weights)
        avg_ur = (torch.sum(ur_stacked * w_stacked) / (torch.sum(w_stacked) + 1e-12)).item()

        sum_lr_weighted = sum(
            lr * p.numel() for p, _, lr, _ in metrics if self.state.get(p) is not None
        )

        gns_values = []
        gns_weights = []
        device_g = None
        for p, _, _, gns_t in metrics:
            if gns_t is None or self.state.get(p) is None:
                continue
            v_t = torch.as_tensor(gns_t, device=p.device, dtype=torch.float32).reshape(())
            if device_g is None:
                device_g = v_t.device
            gns_values.append(v_t.to(device_g))
            gns_weights.append(torch.tensor(p.numel(), device=device_g, dtype=torch.float32))

        dr = float(group["rms_max_decay_rate"])
        if "rms_ema" not in group:
            group["rms_ema"] = torch.tensor(avg_rms, dtype=torch.float32, device=ref_device)
        else:
            prev = self._group_scalar_item(group, "rms_ema", 0.0)
            group["rms_ema"] = torch.tensor(
                prev * dr + avg_rms * (1.0 - dr), dtype=torch.float32, device=ref_device
            )
        group["update_rms"] = torch.tensor(avg_ur, dtype=torch.float32, device=ref_device)
        group["lr_mean"] = torch.tensor(
            sum_lr_weighted / total_numel, dtype=torch.float32, device=ref_device
        )
        group["grad_rms"] = torch.tensor(avg_gr, dtype=torch.float32, device=ref_device)
        if gns_values:
            gv = torch.stack(gns_values)
            gw = torch.stack(gns_weights)
            avg_gns = (torch.sum(gv * gw) / (torch.sum(gw) + 1e-12)).item()
            group["gns"] = torch.tensor(avg_gns, dtype=torch.float32, device=ref_device)
        else:
            group["gns"] = torch.tensor(0.0, dtype=torch.float32, device=ref_device)

        eps0 = group["eps"][0] if isinstance(group["eps"], (tuple, list)) else group["eps"]
        eps_t = torch.tensor(float(eps0), dtype=torch.float32, device=ref_device)
        u_rms_t = group["update_rms"]
        u_max = group["update_rms_max"]
        if isinstance(u_max, torch.Tensor):
            u_max = u_max.to(ref_device)
        else:
            u_max = torch.tensor(float(u_max), dtype=torch.float32, device=ref_device)
        g_mean_t = group["grad_rms"]
        group["step_efficiency"] = u_rms_t / (u_max + eps_t)
        group["dynamic_gain"] = u_rms_t / (g_mean_t + eps_t)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-
                    1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    def _update_saddle_point_boost(self, is_stagnant: bool) -> float:
        step = self._saddle_point_step
        b = self._saddle_point_boost
        if is_stagnant:
            b = b + step
        else:
            b = max(1.0, b - step)
        self._saddle_point_boost = b
        return b

    def _detect_saddle_point(self, current_rms: float) -> None:
        """Update saddle_point_boost from RMS(parameter) stagnation (loss-independent)."""
        is_stagnant, _cv = self._saddle_point_detector.check(current_rms)
        self._update_saddle_point_boost(is_stagnant)

    # adafactor manages its own lr
    def get_learning_rates(self):
        """
        One value per group: mean LR over params in group (``lr_mean`` on the param group; 0 before first step).
        """
        out = []
        for group in self.param_groups:
            out.append(self._group_scalar_item(group, "lr_mean", 0.0))
        return out

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.step_hook()
        loss = None
        if closure is not None:
            loss = closure()

        # Detect RMS(parameter) stagnation from previous steps.
        current_rms = self._mean_group_rms_ema_for_saddle()
        self._detect_saddle_point(current_rms)

        self._global_lr()

        for group in self.param_groups:
            decay_rate = group["rms_max_decay_rate"]
            if "rms_max" in group:
                group["rms_max"] = group["rms_max"] * decay_rate
            if "update_rms_max" in group:
                group["update_rms_max"] = group["update_rms_max"] * decay_rate
            if "grad_rms_max" in group:
                group["grad_rms_max"] = group["grad_rms_max"] * decay_rate

            # Pre-compute mean directional consistency once per group for _get_lr
            group["dir_consistency_mean"] = self._get_group_scalars(group, "dir_consistency", default=0.0, reduction='mean')

            # Soft Brake: accumulate instability score when emergency_brake is enabled
            if group.get("emergency_brake", None) is not None:
                dc_mean = group.get("dir_consistency_mean") or 0.0
                score = group.get("instability_score") or 0.0

                if dc_mean < 0:
                    # Accumulate penalty for inconsistent directions (gradient reversals)
                    # Penalty scales with magnitude: stronger for -1.0 (180° reversal) than -0.1
                    score += abs(dc_mean) * 0.1
                else:
                    # Decay penalty when directions are consistent
                    score *= 0.95

                # Clamp score to [0.0, 4.0] to prevent LR from dropping to zero permanently
                group["instability_score"] = min(max(score, 0.0), 4.0)

            metrics: List[
                Tuple[torch.nn.Parameter, torch.Tensor, float, Optional[torch.Tensor]]
            ] = []

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.to(torch.float32)
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adafactor does not support sparse gradients.")
                
                # if p has atts _scale then it is quantized. We need to divide the grad by the scale
                # if hasattr(p, "_scale"):
                #     grad = grad / p._scale

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(
                    group, grad_shape)
                # State Initialization

                if len(state) == 0:
                    # state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad_shape[:-1]).to(grad)
                        # For 2D tensors, grad_shape[:-2] is empty tuple, which is correct for column stats
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                else:
                    if use_first_moment:
                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(grad)
                        else:
                            state["exp_avg"] = state["exp_avg"].to(grad)

                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(
                            grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(
                            grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p
                is_quantized = isinstance(p_data_fp32, QBytesTensor)
                
                if is_quantized:
                    p_data_fp32 = p_data_fp32.dequantize()
                if p.dtype != torch.float32:
                    p_data_fp32 = p_data_fp32.clone().float()

                # state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                rms_t = state["RMS"]
                self._group_running_max_update(group, "rms_max", rms_t)

                state["grad_rms"] = self._rms(grad)
                gr = state["grad_rms"]
                self._group_running_max_update(group, "grad_rms_max", gr)

                beta2 = group["beta2"]
                eps = group["eps"]
                if isinstance(eps, tuple) or isinstance(eps, list):
                    eps = eps[0]
                update = (grad**2) + eps
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2).add_(
                        update.mean(dim=-1), alpha=(1.0 - beta2))
                    exp_avg_sq_col.mul_(beta2).add_(
                        update.mean(dim=-2), alpha=(1.0 - beta2))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2).add_(update, alpha=(1.0 - beta2))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                # Preconditioned + clipped direction (before LR) for fresh brake signal
                update_hat = update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    # Directional Consistency (before EMA, before LR) — fresh for _get_lr brake
                    state["dir_consistency"] = torch.nn.functional.cosine_similarity(
                        update_hat.flatten(), exp_avg.flatten(), dim=0, eps=1e-8
                    )

                lr = self._get_lr(group, state)

                if use_first_moment:
                    exp_avg = state["exp_avg"]

                    # Use beta1 if available, otherwise use default 0.9 when emergency_brake is enabled
                    beta1_for_ema = group["beta1"] if group["beta1"] is not None else 0.9

                    # Update EMA of direction without LR (always when use_first_moment=True)
                    exp_avg.mul_(beta1_for_ema).add_(update_hat, alpha=(1 - beta1_for_ema))

                    # GNS calculation (only when beta1 is not None)
                    if group["beta1"] is not None:
                        signal_sq = exp_avg.pow(2).mean()
                        current_update_sq = update_hat.pow(2).mean()
                        gns_tensor = (current_update_sq - signal_sq) / (signal_sq + 1e-12)
                    else:
                        gns_tensor = None

                    # Final update: use exp_avg only if beta1 is not None (momentum mode)
                    if group["beta1"] is not None:
                        update = exp_avg.mul(lr)
                    else:
                        update = update_hat.mul(lr)

                else:
                    gns_tensor = None
                    update = update_hat.mul(lr)

                update_rms = self._rms(update)

                if group["weight_decay"] != 0:
                    p_data_fp32.mul_(1.0 - group["weight_decay"] * update_rms)

                p_data_fp32.add_(-update)

                self._group_running_max_update(group, "update_rms_max", update_rms)

                if (p.dtype != torch.float32 or is_quantized) and self.stochastic_rounding:
                    # apply stochastic rounding
                    copy_stochastic(p, p_data_fp32)

                metrics.append(
                    (
                        p,
                        update_rms.detach(),
                        float(lr),
                        gns_tensor.detach() if gns_tensor is not None else None,
                    )
                )

            self._finalize_group_step_metrics(group, metrics)

        return loss
        
    def get_avg_learning_rate(self):
        """Average learning rate across groups (unified tensor reduction, same as get_avg_update_rms*)."""
        return self._scalars_per_group_to_avg(self.get_learning_rates())

    def get_update_rms(self):
        """
        Get RMS (root mean square) of weight updates for each parameter group (group-level ``update_rms``).

        Returns:
            List[float]: One value per group; 0.0 for groups that haven't been updated yet.
        """
        out = []
        for group in self.param_groups:
            out.append(self._group_scalar_item(group, "update_rms", 0.0))
        return out

    def get_update_rms_max(self):
        """
        Get running max of update RMS for each parameter group (single scalar per group on ``update_rms_max``).

        Returns:
            List[float]: One value per group; 0.0 for groups with no ``update_rms_max`` yet.
        """
        out = []
        for group in self.param_groups:
            out.append(self._group_scalar_item(group, "update_rms_max", 0.0))
        return out

    def get_rms(self):
        """
        Get RMS (root mean square) of parameters for each parameter group.
        Per-group value is mean over params in group via tensor reduction (_get_group_scalars).

        Returns:
            List[float]: One value per group; 0.0 for groups that haven't been updated yet.
        """
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "RMS", default=0.0, reduction='mean')
            out.append(v if v is not None else 0.0)
        return out

    def get_avg_rms(self):
        """
        Average RMS of parameters across all parameter groups (unified tensor reduction).
        """
        return self._scalars_per_group_to_avg(self.get_rms())

    def get_avg_update_rms(self):
        """
        Average RMS of weight updates across all parameter groups (unified tensor reduction).
        Useful for monitoring training stability and convergence.
        """
        return self._scalars_per_group_to_avg(self.get_update_rms())

    def get_avg_update_rms_max(self):
        """
        Average of per-group update_rms_max across groups (unified tensor reduction).
        Use with get_avg_update_rms() to monitor normalization scale and update magnitude vs recent max.
        """
        return self._scalars_per_group_to_avg(self.get_update_rms_max())

    def get_dynamic_gain(self):
        """
        Get dynamic gain for each parameter group (``dynamic_gain`` on the param group:
        group update_rms / (weighted mean grad_rms + eps)).

        If dynamic_gain falls below 0.01 - you are barely learning.
        If it is above 1.0 - you are flying blind.

        Returns:
            List[float]: One value per group; 0.0 for groups that haven't been updated yet.
        """
        out = []
        for group in self.param_groups:
            out.append(self._group_scalar_item(group, "dynamic_gain", 0.0))
        return out

    def get_avg_dynamic_gain(self):
        """
        Average dynamic gain across all parameter groups (unified tensor reduction).
        """
        return self._scalars_per_group_to_avg(self.get_dynamic_gain())

    def get_grad_rms(self):
        """
        Get weighted mean gradient RMS for each parameter group (``grad_rms`` on the param group).

        Returns:
            List[float]: One value per group; 0.0 for groups that haven't been updated yet.
        """
        out = []
        for group in self.param_groups:
            out.append(self._group_scalar_item(group, "grad_rms", 0.0))
        return out

    def get_grad_rms_max(self):
        """
        Get running max of gradient RMS for each parameter group (``grad_rms_max`` on the param group).

        Returns:
            List[float]: One value per group; 0.0 for groups with no ``grad_rms_max`` yet.
        """
        out = []
        for group in self.param_groups:
            out.append(self._group_scalar_item(group, "grad_rms_max", 0.0))
        return out

    def get_avg_grad_rms(self):
        """
        Average RMS of gradients across all parameter groups (unified tensor reduction).
        """
        return self._scalars_per_group_to_avg(self.get_grad_rms())

    def get_avg_grad_rms_max(self):
        """
        Average of per-group grad_rms_max across groups (unified tensor reduction).
        Use with get_avg_grad_rms() to monitor gradient scale vs recent max.
        """
        return self._scalars_per_group_to_avg(self.get_grad_rms_max())

    def get_gns(self):
        """Get Gradient Noise Scale per group (``gns`` on the param group)."""
        out = []
        for group in self.param_groups:
            out.append(self._group_scalar_item(group, "gns", 0.0))
        return out

    def get_dir_consistency(self):
        """Get Directional Consistency (cosine similarity to EMA) per group."""
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "dir_consistency", default=0.0, reduction='mean')
            out.append(v if v is not None else 0.0)
        return out

    def get_step_efficiency(self):
        """Get Step Efficiency per group (``step_efficiency`` on the param group)."""
        out = []
        for group in self.param_groups:
            out.append(self._group_scalar_item(group, "step_efficiency", 0.0))
        return out

    def get_avg_gns(self):
        """Average GNS across all parameter groups."""
        return self._scalars_per_group_to_avg(self.get_gns())

    def get_avg_dir_consistency(self):
        """Average Directional Consistency across all parameter groups."""
        return self._scalars_per_group_to_avg(self.get_dir_consistency())

    def get_dir_consistency_mean(self):
        """Get pre-computed mean Directional Consistency per parameter group."""
        out = []
        for group in self.param_groups:
            dc_mean = group.get("dir_consistency_mean", 0.0)
            out.append(dc_mean if dc_mean is not None else 0.0)
        return out

    def get_mean_dir_consistency(self):
        """Average of pre-computed mean Directional Consistency across all parameter groups."""
        return self._scalars_per_group_to_avg(self.get_dir_consistency_mean())

    def get_instability_score(self):
        """Get instability_score per parameter group (soft brake cumulative score)."""
        out = []
        for group in self.param_groups:
            score = group.get("instability_score", 0.0)
            out.append(score if score is not None else 0.0)
        return out

    def get_avg_instability_score(self):
        """Average instability_score across all parameter groups."""
        return self._scalars_per_group_to_avg(self.get_instability_score())

    def get_saddle_point_boost(self):
        """Same global saddle_point_boost repeated per group (for API shape); used in relative_step only."""
        b = float(self._saddle_point_boost)
        return [b] * len(self.param_groups)

    def get_avg_saddle_point_boost(self):
        """Global saddle_point_boost (identical across groups)."""
        return float(self._saddle_point_boost)

    def get_avg_step_efficiency(self):
        """Average Step Efficiency across all parameter groups."""
        return self._scalars_per_group_to_avg(self.get_step_efficiency())

    def get_beta1(self):
        """Get beta1 (momentum coefficient) for each parameter group."""
        out = []
        for group in self.param_groups:
            beta1 = group.get("beta1", 0.0)
            out.append(beta1 if beta1 is not None else 0.0)
        return out

    def get_avg_beta1(self):
        """Average beta1 (momentum coefficient) across all parameter groups."""
        return self._scalars_per_group_to_avg(self.get_beta1())
