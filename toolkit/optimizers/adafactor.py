import math
from typing import List, Optional
import torch
from toolkit.optimizers.optimizer_utils import copy_stochastic, stochastic_grad_accummulation
from toolkit.print import print_acc
from toolkit.util.debug import is_debug_enabled
from optimum.quanto import QBytesTensor
import random


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
        lr (`float`, *optional*, defaults to `1e-4` when `relative_step=True`):
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
            The same decay is used for gradient RMS running max (state["grad_rms_max"]) for consistency.
            When scale_parameter=True and relative_step=True, also used for per-parameter running max of
            parameter RMS (state["rms_max"]), which normalizes each parameter's scale to (0, 1] for LR
            (useful for mixed-scale groups e.g. LoRA). Allows the normalization scale to decrease over
            time so lr can recover from plateaus.
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
            (first moment, like in Adam). If not None, enables momentum.
            Suggested values: 0.9 (default), 0.95 or 0.99 for smoother updates.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `True`):
            If True, learning rate is scaled by root mean square.
            Scaling is stronger when update magnitude is large (to protect small parameters).
        relative_step (`bool`, *optional*, defaults to `True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used
        min_lr (`float`, *optional*, defaults to `1e-6`):
            Minimum learning rate multiplier for warmup phase when `warmup_init=True` and `relative_step=True`.
        lr_smoothing_rate (`float`, *optional*, defaults to `100.0`):
            Divisor for the smoothing scale in step-to-step learning rate smoothing in `_smooth_lr`.
            Larger values yield stronger smoothing (smaller step-to-step LR changes).
        factored (`bool`, *optional*, defaults to `None`):
            If True, use factored second-moment (row/col) for all parameters. If False, use full second-moment.
            If None, auto-detect: use factored for parameters with 2+ dimensions (current default behavior).

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
        eps=(1e-30, 1e-3),
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
        lr_smoothing_rate=100.0,
        warmup_steps: int = 100,
        do_parameter_swapping=False,
        parameter_swapping_factor=0.1,
        stochastic_accumulation=True,
        stochastic_rounding=True,
        factored=None,
        emergency_brake: bool = False,
    ):
        self.stochastic_rounding = stochastic_rounding

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
            "min_lr": min_lr,
            "lr_smoothing_rate": lr_smoothing_rate,
            "warmup_steps": warmup_steps,
            "factored": factored,
            "emergency_brake": emergency_brake,
            "instability_score": 0.0,  # cumulative instability tracking for soft brake
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            group["base_lr_previous"] = 0.0

        # Store LR limits, lr_smoothing_rate, rms_max_decay_rate and lr so they can be reapplied after load_state_dict (restart with new config).
        self._min_lr = min_lr
        self._lr_smoothing_rate = lr_smoothing_rate
        self._rms_max_decay_rate = rms_max_decay_rate
        self._lr = lr
        self._warmup_steps = warmup_steps
        self._beta1 = beta1
        self._beta2 = beta2
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

    def set_emergency_brake(self, value: bool) -> None:
        """Enable or disable emergency_brake at runtime (e.g. from UI)."""
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
        # Apply current run's min_lr/lr_smoothing_rate/rms_max_decay_rate/lr so changed config is used after restart.
        for group in self.param_groups:
            group["min_lr"] = max(group["eps"][0], self._min_lr)
            group["lr"] = max(group["eps"][0], self._lr)
            group["lr_smoothing_rate"] = self._lr_smoothing_rate
            group["rms_max_decay_rate"] = self._rms_max_decay_rate
            group["warmup_steps"] = self._warmup_steps
            group["beta1"] = group.get("beta1", self._beta1)
            group["beta2"] = group.get("beta2", self._beta2)
            # Normalize group_rms_max if present (old checkpoints may not have it)
            if "group_rms_max" in group and not isinstance(group["group_rms_max"], torch.Tensor):
                group["group_rms_max"] = torch.tensor(group["group_rms_max"], dtype=torch.float32)
            restored_lr = None
            for param in group["params"]:
                if param in self.state and "lr_previous" in self.state[param]:
                    lr_prev_val = self.state[param]["lr_previous"]
                    restored_lr = lr_prev_val.item() if isinstance(lr_prev_val, torch.Tensor) else lr_prev_val
                    break
            group["base_lr_previous"] = restored_lr if restored_lr is not None else 0.0
            # Restore instability_score for soft brake continuity across restarts
            group["instability_score"] = group.get("instability_score", 0.0)
            # Ensure warmup_target is initialized if warmup was active at checkpoint time
            if group.get("warmup_active", False) and "warmup_target" not in group:
                group["warmup_target"] = group["lr"]
        # Normalize state from old checkpoints: update_rms_max/update_rms must be tensors for step().
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                for key in ("update_rms_max", "update_rms", "rms_max", "grad_rms", "grad_rms_max"):
                    if key in state and not isinstance(state[key], torch.Tensor):
                        state[key] = torch.tensor(state[key], device=param.device, dtype=torch.float32)

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

    @staticmethod
    def stop_warmup(param_group, param_state=None):
        param_group["warmup_active"] = False
        if param_state is not None:
            param_state.pop("warmup_delta", None)
            param_state.pop("warmup_factor", None)
            param_state.pop("warmup_start", None)
        param_group.pop("warmup_target", None)
        if is_debug_enabled():
            print_acc(f"Adafactor: warmup stopped")

    def set_warmup_steps(self, value: int) -> None:
        """Update warmup_steps at runtime (e.g. from UI)."""
        self._warmup_steps = value
        for group in self.param_groups:
            group["warmup_steps"] = value
        if is_debug_enabled():
            print_acc(f"Adafactor: applied runtime warmup_steps={value}")

    def _get_lr(self, param_group, param_state):
        """
        Compute per-parameter learning rate.

        Manual mode (relative_step=False):
          Returns group["lr"]. If scale_parameter=True, multiplies by max(eps1, param_rms).

        Adaptive mode (relative_step=True):

        Warmup (warmup_init=True, requires relative_step=True):

        Returns:
            float: learning rate for this parameter
        """
        # Extract LR config parameters
        base_lr    = param_group["lr"]              # Maximum (cap) learning rate
        min_lr     = param_group["min_lr"]          # Minimum learning rate
        eps0       = param_group["eps"][0]          # Small constant for numerical stability (division guard)
        eps1       = param_group["eps"][1]          # Parameter scale regularization constant
        param_rms  = param_state["RMS"].item()      # Current parameter RMS magnitude
        scale      = 1.0                            # Default scale for LR
        relative   = 1.0                            # Default relative for LR
        warmup     = 1.0                            # Default warmup for LR

        # Initialize warmup_active on first use if not present
        if "warmup_active" not in param_group:
            param_group["warmup_active"] = param_group["warmup_init"]

        # Track base_lr changes and activate warmup for both increase and decrease
        base_lr_prev = param_group.get("base_lr_previous", 0.0)
        if base_lr != base_lr_prev and param_group["warmup_init"]:
            # Get current effective LR (what was actually applied last step)
            # This ensures smooth transition from current interpolated LR, not from old target
            current_effective_lr = param_state.get("lr_previous", base_lr_prev)

            param_state["warmup_start"] = current_effective_lr
            param_group["warmup_target"] = base_lr
            param_group["warmup_active"] = True
            param_state.pop("warmup_factor", None)
            param_state.pop("warmup_delta", None)
            if is_debug_enabled():
                direction = "up" if base_lr > base_lr_prev else "down"
                print_acc(
                    f"Adafactor: base_lr changed ({current_effective_lr:.2e} -> {base_lr:.2e}, {direction}), starting warmup"
                )
        param_group["base_lr_previous"] = base_lr

        if param_group["scale_parameter"]:
            # Scale LR by parameter magnitude for better adaptation to parameter scale
            scale = max(eps1, param_rms)

        if param_group["relative_step"]:
            # Adaptive LR mode: compute LR from gradient and parameter statistics
            grad_rms      = param_state["grad_rms"].item()        # Current gradient RMS
            group_rms_max = param_group.get("group_rms_max", torch.tensor(eps1)).item()  # Group-level max parameter RMS

            brake = 1.0
            soft_brake = 1.0

            if param_group.get("emergency_brake", False):
                # Instant Brake: multiplicative factor based on current directional consistency
                # Prefer fresh per-parameter dir_consistency; fallback to group mean (when beta1=None)
                dc = param_state.get("dir_consistency")
                if dc is not None:
                    dir_val = dc.item() if isinstance(dc, torch.Tensor) else float(dc)
                else:
                    dir_val = param_group.get("dir_consistency_mean") or 0.0

                brake = max(0.9, min(1 + dir_val, 1.0))

                # Soft Brake: exponential damping based on cumulative instability
                # exp(-score) smoothly reduces LR: score=0 → 1.0, score=2 → 0.135, score=4 → 0.018
                instability_score = param_group.get("instability_score", 0.0)
                soft_brake = math.exp(-instability_score)
                soft_brake = max(0.5, soft_brake)

            # Ratio of parameter RMS to group RMS max
            ratio = max(eps0, (group_rms_max - param_rms) / (group_rms_max + eps0))

            relative = (1 + min_lr * ratio) * brake * soft_brake

        if param_group.get("warmup_active", False):
            warmup_steps = param_group.get("warmup_steps", self._warmup_steps)
            warmup_start = param_state.get("warmup_start", 0.0)
            warmup_target = param_group.get("warmup_target", base_lr)

            # Initialize warmup_factor: progress from 0.0 to 1.0
            if "warmup_factor" not in param_state:
                param_state["warmup_factor"] = 0.0
                param_state["warmup_delta"] = 1.0 / max(1, warmup_steps)

            warmup_factor = param_state["warmup_factor"]
            delta = param_state["warmup_delta"]

            # Increment warmup progress
            new_warmup_factor = min(1.0, warmup_factor + delta)
            param_state["warmup_factor"] = new_warmup_factor

            # Interpolate base_lr from start to target
            interpolated_base_lr = warmup_start + new_warmup_factor * (warmup_target - warmup_start)

            # Stop warmup when factor reaches 1.0
            if new_warmup_factor >= 1.0:
                self.stop_warmup(param_group, param_state)
                base_lr = warmup_target
            else:
                base_lr = interpolated_base_lr

        new_lr = base_lr * scale * relative

        # Update-to-Weight Ratio safeguard: prevent updates > 10% of parameter magnitude
        if param_group.get("emergency_brake", False):
            if param_rms > eps1:
                # Cap LR to limit update magnitude relative to parameter scale
                # Ratio > 0.1 (10% per step) typically indicates overshoot
                max_allowed_lr = max(base_lr/10, param_rms * 0.1)
                if new_lr > max_allowed_lr:
                    new_lr = max_allowed_lr

        param_state["lr_previous"] = new_lr
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

    def _update_beta2_from_gns(self, group, group_state):
        """
        Softly adjust group["beta2"] toward a GNS-based target (only when relative_step=True).
        Low GNS (< 4) -> target 0.88; high GNS (> 10) -> target 0.99; else 0.9.
        """
        target_beta2 = 0.9
        gns_t = group_state.get("gns")
        current_gns = gns_t.item() if gns_t is not None else 0.0

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
        use_first_moment = param_group["beta1"] is not None or param_group.get("emergency_brake", False)
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _get_group_scalars(self, group, state_key, default=0.0, reduction='mean'):
        """
        Collect per-parameter scalars from state for a group, reduce in tensor space, return float.
        Used for unified metric aggregation (RMS, update_rms, update_rms_max) so get_* use the same
        path. Only params that have state_key in state are included (same as get_update_rms/get_update_rms_max).
        """
        values = []
        weights = []
        device = None
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

    # adafactor manages its own lr
    def get_learning_rates(self):
        """
        One value per group: mean LR over params in group (same aggregation as get_update_rms/get_update_rms_max).
        Fallback to group["lr"] or 0.0 when no param in group has state yet.
        """
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "lr_previous", default=0.0, reduction='mean')
            out.append(v if v is not None else (group["lr"] if group["lr"] is not None else 0.0))
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

        for group in self.param_groups:
            # Decay group_rms_max once per step
            if "group_rms_max" in group:
                group["group_rms_max"] = group["group_rms_max"] * group["rms_max_decay_rate"]

            # Pre-compute mean directional consistency once per group for _get_lr
            group["dir_consistency_mean"] = self._get_group_scalars(group, "dir_consistency", default=0.0, reduction='mean')

            # Soft Brake: accumulate instability score when emergency_brake is enabled
            if group.get("emergency_brake", False):
                dc_mean = group.get("dir_consistency_mean", 0.0)
                score = group.get("instability_score", 0.0)

                if dc_mean < 0:
                    # Accumulate penalty for inconsistent directions (gradient reversals)
                    # Penalty scales with magnitude: stronger for -1.0 (180° reversal) than -0.1
                    score += abs(dc_mean) * 0.1
                else:
                    # Decay penalty when directions are consistent
                    score *= 0.95

                # Clamp score to [0.0, 4.0] to prevent LR from dropping to zero permanently
                group["instability_score"] = min(max(score, 0.0), 4.0)

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

                    # All metric scalars in state are 0-dim tensors for unified collection via _get_group_scalars.
                    state["RMS"] = torch.tensor(0.0, device=grad.device)
                else:
                    if use_first_moment:
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
                if "rms_max" not in state:
                    state["rms_max"] = state["RMS"].clone().detach()
                else:
                    state["rms_max"] = torch.maximum(
                        state["rms_max"] * group["rms_max_decay_rate"], state["RMS"]
                    )
                # Update group running max
                if "group_rms_max" not in group:
                    group["group_rms_max"] = state["RMS"].clone().detach()
                else:
                    group["group_rms_max"] = torch.maximum(
                        group["group_rms_max"], state["RMS"]
                    )
                # Store grad_rms and grad_rms_max as 0-dim tensors (same path as update_rms/update_rms_max).
                state["grad_rms"] = self._rms(grad)
                current_grad_max = state.get("grad_rms_max")
                if current_grad_max is None:
                    current_grad_max = torch.tensor(0.0, device=grad.device, dtype=grad.dtype)
                elif not isinstance(current_grad_max, torch.Tensor):
                    current_grad_max = torch.tensor(current_grad_max, device=grad.device, dtype=grad.dtype)
                state["grad_rms_max"] = torch.maximum(
                    current_grad_max * group["rms_max_decay_rate"], state["grad_rms"]
                )

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
                        state["gns"] = (current_update_sq - signal_sq) / (signal_sq + 1e-12)
                    else:
                        state["gns"] = torch.tensor(0.0, device=update_hat.device)

                    # Final update: use exp_avg only if beta1 is not None (momentum mode)
                    if group["beta1"] is not None:
                        update = exp_avg.mul(lr)
                    else:
                        update = update_hat.mul(lr)

                else:
                    state["gns"] = torch.tensor(0.0, device=update_hat.device)
                    update = update_hat.mul(lr)

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)

                # Store as 0-dim tensors for unified metric aggregation via _get_group_scalars (same path as RMS, get_update_rms*).
                state["update_rms"] = self._rms(update)
                current_max = state.get("update_rms_max")
                if current_max is None:
                    current_max = torch.tensor(0.0, device=update.device, dtype=update.dtype)
                elif not isinstance(current_max, torch.Tensor):
                    current_max = torch.tensor(current_max, device=update.device, dtype=update.dtype)
                state["update_rms_max"] = torch.maximum(
                    current_max * group["rms_max_decay_rate"], state["update_rms"]
                )
                # Step Efficiency: ratio of current update RMS to historical maximum
                eps = group["eps"][0] if isinstance(group["eps"], (tuple, list)) else group["eps"]
                state["step_efficiency"] = state["update_rms"] / (state["update_rms_max"] + eps)

                if (p.dtype != torch.float32 or is_quantized) and self.stochastic_rounding:
                    # apply stochastic rounding
                    copy_stochastic(p, p_data_fp32)

                state["dynamic_gain"] = state["update_rms"] / (state["grad_rms"] + group["eps"][0])

        # Compute global mean dynamic_gain using unified metric pattern
        global_mean_dynamic_gain = self.get_avg_dynamic_gain()

        # Update beta1 once per group using global_mean_dynamic_gain; groups without dynamic_gain are effectively skipped
        # if global_mean_dynamic_gain > 0:
        #     for group in self.param_groups:
        #         # If group has no parameters with dynamic_gain, skip updating beta1 for this group
        #         has_dynamic_gain = any(
        #             (p in self.state and "dynamic_gain" in self.state[p])
        #             for p in group["params"]
        #         )
        #         if not has_dynamic_gain:
        #             continue
        #         self._update_beta1_from_dynamic_gain(group, global_mean_dynamic_gain)

        return loss
        
    def get_avg_learning_rate(self):
        """Average learning rate across groups (unified tensor reduction, same as get_avg_update_rms*)."""
        return self._scalars_per_group_to_avg(self.get_learning_rates())

    def get_update_rms(self):
        """
        Get RMS (root mean square) of weight updates for each parameter group.
        Per-group value is mean over params in group via tensor reduction (_get_group_scalars).

        Returns:
            List[float]: One value per group; 0.0 for groups that haven't been updated yet.
        """
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "update_rms", default=0.0, reduction='mean')
            out.append(v if v is not None else 0.0)
        return out

    def get_update_rms_max(self):
        """
        Get running max of update RMS for each parameter group.
        Per-group value is mean over params in group via tensor reduction (_get_group_scalars).

        Returns:
            List[float]: One value per group; 0.0 for groups with no update_rms_max in state yet.
        """
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "update_rms_max", default=0.0, reduction='mean')
            out.append(v if v is not None else 0.0)
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
        Get dynamic gain (update_rms / grad_rms) for each parameter group.
        Per-group value is mean over params in group via tensor reduction.

        If dynamic_gain falls below 0.01 - you are barely learning.
        If it is above 1.0 - you are flying blind.

        Returns:
            List[float]: One value per group; 0.0 for groups that haven't been updated yet.
        """
        out = []
        for group in self.param_groups:
            update_rms = self._get_group_scalars(group, "update_rms", default=0.0, reduction='mean')
            grad_rms = self._get_group_scalars(group, "grad_rms", default=0.0, reduction='mean')
            if update_rms is not None and grad_rms is not None and grad_rms > 0:
                eps = group["eps"][0]
                out.append(update_rms / (grad_rms + eps))
            else:
                out.append(0.0)
        return out

    def get_avg_dynamic_gain(self):
        """
        Average dynamic gain across all parameter groups (unified tensor reduction).
        """
        return self._scalars_per_group_to_avg(self.get_dynamic_gain())

    def get_grad_rms(self):
        """
        Get RMS (root mean square) of gradients for each parameter group.
        Per-group value is mean over params in group via tensor reduction (_get_group_scalars).

        Returns:
            List[float]: One value per group; 0.0 for groups that haven't been updated yet.
        """
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "grad_rms", default=0.0, reduction='mean')
            out.append(v if v is not None else 0.0)
        return out

    def get_grad_rms_max(self):
        """
        Get running max of gradient RMS for each parameter group.
        Per-group value is mean over params in group via tensor reduction (_get_group_scalars).

        Returns:
            List[float]: One value per group; 0.0 for groups with no grad_rms_max in state yet.
        """
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "grad_rms_max", default=0.0, reduction='mean')
            out.append(v if v is not None else 0.0)
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
        """Get Gradient Noise Scale per group."""
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "gns", default=0.0, reduction='mean')
            out.append(v if v is not None else 0.0)
        return out

    def get_dir_consistency(self):
        """Get Directional Consistency (cosine similarity to EMA) per group."""
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "dir_consistency", default=0.0, reduction='mean')
            out.append(v if v is not None else 0.0)
        return out

    def get_step_efficiency(self):
        """Get Step Efficiency (current_rms / max_rms) per group."""
        out = []
        for group in self.param_groups:
            v = self._get_group_scalars(group, "step_efficiency", default=0.0, reduction='mean')
            out.append(v if v is not None else 0.0)
        return out

    def get_avg_gns(self):
        """Average GNS across all parameter groups."""
        return self._scalars_per_group_to_avg(self.get_gns())

    def get_avg_dir_consistency(self):
        """Average Directional Consistency across all parameter groups."""
        return self._scalars_per_group_to_avg(self.get_dir_consistency())

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
