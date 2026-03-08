import math
from typing import List
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
            Coefficient used to compute running averages of square
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
        rms_max_decay_rate=0.97,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
        min_lr=1e-6,
        lr_smoothing_rate=100.0,
        do_parameter_swapping=False,
        parameter_swapping_factor=0.1,
        stochastic_accumulation=True,
        stochastic_rounding=True,
    ):
        self.stochastic_rounding = stochastic_rounding
        if warmup_init and not relative_step:
            raise ValueError(
                "`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "rms_max_decay_rate": rms_max_decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
            "min_lr": min_lr,
            "lr_smoothing_rate": lr_smoothing_rate,
        }
        super().__init__(params, defaults)
        
        # Store LR limits, lr_smoothing_rate, rms_max_decay_rate and lr so they can be reapplied after load_state_dict (restart with new config).
        self._min_lr = min_lr
        self._lr_smoothing_rate = lr_smoothing_rate
        self._rms_max_decay_rate = rms_max_decay_rate
        self._lr = lr

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

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Apply current run's min_lr/lr_smoothing_rate/rms_max_decay_rate/lr so changed config is used after restart.
        for group in self.param_groups:
            group["min_lr"] = max(group["eps"][0], self._min_lr)
            group["lr"] = max(group["eps"][0], self._lr)
            group["lr_smoothing_rate"] = self._lr_smoothing_rate
            group["rms_max_decay_rate"] = self._rms_max_decay_rate
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

    def _get_lr(self, param_group, param_state):
        new_lr = param_group["lr"] # external lr or ceiling
        cap_lr = param_group["lr"] # ceiling when relative_step=True
        eps0 = param_group["eps"][0]
        eps1 = param_group["eps"][1]
        min_lr = param_group["min_lr"]
        param_scale = 1.0
        rms_val = param_state["RMS"]
        rms_val = rms_val.item() if isinstance(rms_val, torch.Tensor) else rms_val

        if param_group["scale_parameter"]:
            # Tie param_scale to the parameter's RMS so we don't give large lr when RMS is small; floor at eps1
            # so small weights still get a minimum scale (used as-is when relative_step=False, else in relative_step formula).
            param_scale = max(eps1, rms_val)
            if not param_group["relative_step"]:
                new_lr *= param_scale

        if param_group["relative_step"]:
            grad_rms_val = param_state["grad_rms"]
            grad_rms_val = grad_rms_val.item() if isinstance(grad_rms_val, torch.Tensor) else grad_rms_val
            grad_rms_max_val = param_state["grad_rms_max"]
            grad_rms_max_val = grad_rms_max_val.item() if isinstance(grad_rms_max_val, torch.Tensor) else grad_rms_max_val

            activity = min(1.0, 0.5 + 0.5 * grad_rms_val / (grad_rms_max_val + eps1))
            new_lr = cap_lr * param_scale * activity

            if param_group.get("warmup_init", False):
                lr_previous = param_state.get("lr_previous", 0.0)
                warmup_target = param_group["lr"]
                gap = warmup_target - lr_previous
                if gap > 0:
                    update_rms_max = param_state.get("update_rms_max", 0.0)
                    update_rms_max = update_rms_max.item() if isinstance(update_rms_max, torch.Tensor) else update_rms_max
                    warmup_step = max(cap_lr * eps1, update_rms_max + eps0)
                    step_actual = min(warmup_step, gap)
                    new_lr = lr_previous + step_actual
                else:
                    param_group["warmup_init"] = False

            new_lr = max(min_lr, min(new_lr, cap_lr))

        param_state["lr_previous"] = new_lr

        return new_lr

    def _smooth_lr(self, param_group, param_state, raw_lr):
        # Blend raw_lr with previous step's final lr to reduce step-to-step jumps.
        # Larger |raw_lr - lr_previous| → more weight on lr_previous → smoother.
        min_lr = param_group["min_lr"]
        cap_lr = param_group["lr"]
        lr_smoothing_rate = param_group["lr_smoothing_rate"]
        lr_previous = param_state.get("lr_previous", raw_lr)
        smoothing_scale = (cap_lr - min_lr) / lr_smoothing_rate
        lr_delta = raw_lr - lr_previous
        denominator = abs(lr_delta) + smoothing_scale + param_group["eps"][0]
        blend_weight = abs(lr_delta) / denominator
        smoothed_lr = (1 - blend_weight) * raw_lr + blend_weight * lr_previous
        return smoothed_lr

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
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
        tensors = []
        device = None
        for p in group["params"]:
            if p not in self.state or state_key not in self.state[p]:
                continue
            val = self.state[p][state_key]
            t = torch.as_tensor(val, device=p.device)
            if device is None:
                device = t.device
            tensors.append(t.to(device))
        if not tensors:
            return None
        stacked = torch.stack(tensors)
        if reduction == 'max':
            return stacked.max().item()
        return stacked.mean().item()

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
                    state["step"] = 0

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

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                if "rms_max" not in state:
                    state["rms_max"] = state["RMS"].clone().detach()
                else:
                    state["rms_max"] = torch.maximum(
                        state["rms_max"] * group["rms_max_decay_rate"], state["RMS"]
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
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                beta2t = min(0.99, beta2t)
                eps = group["eps"]
                if isinstance(eps, tuple) or isinstance(eps, list):
                    eps = eps[0]
                update = (grad**2) + eps
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=(1.0 - beta2t))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(
                        update, alpha=(1 - group["beta1"]))
                    update = exp_avg

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

                if (p.dtype != torch.float32 or is_quantized) and self.stochastic_rounding:
                    # apply stochastic rounding
                    copy_stochastic(p, p_data_fp32)

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
