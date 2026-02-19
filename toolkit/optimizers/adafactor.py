import math
from typing import List
import torch
from toolkit.optimizers.optimizer_utils import copy_stochastic, stochastic_grad_accummulation
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
        lr (`float`, *optional*):
            The external learning rate.
        eps (`Tuple[float, float]`, *optional*, defaults to `(1e-30, 0.001)`):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (`float`, *optional*, defaults to 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (`float`, *optional*, defaults to -0.8):
            Coefficient used to compute running averages of square
        rms_max_decay_rate (`float`, *optional*, defaults to `0.97`):
            Decay rate for running max of update RMS used in activity normalization.
            Applied each step: ``update_rms_max = max(update_rms_max * rms_max_decay_rate, update_rms)``.
            Also used for group-level running max of parameter RMS (param_rms_max) when scale_parameter=True,
            to normalize per-parameter scale to (0, 1] within the group (useful for LoRA).
            Allows the normalization scale to decrease over time so lr can recover from plateaus.
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
            (first moment, like in Adam). If not None, enables momentum.
            Suggested values: 0.9 (default), 0.95 or 0.99 for smoother updates.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `True`):
            If True, learning rate is scaled by root mean square
        relative_step (`bool`, *optional*, defaults to `True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used
        min_lr (`float`, *optional*, defaults to `1e-6`):
            Minimum learning rate multiplier for warmup phase when `warmup_init=True` and `relative_step=True`.
            Controls the linear growth rate: `lr = min_lr * step` during warmup.
        max_lr (`float`, *optional*, defaults to `1e-2`):
            Maximum learning rate cap for relative step mode when `relative_step=True`.
            Acts as upper bound for `min_step` when `warmup_init=False` or when warmup phase completes.
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
        lr=None,
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
        max_lr=1e-4,
        lr_smoothing_rate=100.0,
        do_parameter_swapping=False,
        parameter_swapping_factor=0.1,
        stochastic_accumulation=True,
        stochastic_rounding=True,
    ):
        self.stochastic_rounding = stochastic_rounding
        if lr is not None and lr != 0 and relative_step:
            raise ValueError(
                "Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError(
                "`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "rms_max_decay_rate": rms_max_decay_rate,
            "param_rms_max": 0.0,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
            "min_lr": min_lr,
            "max_lr": max_lr,
            "lr_smoothing_rate": lr_smoothing_rate,
        }
        super().__init__(params, defaults)
        
        # Store LR limits, lr_smoothing_rate, rms_max_decay_rate and external lr so they can be reapplied after load_state_dict (restart with new config).
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._lr_smoothing_rate = lr_smoothing_rate
        self._rms_max_decay_rate = rms_max_decay_rate
        self._lr = lr

        self.base_lrs: List[float] = [
            group['lr'] for group in self.param_groups
        ]

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

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Apply current run's min_lr/max_lr/lr_smoothing_rate/rms_max_decay_rate/lr so changed config is used after restart.
        for group in self.param_groups:
            group["min_lr"] = self._min_lr
            group["max_lr"] = self._max_lr
            group["lr_smoothing_rate"] = self._lr_smoothing_rate
            group["rms_max_decay_rate"] = self._rms_max_decay_rate
            group["param_rms_max"] = group.get("param_rms_max", 0.0)
            if self._lr is not None:
                group["lr"] = self._lr
        # Normalize state from old checkpoints: update_rms_max/update_rms must be tensors for step().
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                for key in ("update_rms_max", "update_rms"):
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
        rel_step_sz = param_group["lr"]  # external lr when relative_step=False
        eps0 = param_group["eps"][0]
        eps1 = param_group["eps"][1]
        min_lr = param_group["min_lr"]
        max_lr = param_group["max_lr"]
        param_scale = 1.0

        if param_group["scale_parameter"]:
            # State holds tensors; convert to float so _get_lr returns float and step/_smooth_lr/get_learning_rates keep working.
            rms_val = param_state["RMS"]
            rms_val = rms_val.item() if isinstance(rms_val, torch.Tensor) else rms_val
            param_scale = max(eps1, rms_val)

        if param_group["relative_step"]:
            # Activity = prev_update_rms normalized to (0, 1] via running max.
            # Large updates → activity near 1 → lr near max_lr; small → activity near 0 → lr near min_lr.

            # Normalize param_scale to (0, 1] by group's param_rms_max; floor at eps1 so small weights
            # (e.g. LoRA near zero) still get a minimum effective LR (~eps1 relative to 1.0) and can
            # escape flat regions instead of stalling when param_rms_max grows much larger than their RMS.
            param_scale = max(
                eps1,
                min(1.0, param_scale / (param_group.get("param_rms_max", 0.0) + eps0)),
            )

            # State holds tensors; use .item() so activity and lr stay float (needed for step, _smooth_lr, get_learning_rates).
            prev_update_rms = param_state.get("update_rms", 0.0)
            prev_update_rms = prev_update_rms.item() if isinstance(prev_update_rms, torch.Tensor) else prev_update_rms
            update_rms_max = param_state.get("update_rms_max", 0.0)
            update_rms_max = update_rms_max.item() if isinstance(update_rms_max, torch.Tensor) else update_rms_max
            lr_previous = param_state.get("lr_previous", 0.0)
            lr_previous = lr_previous.item() if isinstance(lr_previous, torch.Tensor) else lr_previous

            part1 = min(1.0, prev_update_rms / (update_rms_max + eps0)) / 2
            part2 = max(0, 1 - abs(lr_previous - prev_update_rms) / (lr_previous + eps0)) / 2
            activity = part1 + part2 # in [0, 1]

            if min_lr == 0:
                new_lr = max(max_lr/10, activity * max_lr)  # floor eps0 to avoid exact zero
            else:
                new_lr = (1.0 - activity) * min_lr + activity * max_lr

            # param_scale influence: weaker at high activity, stronger at low activity
            effective_scale = (1.0 - activity) * param_scale + activity * 1.0
            new_lr = new_lr * effective_scale

            if param_group.get("warmup_init", False):
                warmup_target = new_lr
                current = lr_previous
                gap = warmup_target - current
                if gap > 0:
                    warmup_step = update_rms_max - prev_update_rms + eps0
                    step_actual = min(warmup_step, gap)
                    new_lr = current + step_actual

        else:
            new_lr = param_scale * rel_step_sz  # external schedule, scaled by param RMS

        # Smooth step-to-step changes and clamp to [min_lr, max_lr].
        smooth_lr = self._smooth_lr(param_group, param_state, new_lr)
        new_lr = max(min_lr, min(smooth_lr, max_lr))
        if min_lr == 0:
            new_lr = max(eps0, new_lr)  # re-apply floor after smoothing

        param_state["lr_previous"] = new_lr  # used by _smooth_lr next step
        return (new_lr.item() if isinstance(new_lr, torch.Tensor) else new_lr)

    def _smooth_lr(self, param_group, param_state, raw_lr):
        # Blend raw_lr with previous step's final lr to reduce step-to-step jumps.
        # Larger |raw_lr - lr_previous| → more weight on lr_previous → smoother.
        min_lr = param_group["min_lr"]
        max_lr = param_group["max_lr"]
        lr_smoothing_rate = param_group["lr_smoothing_rate"]
        lr_previous = param_state.get("lr_previous", raw_lr)
        smoothing_scale = (max_lr - min_lr) / lr_smoothing_rate
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
        Used for unified metric aggregation (RMS, update_rms, update_rms_max) so get_* and
        param_rms_max use the same path. Only params that have state_key in state are included
        (same as current get_update_rms/get_update_rms_max).
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
        lrs = []
        for group in self.param_groups:
            lr_list = [
                self._get_lr(group, self.state[param])
                for param in group["params"]
                if param in self.state and len(self.state[param]) > 0
            ]
            if lr_list:
                lrs.append(torch.tensor(lr_list, dtype=torch.float64).mean().item())
            else:
                lrs.append(group["lr"] if group["lr"] is not None else 0.0)
        return lrs if lrs else self.base_lrs

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
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
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

            # Unified group scalar collection with max reduction (same path as get_update_rms/get_update_rms_max).
            group_rms_max = self._get_group_scalars(group, "RMS", default=0.0, reduction='max')
            if group_rms_max is not None:
                group["param_rms_max"] = max(
                    group["param_rms_max"] * group["rms_max_decay_rate"],
                    group_rms_max,
                )

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
