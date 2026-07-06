#based off https://github.com/catid/dora/blob/main/dora.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Union, List

from optimum.quanto import QBytesTensor, QTensor

from toolkit.network_mixins import ToolkitModuleMixin, ExtractableModuleMixin

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork

# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear'
    # 'GroupNorm',
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]

def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T

class DoRAModule(ToolkitModuleMixin, ExtractableModuleMixin, torch.nn.Module):
    # def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
    def __init__(
            self,
            lora_name,
            org_module: torch.nn.Module,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            network: 'LoRASpecialNetwork' = None,
            use_bias: bool = False,
            **kwargs
    ):
        self.can_merge_in = False
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        self.lora_name = lora_name
        self.register_buffer("scalar", torch.tensor(1.0, device=org_module.weight.device), persistent=False)

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ in CONV_MODULES:
            raise NotImplementedError("Convolutional layers are not supported yet")

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        # self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える eng: treat as constant

        self.multiplier: Union[float, List[float]] = multiplier
        # wrap the original module so it doesn't get weights updated
        self.org_module = [org_module]
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False
        self._cached_base_weight_cpu = None
        self._cached_base_bias_cpu = None
        self._use_cpu_cached_base_norm = bool(kwargs.get("dora_cpu_cached_base_norm", False))
        self._cache_quantized_bias = bool(kwargs.get("dora_cache_quantized_bias", False))

        d_out = org_module.out_features
        d_in = org_module.in_features

        std_dev = 1 / torch.sqrt(torch.tensor(self.lora_dim).float())
        # self.lora_up = nn.Parameter(torch.randn(d_out, self.lora_dim) * std_dev)  # lora_A
        # self.lora_down = nn.Parameter(torch.zeros(self.lora_dim, d_in))  # lora_B
        self.lora_up = nn.Linear(self.lora_dim, d_out, bias=False)  # lora_B
        # self.lora_up.weight.data = torch.randn_like(self.lora_up.weight.data) * std_dev
        self.lora_up.weight.data = torch.zeros_like(self.lora_up.weight.data)
        # self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        # self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.lora_down = nn.Linear(d_in, self.lora_dim, bias=False)  # lora_A
        # self.lora_down.weight.data = torch.zeros_like(self.lora_down.weight.data)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        base_weight_ref = self.org_module[0].weight
        if (
            self._use_cpu_cached_base_norm
            and self._is_quantized_tensor(base_weight_ref)
            and not getattr(base_weight_ref, "requires_grad", False)
        ):
            self._cached_base_weight_cpu = self._materialize_weight_cpu(base_weight_ref)
            base_bias_ref = getattr(self.org_module[0], "bias", None)
            if (
                self._cache_quantized_bias
                and base_bias_ref is not None
                and self._is_quantized_tensor(base_bias_ref)
            ):
                self._cached_base_bias_cpu = self._materialize_bias_cpu(base_bias_ref)

        # m = Magnitude column-wise across output dimension
        lora_weight  = self.lora_up.weight @ self.lora_down.weight
        if self._cached_base_weight_cpu is not None:
            weight_norm = self._get_weight_norm_cpu(self._cached_base_weight_cpu, lora_weight)
        else:
            weight = self.get_orig_weight()
            weight = weight.to(self.lora_up.weight.device, dtype=self.lora_up.weight.dtype)
            weight_norm = self._get_weight_norm(weight, lora_weight)
        self.magnitude = nn.Parameter(weight_norm.detach().clone(), requires_grad=True)
        self.register_buffer("train_base_weight_norm", weight_norm.detach().clone())
        self.initial_base_weight_norm = weight_norm.detach().clone()

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
        # del self.org_module

    def get_orig_weight(self):
        weight = self.org_module[0].weight
        if isinstance(weight, QTensor) or isinstance(weight, QBytesTensor):
            return weight.dequantize().data.detach()
        elif weight.__class__.__name__ == "AffineQuantizedTensor":
            return weight.dequantize().data.detach()
        else:
            return weight.data.detach()

    def get_orig_bias(self):
        if hasattr(self.org_module[0], 'bias') and self.org_module[0].bias is not None:
            bias = self.org_module[0].bias
            if isinstance(bias, QTensor) or isinstance(bias, QBytesTensor):
                return bias.dequantize().data.detach()
            elif bias.__class__.__name__ == "AffineQuantizedTensor":
                return bias.dequantize().data.detach()
            else:
                return bias.data.detach()
        return None

    def _is_quantized_tensor(self, tensor: torch.Tensor) -> bool:
        return (
            isinstance(tensor, QTensor)
            or isinstance(tensor, QBytesTensor)
            or tensor.__class__.__name__ == "AffineQuantizedTensor"
        )

    def _materialize_weight_cpu(self, weight: torch.Tensor) -> torch.Tensor:
        if self._is_quantized_tensor(weight):
            try:
                return weight.to("cpu").dequantize().detach().to(dtype=torch.float32)
            except Exception:
                return weight.dequantize().detach().to("cpu", dtype=torch.float32)
        return weight.data.detach().to("cpu", dtype=torch.float32)

    def _materialize_bias_cpu(self, bias: torch.Tensor) -> torch.Tensor:
        if self._is_quantized_tensor(bias):
            try:
                return bias.to("cpu").dequantize().detach().to(dtype=torch.float32)
            except Exception:
                return bias.dequantize().detach().to("cpu", dtype=torch.float32)
        return bias.data.detach().to("cpu", dtype=torch.float32)

    # def dora_forward(self, x, *args, **kwargs):
    #     lora = torch.matmul(self.lora_A, self.lora_B)
    #     adapted = self.get_orig_weight() + lora
    #     column_norm = adapted.norm(p=2, dim=0, keepdim=True)
    #     norm_adapted = adapted / column_norm
    #     calc_weights = self.magnitude * norm_adapted
    #     return F.linear(x, calc_weights, self.get_orig_bias())

    def _get_weight_norm(self, weight, scaled_lora_weight) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaled_lora_weight.to(weight.device)
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm

    def _get_weight_norm_cpu(self, base_weight_cpu, scaled_lora_weight) -> torch.Tensor:
        with torch.no_grad():
            scaled_lora_weight_cpu = scaled_lora_weight.detach().to("cpu", dtype=torch.float32)
            combined = base_weight_cpu + scaled_lora_weight_cpu
            return torch.linalg.norm(combined, dim=1)

    def apply_dora(self, org_forwarded, scaled_lora_output, scaled_lora_weight):
        # ref https://github.com/huggingface/peft/blob/1e6d1d73a0850223b0916052fd8d2382a90eae5a/src/peft/tuners/lora/layer.py#L192
        # lora weight is already scaled

        # magnitude = self.lora_magnitude_vector[active_adapter]
        if self._cached_base_weight_cpu is not None:
            weight_norm = self._get_weight_norm_cpu(
                self._cached_base_weight_cpu,
                scaled_lora_weight,
            ).to(scaled_lora_weight.device, dtype=scaled_lora_weight.dtype)
        else:
            weight = self.get_orig_weight()
            weight = weight.to(scaled_lora_weight.device, dtype=scaled_lora_weight.dtype)
            weight_norm = self._get_weight_norm(weight, scaled_lora_weight)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()

        # Add epsilon clamping to avoid division-by-zero / NaNs
        weight_norm_fp32 = weight_norm.to(torch.float32)
        weight_norm_fp32 = torch.clamp(weight_norm_fp32, min=1e-6)
        weight_norm = weight_norm_fp32.to(weight_norm.dtype)

        bias = None
        if self._cached_base_bias_cpu is not None:
            bias = self._cached_base_bias_cpu.to(
                scaled_lora_output.device, dtype=scaled_lora_output.dtype
            )
        else:
            bias = self.get_orig_bias()
            if bias is not None:
                bias = bias.to(scaled_lora_output.device, dtype=scaled_lora_output.dtype)
        if bias is not None:
            direction_output = org_forwarded - bias
        else:
            direction_output = org_forwarded

        direction_output = direction_output + scaled_lora_output

        # Handle magnitude calibration across shared parameters of different base models
        if hasattr(self, "initial_base_weight_norm") and self.initial_base_weight_norm is not None:
            # Move initial_base_weight_norm to same device & dtype as magnitude
            if (
                self.initial_base_weight_norm.device != self.magnitude.device
                or self.initial_base_weight_norm.dtype != self.magnitude.dtype
            ):
                self.initial_base_weight_norm = self.initial_base_weight_norm.to(
                    self.magnitude.device, dtype=self.magnitude.dtype
                )
            
            # Fetch shared training base weight norm (if missing, default to initial_base_weight_norm)
            train_norm = getattr(self, "train_base_weight_norm", self.initial_base_weight_norm)
            if train_norm is None:
                train_norm = self.initial_base_weight_norm
            
            if train_norm.device != self.magnitude.device or train_norm.dtype != self.magnitude.dtype:
                train_norm = train_norm.to(self.magnitude.device, dtype=self.magnitude.dtype)
                
            train_norm_fp32 = train_norm.to(torch.float32)
            train_norm_fp32 = torch.clamp(train_norm_fp32, min=1e-6)
            
            calibration_ratio = (self.initial_base_weight_norm.to(torch.float32) / train_norm_fp32).to(self.magnitude.dtype)
            calibrated_magnitude = self.magnitude * calibration_ratio
        else:
            calibrated_magnitude = self.magnitude

        mag_norm_scale = (calibrated_magnitude / weight_norm - 1).view(1, -1).to(direction_output.dtype)
        return mag_norm_scale * direction_output
