# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import HRAConfig


def _cwy_factors(opt_u: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the compact-WY factors for a product of Householder reflections.

    If the columns of ``U`` are normalized Householder vectors, then

        H(U[:, 0]) ... H(U[:, r - 1]) = I - U @ inv(S) @ U.T,

    where ``S = 0.5 I + triu(U.T @ U, diagonal=1)``.  Reversing the
    columns represents the transpose/inverse product.  Factor construction
    uses at least float32: the small triangular solve is substantially more
    sensitive to reduced precision than the final matrix products, and CUDA
    does not implement the solve for float16/bfloat16.
    """
    with torch.autocast(device_type=opt_u.device.type, enabled=False):
        factor_u = opt_u.float() if opt_u.dtype in (torch.float16, torch.bfloat16) else opt_u
        normalized_u = factor_u / factor_u.norm(dim=0)
        if reverse:
            normalized_u = normalized_u.flip(dims=(1,))

        rank = normalized_u.shape[1]
        identity = torch.eye(rank, device=normalized_u.device, dtype=normalized_u.dtype)
        s = 0.5 * identity + torch.triu(normalized_u.transpose(0, 1) @ normalized_u, diagonal=1)
        inverse_s = torch.linalg.solve_triangular(s, identity, upper=True)
    return normalized_u, inverse_s


def _right_multiply_hra(
    value: torch.Tensor,
    opt_u: torch.Tensor,
    apply_GS: bool,
    reverse: bool = False,
    cast_input: bool = True,
) -> torch.Tensor:
    """Right-multiply ``value`` by HRA's orthogonal transform without forming it.

    The non-GS path is the compact-WY (CWY) representation of exactly the
    same ordered Householder product used by HRA.  It replaces ``r`` serial
    updates of a dense ``d x d`` matrix with parallel matrix multiplications
    through ``d x r`` factors and one ``r x r`` triangular solve.
    """
    if apply_GS:
        # Only the projector Q Q^T is used, so QR column signs do not matter.
        # This is the vectorized equivalent of the previous classical
        # Gram-Schmidt loop.
        with torch.autocast(device_type=opt_u.device.type, enabled=False):
            factor_u = opt_u.float() if opt_u.dtype in (torch.float16, torch.bfloat16) else opt_u
            normalized_u = torch.linalg.qr(factor_u, mode="reduced").Q
        inverse_s = None
    else:
        normalized_u, inverse_s = _cwy_factors(opt_u, reverse=reverse)

    device_type = value.device.type
    if torch.is_autocast_enabled(device_type):
        compute_dtype = torch.get_autocast_dtype(device_type)
    elif cast_input:
        compute_dtype = opt_u.dtype
    else:
        compute_dtype = value.dtype

    value = value.to(compute_dtype)
    normalized_u = normalized_u.to(compute_dtype)
    if apply_GS:
        return value - 2 * (value @ normalized_u) @ normalized_u.transpose(0, 1)

    inverse_s = inverse_s.to(compute_dtype)
    return value - ((value @ normalized_u) @ inverse_s) @ normalized_u.transpose(0, 1)


class HRALayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("hra_u",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("hra_r", "hra_apply_GS")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.hra_r = {}
        self.hra_apply_GS = {}
        self.hra_u = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            self.in_features, self.out_features = base_layer.in_channels, base_layer.out_channels
        else:
            raise TypeError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        config: HRAConfig,
        **kwargs,
    ) -> None:
        """Internal function to create hra adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            config (`HRAConfig`): The adapter configuration for this layer.
        """
        apply_GS = config.apply_GS
        init_weights = config.init_weights
        inference_mode = config.inference_mode

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.hra_r[adapter_name] = r
        self.hra_apply_GS[adapter_name] = apply_GS

        # Determine shape of HRA weights
        base_layer = self.get_base_layer()
        base_device, base_dtype = self._get_base_layer_device_and_dtype(base_layer)
        factory_kwargs = {}
        if base_device is not None:
            factory_kwargs["device"] = base_device
        if base_dtype is not None and (base_dtype.is_floating_point or base_dtype.is_complex):
            factory_kwargs["dtype"] = base_dtype
        if isinstance(base_layer, nn.Linear):
            self.hra_u[adapter_name] = nn.Parameter(
                torch.empty(self.in_features, r, **factory_kwargs), requires_grad=True
            )
        elif isinstance(base_layer, nn.Conv2d):
            self.hra_u[adapter_name] = nn.Parameter(
                torch.empty(
                    self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0],
                    r,
                    **factory_kwargs,
                ),
                requires_grad=True,
            )
        else:
            raise TypeError(f"HRA is not implemented for base layers of type {type(base_layer).__name__}")

        # Initialize weights
        if init_weights:
            self.reset_hra_parameters(adapter_name)
        else:
            self.reset_hra_parameters_random(adapter_name)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_hra_parameters(self, adapter_name: str):
        if self.hra_r[adapter_name] % 2 != 0:
            warnings.warn("The symmetric initialization can NOT be performed when r is odd!")
            nn.init.kaiming_uniform_(self.hra_u[adapter_name], a=math.sqrt(5))
        else:
            # Adjacent equal Householder vectors give H(u)H(u) = I.  Initialize the
            # even columns in place and copy them to their partners.  The old code
            # allocated a second d x (r/2) tensor on CPU and then repeat_interleave'd
            # it into a third d x r tensor for every target layer, which made HRA
            # wrapping of 7--8B LLMs take many minutes before step 1.  The strided
            # view has the same shape as the former half_u, preserving Kaiming's fan
            # calculation and the exact symmetric initialization.
            with torch.no_grad():
                representatives = self.hra_u[adapter_name][:, 0::2]
                nn.init.kaiming_uniform_(representatives, a=math.sqrt(5))
                self.hra_u[adapter_name][:, 1::2].copy_(representatives)

    def reset_hra_parameters_random(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.hra_u[adapter_name], a=math.sqrt(5))

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.hra_u.keys():
                continue

            warnings.warn("Scaling operation for HRA not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.hra_u.keys():
                continue

            warnings.warn("Unscaling operation for HRA not supported! Keeping scale at 1.")


class HRALinear(nn.Module, HRALayer):
    """
    HRA implemented in a dense layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: HRAConfig,
        r: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        HRALayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, config=config, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.hra_u.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_weight = _right_multiply_hra(
                        orig_weight,
                        self.hra_u[active_adapter],
                        self.hra_apply_GS[active_adapter],
                    )

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight.to(orig_dtype)
                else:
                    new_weight = _right_multiply_hra(
                        base_layer.weight.data,
                        self.hra_u[active_adapter],
                        self.hra_apply_GS[active_adapter],
                    )
                    base_layer.weight.data = new_weight.to(orig_dtype)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            base_layer = self.get_base_layer()
            orig_dtype = base_layer.weight.dtype
            if active_adapter in self.hra_u.keys():
                orig_weight = base_layer.weight.data.clone()
                new_weight = _right_multiply_hra(
                    orig_weight,
                    self.hra_u[active_adapter],
                    self.hra_apply_GS[active_adapter],
                    reverse=True,
                )
                base_layer.weight.data = new_weight.to(orig_dtype)

    def get_delta_weight(self, adapter_name: str, reverse: bool = False) -> torch.Tensor:
        opt_u = self.hra_u[adapter_name]
        identity = torch.eye(opt_u.shape[0], device=opt_u.device, dtype=opt_u.dtype)
        return _right_multiply_hra(identity, opt_u, self.hra_apply_GS[adapter_name], reverse=reverse)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            # W' = W Q, hence x W'^T = (x Q^T) W^T.  Applying Q^T to
            # activations avoids both the dense d x d rotation and a rotated
            # copy of the base weight.  Reverse adapter order preserves the
            # exact composition when more than one adapter is active.
            transformed_x = x
            for active_adapter in reversed(self.active_adapters):
                if active_adapter not in self.hra_u.keys():
                    continue
                transformed_x = _right_multiply_hra(
                    transformed_x,
                    self.hra_u[active_adapter],
                    self.hra_apply_GS[active_adapter],
                    reverse=True,
                    cast_input=self.cast_input_dtype_enabled,
                )

            transformed_x = transformed_x.to(self.get_base_layer().weight.dtype)
            result = self.base_layer(transformed_x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hra." + rep


class HRAConv2d(nn.Module, HRALayer):
    """HRA implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: HRAConfig,
        r: int = 0,
        **kwargs,
    ):
        super().__init__()
        HRALayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, config=config, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.hra_u.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_weight = orig_weight.view(
                        self.out_features,
                        self.in_features * base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
                    )
                    orig_weight = _right_multiply_hra(
                        orig_weight,
                        self.hra_u[active_adapter],
                        self.hra_apply_GS[active_adapter],
                    )
                    orig_weight = orig_weight.view(
                        self.out_features,
                        self.in_features,
                        base_layer.kernel_size[0],
                        base_layer.kernel_size[0],
                    )

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight.to(orig_dtype)
                else:
                    orig_weight = base_layer.weight.data
                    orig_weight = orig_weight.view(
                        self.out_features,
                        self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
                    )
                    orig_weight = _right_multiply_hra(
                        orig_weight,
                        self.hra_u[active_adapter],
                        self.hra_apply_GS[active_adapter],
                    )
                    orig_weight = orig_weight.view(
                        self.out_features,
                        self.in_features,
                        base_layer.kernel_size[0],
                        base_layer.kernel_size[0],
                    )

                    base_layer.weight.data = orig_weight.to(orig_dtype)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            base_layer = self.get_base_layer()
            orig_dtype = base_layer.weight.dtype
            if active_adapter in self.hra_u.keys():
                orig_weight = base_layer.weight.data.clone()
                orig_weight = orig_weight.view(
                    self.out_features,
                    self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0],
                )
                orig_weight = _right_multiply_hra(
                    orig_weight,
                    self.hra_u[active_adapter],
                    self.hra_apply_GS[active_adapter],
                    reverse=True,
                )
                orig_weight = orig_weight.view(
                    self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                )

                base_layer.weight.data = orig_weight.to(orig_dtype)

    def get_delta_weight(self, adapter_name: str, reverse: bool = False) -> torch.Tensor:
        opt_u = self.hra_u[adapter_name]
        identity = torch.eye(opt_u.shape[0], device=opt_u.device, dtype=opt_u.dtype)
        return _right_multiply_hra(identity, opt_u, self.hra_apply_GS[adapter_name], reverse=reverse)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            orig_weight = self.base_layer.weight.data
            new_weight = orig_weight.view(
                self.out_features,
                self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
            )
            for active_adapter in self.active_adapters:
                if active_adapter not in self.hra_u.keys():
                    continue
                new_weight = _right_multiply_hra(
                    new_weight,
                    self.hra_u[active_adapter],
                    self.hra_apply_GS[active_adapter],
                    cast_input=self.cast_input_dtype_enabled,
                )

            bias = self._cast_input_dtype(self.base_layer.bias, new_weight.dtype)
            new_weight = new_weight.view(
                self.out_features,
                self.in_features,
                self.base_layer.kernel_size[0],
                self.base_layer.kernel_size[0],
            )

            if self.cast_input_dtype_enabled:
                x = self._cast_input_dtype(x, new_weight.dtype)
            else:
                x = x.to(self.get_base_layer().weight.data.dtype)
            result = F.conv2d(
                input=x,
                weight=new_weight,
                bias=bias,
                padding=self.base_layer.padding[0],
                stride=self.base_layer.stride[0],
            )

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hra." + rep
