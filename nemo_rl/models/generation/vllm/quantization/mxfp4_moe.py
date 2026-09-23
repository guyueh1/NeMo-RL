# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Online OCP MXFP4 MoE weights with dynamic MXFP8 activations for vLLM.

Dense layers continue to use vLLM's ModelOpt MXFP8 implementation. Routed
experts are restored in their logical BF16 layout, quantized to packed E2M1
plus block-32 E8M0 scales after loading, and executed by FlashInfer CUTLASS.
vLLM's layerwise reload API repeats this conversion after every policy refit.
"""

from typing import Any

import torch
import torch.nn.functional as F

NEMO_MXFP4_MOE_MXFP8 = "nemo_mxfp4_moe_mxfp8"
MXFP4_BLOCK_SIZE = 32
_registered = False


def _quantize_stacked_experts_for_cutlass(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize experts with FlashInfer into packed CUTLASS-ready storage."""
    from flashinfer import mxfp4_quantize

    quantized = []
    scales = []
    for expert_weight in weight.unbind(0):
        expert_quantized, expert_scales = mxfp4_quantize(expert_weight.contiguous())
        quantized.append(expert_quantized)
        scales.append(expert_scales)
    return torch.stack(quantized), torch.stack(scales)


def register_nemo_mxfp4_moe_mxfp8() -> None:
    """Register the mixed MXFP8-dense/MXFP4-MoE vLLM quantization config."""
    global _registered
    if _registered:
        return

    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe import (
        FusedMoEMethodBase,
        RoutedExperts,
        SharedExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        make_mxfp4_moe_kernel,
        make_mxfp4_moe_quant_config,
        mxfp4_round_up_hidden_size_and_intermediate_size,
        select_mxfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
    from vllm.model_executor.model_loader.reload.layerwise import (
        initialize_online_processing,
    )
    from vllm.model_executor.utils import replace_parameter, set_weight_attrs

    class NemoMxfp4MoeMethod(FusedMoEMethodBase):
        """Load BF16 experts and materialize native W4A8 CUTLASS weights."""

        # Tell the dummy loader to leave the logical BF16 parameters on meta.
        # It will materialize and process one expert layer at a time instead of
        # allocating the whole unquantized MoE before post-load quantization.
        uses_meta_device: bool = True

        def __init__(self, quant_config: Any, moe_config: Any) -> None:
            super().__init__(moe_config)
            self.quant_config = quant_config
            self.weight_dtype = "mxfp4"
            self.weight_block_size = [1, MXFP4_BLOCK_SIZE]
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
                config=self.moe, activation_key=kMxfp8Dynamic
            )
            if self.mxfp4_backend != Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8:
                raise ValueError(
                    "Native MXFP4-MoE/MXFP8-activation refit requires "
                    "moe_backend='flashinfer_cutlass_afp8'."
                )
            self.moe_kernel: mk.FusedMoEKernel | None = None
            self.moe_quant_config = None
            self.logical_hidden_size = 0
            self.logical_intermediate_size_per_partition = 0

        def maybe_roundup_sizes(
            self,
            hidden_size: int,
            intermediate_size_per_partition: int,
            act_dtype: torch.dtype,
            moe_parallel_config: Any,
        ) -> tuple[int, int]:
            hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                act_dtype=act_dtype,
                moe_parallel_config=moe_parallel_config,
            )
            # RoutedExperts mutates its shared FusedMoEConfig to the padded
            # dimensions immediately after this call. Preserve the checkpoint
            # layout here so online loading and every reload reconstruct the
            # logical BF16 tensors rather than expecting padded checkpoint data.
            self.logical_hidden_size = hidden_size
            self.logical_intermediate_size_per_partition = (
                intermediate_size_per_partition
            )
            return mxfp4_round_up_hidden_size_and_intermediate_size(
                self.mxfp4_backend, hidden_size, intermediate_size_per_partition
            )

        def create_weights(
            self,
            layer: RoutedExperts,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            if hidden_size % MXFP4_BLOCK_SIZE != 0:
                raise ValueError(
                    f"MXFP4 requires hidden size divisible by 32: {hidden_size}"
                )
            if intermediate_size_per_partition % MXFP4_BLOCK_SIZE != 0:
                raise ValueError(
                    "MXFP4 requires intermediate size divisible by 32: "
                    f"{intermediate_size_per_partition}"
                )

            layer.num_experts = num_experts
            layer.params_dtype = params_dtype
            layer.orig_dtype = params_dtype
            layer.weight_block_size = None
            w13_num_shards = 2 if self.moe.is_act_and_mul else 1
            weight_loader = extra_weight_attrs.get("weight_loader")
            logical_hidden_size = self.logical_hidden_size
            logical_intermediate_size = self.logical_intermediate_size_per_partition

            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    w13_num_shards * logical_intermediate_size,
                    logical_hidden_size,
                    device="meta",
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)

            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    logical_hidden_size,
                    logical_intermediate_size,
                    device="meta",
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)

            if self.moe.has_bias:
                for name, shape in (
                    (
                        "w13_bias",
                        (num_experts, w13_num_shards * logical_intermediate_size),
                    ),
                    ("w2_bias", (num_experts, logical_hidden_size)),
                ):
                    bias = torch.nn.Parameter(
                        torch.zeros(shape, device="meta", dtype=params_dtype),
                        requires_grad=False,
                    )
                    layer.register_parameter(name, bias)
                    set_weight_attrs(bias, extra_weight_attrs)

            layer.w13_input_scale = None
            layer.w2_input_scale = None
            # Preserve the grouped expert loader exactly as vLLM supplied it.
            if weight_loader is not None:
                layer.w13_weight.weight_loader = weight_loader
                layer.w2_weight.weight_loader = weight_loader
            initialize_online_processing(layer)

        def _pad_logical_weights(
            self, layer: RoutedExperts
        ) -> tuple[torch.Tensor, torch.Tensor]:
            hidden_pad = layer.hidden_size - layer.w13_weight.shape[2]
            intermediate_pad = (
                layer.intermediate_size_per_partition - layer.w2_weight.shape[2]
            )
            w13_shards = 2 if self.moe.is_act_and_mul else 1
            w13 = layer.w13_weight
            if intermediate_pad:
                w13 = torch.cat(
                    [
                        F.pad(shard, (0, 0, 0, intermediate_pad))
                        for shard in torch.chunk(w13, w13_shards, dim=1)
                    ],
                    dim=1,
                )
            if hidden_pad:
                w13 = F.pad(w13, (0, hidden_pad))
            w2 = F.pad(layer.w2_weight, (0, intermediate_pad, 0, hidden_pad))
            return w13, w2

        def _setup_kernel(
            self,
            layer: RoutedExperts,
            w13: torch.Tensor,
            w2: torch.Tensor,
            w13_scale: torch.Tensor,
            w2_scale: torch.Tensor,
        ) -> None:
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)
            if self.moe.is_act_and_mul:
                if w13_bias is not None:
                    b1, b3 = torch.chunk(w13_bias, 2, dim=1)
                    w13_bias = torch.cat((b3, b1), dim=1).contiguous()

            if w13_bias is not None:
                logical_intermediate = self.logical_intermediate_size_per_partition
                intermediate_pad = (
                    layer.intermediate_size_per_partition - logical_intermediate
                )
                if intermediate_pad:
                    w13_bias = torch.cat(
                        [
                            F.pad(shard, (0, intermediate_pad))
                            for shard in torch.chunk(
                                w13_bias, 2 if self.moe.is_act_and_mul else 1, dim=1
                            )
                        ],
                        dim=1,
                    )
            if w2_bias is not None and w2_bias.shape[1] < layer.hidden_size:
                w2_bias = F.pad(w2_bias, (0, layer.hidden_size - w2_bias.shape[1]))

            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2.contiguous())
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
            if w13_bias is not None:
                replace_parameter(layer, "w13_bias", w13_bias)
            if w2_bias is not None:
                replace_parameter(layer, "w2_bias", w2_bias)

            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            assert self.experts_cls is not None
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

        def get_fused_moe_quant_config(self, layer: RoutedExperts) -> Any:
            return make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None) or 1.0,
                gemm1_beta=getattr(layer, "swiglu_beta", None) or 0.0,
                swiglu_limit=getattr(layer, "swiglu_limit", None) or 0.0,
                layer=layer,
            )

        @torch.no_grad()
        def process_weights_after_loading(self, layer: RoutedExperts) -> None:
            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return
            reload_kernel = self.moe_kernel
            reload_quant_config = self.moe_quant_config
            logical_w13, logical_w2 = self._pad_logical_weights(layer)
            if self.moe.is_act_and_mul:
                # vLLM loads W13; FlashInfer CUTLASS consumes W31. Swap before
                # quantization so FlashInfer can directly emit swizzled scales.
                w1, w3 = torch.chunk(logical_w13, 2, dim=1)
                logical_w13 = torch.cat((w3, w1), dim=1).contiguous()
            w13, w13_scale = _quantize_stacked_experts_for_cutlass(logical_w13)
            w2, w2_scale = _quantize_stacked_experts_for_cutlass(logical_w2)
            self._setup_kernel(layer, w13, w2, w13_scale, w2_scale)
            if reload_kernel is not None:
                # Layerwise reload copies the newly packed tensors into the old,
                # CUDA-graph-stable storage. Keep the kernel/config references
                # that already point at that storage.
                self.moe_kernel = reload_kernel
                self.moe_quant_config = reload_quant_config
            layer.weight_block_size = self.weight_block_size
            layer._already_called_process_weights_after_loading = True

        @property
        def supports_eplb(self) -> bool:
            return True

        def apply(
            self,
            layer: RoutedExperts,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts: SharedExperts | None,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            assert not self.is_monolithic and self.moe_kernel is not None
            return self.moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

        def apply_monolithic(
            self,
            layer: RoutedExperts,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del input_ids
            assert self.is_monolithic and self.moe_kernel is not None
            return self.moe_kernel.apply_monolithic(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                router_logits=router_logits,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                e_score_correction_bias=layer.e_score_correction_bias,
                routed_scaling_factor=layer.routed_scaling_factor,
            )

    class NemoMxfp4MoeMxfp8Config(ModelOptMxFp8Config):
        FusedMoEMethodCls = NemoMxfp4MoeMethod

        def get_name(self) -> str:
            return NEMO_MXFP4_MOE_MXFP8

        @classmethod
        def override_quantization_method(
            cls,
            hf_quant_cfg: dict[str, Any],
            user_quant: str | None,
            hf_config: Any = None,
        ) -> str | None:
            del hf_config
            if user_quant != NEMO_MXFP4_MOE_MXFP8:
                return None
            algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
            return NEMO_MXFP4_MOE_MXFP8 if algo == "MXFP8" else None

    register_quantization_config(NEMO_MXFP4_MOE_MXFP8)(NemoMxfp4MoeMxfp8Config)
    _registered = True
