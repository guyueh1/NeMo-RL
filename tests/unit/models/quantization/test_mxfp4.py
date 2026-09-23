# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import sys
from types import ModuleType

import pytest
import torch

from nemo_rl.models.quantization.mxfp4 import quantize_dequantize_mxfp4


def test_quantize_dequantize_mxfp4_rounds_to_e2m1_levels() -> None:
    values = torch.tensor([0.1, 0.4, 0.9, 1.4, 1.9, 2.6, 3.7, 5.8], dtype=torch.float32)
    weight = values.repeat(4)

    result = quantize_dequantize_mxfp4(weight)

    expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.equal(result[:8], expected)


def test_quantize_dequantize_mxfp4_uses_power_of_two_block_scale() -> None:
    weight = torch.full((32,), 12.0, dtype=torch.bfloat16)

    result = quantize_dequantize_mxfp4(weight)

    # 12 / E2M1_MAX(6) selects scale 2.
    assert torch.equal(result, weight)


def test_quantize_dequantize_mxfp4_preserves_results_across_chunks(
    monkeypatch,
) -> None:
    from nemo_rl.models.quantization import mxfp4

    weight = torch.cat(
        [
            torch.linspace(-6, 6, 32),
            torch.linspace(-12, 12, 32),
            torch.linspace(-3, 3, 32),
        ]
    )
    expected = quantize_dequantize_mxfp4(weight)
    monkeypatch.setattr(mxfp4, "_MXFP4_BLOCKS_PER_CHUNK", 1)

    result = quantize_dequantize_mxfp4(weight)

    assert torch.equal(result, expected)


def test_quantize_dequantize_mxfp4_rejects_unaligned_weights() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        quantize_dequantize_mxfp4(torch.ones(2, 31))


def test_native_mxfp4_uses_flashinfer_quantizer(monkeypatch) -> None:
    from nemo_rl.models.generation.vllm.quantization.mxfp4_moe import (
        _quantize_stacked_experts_for_cutlass,
    )

    calls = []
    fake_flashinfer = ModuleType("flashinfer")

    def mxfp4_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(weight)
        return weight.to(torch.uint8), torch.ones(weight.shape[0], 1, dtype=torch.uint8)

    fake_flashinfer.mxfp4_quantize = mxfp4_quantize
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)

    weight = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    quantized, scales = _quantize_stacked_experts_for_cutlass(weight)

    assert len(calls) == 2
    assert torch.equal(calls[0], weight[0])
    assert torch.equal(calls[1], weight[1])
    assert quantized.shape == weight.shape
    assert scales.shape == (2, 2, 1)


def test_register_mxfp4_moe_weight_hooks_targets_only_expert_fc(monkeypatch) -> None:
    from nemo_rl.models.megatron import mxfp4_runtime

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
            mlp = torch.nn.Module()
            mlp.experts = torch.nn.Module()
            mlp.experts.linear_fc1 = torch.nn.Linear(32, 32, bias=False)
            mlp.experts.linear_fc2 = torch.nn.Linear(32, 32, bias=False)
            mlp.shared_experts = torch.nn.Module()
            mlp.shared_experts.linear_fc1 = torch.nn.Linear(32, 32, bias=False)
            self.decoder.layers[0].mlp = mlp
            self.mtp = torch.nn.Module()
            self.mtp.layers = torch.nn.ModuleList([torch.nn.Module()])
            mtp_mlp = torch.nn.Module()
            mtp_mlp.experts = torch.nn.Module()
            mtp_mlp.experts.linear_fc1 = torch.nn.Linear(32, 32, bias=False)
            mtp_mlp.experts.linear_fc2 = torch.nn.Linear(32, 32, bias=False)
            self.mtp.layers[0].mlp = mtp_mlp

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            mlp = self.decoder.layers[0].mlp
            value = mlp.experts.linear_fc1(value)
            value = mlp.experts.linear_fc2(value)
            return mlp.shared_experts.linear_fc1(value)

    model = ToyModel()
    seen = []
    monkeypatch.setattr(
        mxfp4_runtime,
        "_quantize_dequantize_mxfp4_weight_",
        lambda weight: seen.append(weight),
    )

    hook = mxfp4_runtime.register_mxfp4_moe_weight_hooks(model)
    x = torch.ones(1, 32)
    model(x)
    model(x)

    assert seen == [
        model.decoder.layers[0].mlp.experts.linear_fc1.weight,
        model.decoder.layers[0].mlp.experts.linear_fc2.weight,
    ]
    assert all(
        weight is not model.mtp.layers[0].mlp.experts.linear_fc1.weight
        for weight in seen
    )
    assert all(
        weight is not model.mtp.layers[0].mlp.experts.linear_fc2.weight
        for weight in seen
    )
    hook.arm()
    model(x)
    assert len(seen) == 4


def test_patch_mcore_language_loss_clones_detached_mtp_logits_and_is_idempotent(
    monkeypatch,
) -> None:
    from nemo_rl.models.megatron import mxfp4_runtime

    class FakeLanguageModule:
        def compute_language_model_loss(self, labels, logits):
            del labels
            return logits

    fake_module = ModuleType(
        "megatron.core.models.common.language_module.language_module"
    )
    fake_module.LanguageModule = FakeLanguageModule
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    original = FakeLanguageModule.compute_language_model_loss
    mxfp4_runtime.patch_mcore_language_loss_for_detached_mtp_logits()
    patched = FakeLanguageModule.compute_language_model_loss
    mxfp4_runtime.patch_mcore_language_loss_for_detached_mtp_logits()

    logits = torch.ones(2, requires_grad=True)
    cloned = FakeLanguageModule().compute_language_model_loss(None, logits)

    assert patched is FakeLanguageModule.compute_language_model_loss
    assert patched is not original
    assert cloned.data_ptr() != logits.data_ptr()
    cloned.sum().backward()
    assert torch.equal(logits.grad, torch.ones_like(logits))


def test_register_mxfp4_moe_weight_hooks_supports_grouped_linear_weights(
    monkeypatch,
) -> None:
    from nemo_rl.models.megatron import mxfp4_runtime

    class GroupedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight0 = torch.nn.Parameter(torch.ones(32, 32))
            self.weight1 = torch.nn.Parameter(torch.ones(32, 32))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = torch.nn.Module()
            self.experts.linear_fc1 = GroupedLinear()
            self.experts.linear_fc2 = GroupedLinear()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.experts.linear_fc2(self.experts.linear_fc1(value))

    model = ToyModel()
    seen = []
    monkeypatch.setattr(
        mxfp4_runtime,
        "_quantize_dequantize_mxfp4_weight_",
        lambda weight: seen.append(weight),
    )

    hook = mxfp4_runtime.register_mxfp4_moe_weight_hooks(model)
    x = torch.ones(1, 32)
    model(x)

    assert seen == [
        model.experts.linear_fc1.weight0,
        model.experts.linear_fc1.weight1,
        model.experts.linear_fc2.weight0,
        model.experts.linear_fc2.weight1,
    ]
    hook.remove()
