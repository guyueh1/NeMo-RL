# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest
import torch

from nemo_rl.models.generation.vllm.quantization.mxfp4_moe import (
    quantize_mxfp4_weight,
)
from nemo_rl.models.quantization.mxfp4 import (
    fake_quantize_mxfp4,
    is_routed_moe_weight_name,
)


def test_fake_quantize_mxfp4_rounds_to_e2m1_levels() -> None:
    values = torch.tensor([0.1, 0.4, 0.9, 1.4, 1.9, 2.6, 3.7, 5.8], dtype=torch.float32)
    weight = values.repeat(4)

    result = fake_quantize_mxfp4(weight)

    expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.equal(result[:8], expected)


def test_fake_quantize_mxfp4_uses_power_of_two_block_scale() -> None:
    weight = torch.full((32,), 12.0, dtype=torch.bfloat16)

    result = fake_quantize_mxfp4(weight)

    # 12 / E2M1_MAX(6) selects scale 2.
    assert torch.equal(result, weight)


def test_fake_quantize_mxfp4_preserves_results_across_chunks(monkeypatch) -> None:
    from nemo_rl.models.quantization import mxfp4

    weight = torch.cat(
        [
            torch.linspace(-6, 6, 32),
            torch.linspace(-12, 12, 32),
            torch.linspace(-3, 3, 32),
        ]
    )
    expected = fake_quantize_mxfp4(weight)
    monkeypatch.setattr(mxfp4, "_MXFP4_BLOCKS_PER_CHUNK", 1)

    result = fake_quantize_mxfp4(weight)

    assert torch.equal(result, expected)


def test_fake_quantize_mxfp4_rejects_unaligned_weights() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        fake_quantize_mxfp4(torch.ones(2, 31))


def test_native_mxfp4_pack_matches_fake_quant_numerics() -> None:
    weight = torch.linspace(-12, 12, 64, dtype=torch.float32).reshape(2, 32)

    packed, encoded_scale = quantize_mxfp4_weight(weight)

    low = packed & 0xF
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape_as(weight)
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )[(codes & 0x7).long()]
    values = torch.where((codes & 0x8) != 0, -magnitude, magnitude)
    scale = torch.exp2(encoded_scale.to(torch.float32) - 127).repeat_interleave(
        32, dim=-1
    )

    torch.testing.assert_close(values * scale, fake_quantize_mxfp4(weight))
    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 16)
    assert encoded_scale.shape == (2, 1)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", True),
        ("decoder.layers.0.mlp.experts.linear_fc2.weight", True),
        ("model.layers.0.mlp.shared_experts.down_proj.weight", False),
        ("model.layers.0.self_attn.q_proj.weight", False),
        ("model.layers.0.mlp.experts.gate", False),
    ],
)
def test_is_routed_moe_weight_name(name: str, expected: bool) -> None:
    assert is_routed_moe_weight_name(name) is expected


def test_register_mxfp4_moe_weight_hooks_targets_only_expert_fc(monkeypatch) -> None:
    from nemo_rl.models.megatron import mxfp4_fake_quant

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

    model = ToyModel()
    seen = []
    monkeypatch.setattr(
        mxfp4_fake_quant,
        "_fake_quantize_mxfp4_weight_",
        lambda weight: seen.append(weight),
    )

    handles = mxfp4_fake_quant.register_mxfp4_moe_weight_hooks(model)
    x = torch.ones(1, 32)
    model.decoder.layers[0].mlp.experts.linear_fc1(x)
    model.decoder.layers[0].mlp.experts.linear_fc2(x)
    model.decoder.layers[0].mlp.shared_experts.linear_fc1(x)

    assert len(handles) == 2
    assert seen == [
        model.decoder.layers[0].mlp.experts.linear_fc1.weight,
        model.decoder.layers[0].mlp.experts.linear_fc2.weight,
    ]


def test_register_mxfp4_moe_weight_hooks_supports_grouped_linear_weights(
    monkeypatch,
) -> None:
    from nemo_rl.models.megatron import mxfp4_fake_quant

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

    model = ToyModel()
    seen = []
    monkeypatch.setattr(
        mxfp4_fake_quant,
        "_fake_quantize_mxfp4_weight_",
        lambda weight: seen.append(weight),
    )

    handles = mxfp4_fake_quant.register_mxfp4_moe_weight_hooks(model)
    x = torch.ones(1, 32)
    model.experts.linear_fc1(x)
    model.experts.linear_fc2(x)

    assert len(handles) == 2
    assert seen == [
        model.experts.linear_fc1.weight0,
        model.experts.linear_fc1.weight1,
        model.experts.linear_fc2.weight0,
        model.experts.linear_fc2.weight1,
    ]
