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

"""Replay NemotronH layer-0 Mamba scan from captured vLLM projected states."""

import argparse
import os

import torch
from compare import diff_stats, get_layer_entry, select_tensor
from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, print_rank_0
from megatron_forward import (
    MODEL_ALIASES,
    NEMOTRON3_NANO_MODEL,
    resolve_model_ref,
    unwrap,
)
from tensor_capture import find_decoder_layers

DEFAULT_SESSION_DIR = os.path.join(
    "session",
    "20260625_121743",
    "nemotron_bf16_no_bi_triton_single_prompt",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nemotron3-nano",
        choices=sorted(MODEL_ALIASES) + [NEMOTRON3_NANO_MODEL],
    )
    parser.add_argument(
        "--vllm-capture",
        default=os.path.join(DEFAULT_SESSION_DIR, "vllm_capture_nemotron3_nano.pt"),
    )
    parser.add_argument(
        "--megatron-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR, "megatron_capture_nemotron3_nano_mambaprefill.pt"
        ),
    )
    parser.add_argument("--layer", type=int, default=0)
    return parser.parse_args()


def load_tensor(payload, layer_idx, name, selectors, *, outputs=False):
    tensor, selector = select_tensor(
        get_layer_entry(payload, layer_idx, name, outputs=outputs),
        selectors,
    )
    if tensor is None:
        raise KeyError(f"missing tensor {name!r} selectors={selectors}")
    return tensor, selector


def packed_to_padded(tensor, seq_lens):
    max_seq_len = max(seq_lens)
    batch = len(seq_lens)
    padded = tensor.new_zeros((max_seq_len, batch, tensor.shape[-1]))
    offset = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        next_offset = offset + seq_len
        padded[:seq_len, batch_idx] = tensor[offset:next_offset]
        offset = next_offset
    return padded


def padded_to_packed(tensor, seq_lens):
    pieces = []
    for batch_idx, seq_len in enumerate(seq_lens):
        pieces.append(tensor[:seq_len, batch_idx])
    return torch.cat(pieces, dim=0).contiguous()


def print_stats(label, actual, expected):
    stats = diff_stats(actual.cpu(), expected.cpu())
    print(
        f"{label:<34s} max={stats['max_abs_diff']:.6f} "
        f"mean={stats['mean_abs_diff']:.6e} cos={stats['cos_sim']:.9f} "
        f"shape={tuple(actual.shape)}"
    )


def first_mamba_call(payload, key):
    calls = payload.get(key, [])
    if not calls:
        return None
    return calls[0]


def mamba_projection_sizes(mixer):
    return (
        mixer.cp.d_inner_local_tpcp,
        mixer.cp.d_inner_local_tpcp + 2 * mixer.cp.ngroups_local_tpcp * mixer.d_state,
        mixer.cp.nheads_local_tpcp,
    )


def split_projected_states(mixer, projected_states):
    return torch.split(projected_states, mamba_projection_sizes(mixer), dim=-1)


def conv1d_weight_2d(mixer, dtype=None):
    weight = mixer.cp.get_conv1d_weight().squeeze(1).contiguous()
    if dtype is not None:
        weight = weight.to(dtype)
    return weight


def conv1d_bias(mixer, dtype=None):
    bias = mixer.cp.get_conv1d_bias()
    if dtype is not None:
        bias = bias.to(dtype)
    return bias


def run_static_conv_prefill(mixer, projected_states, seq_lens, compute_dtype=None):
    from causal_conv1d import causal_conv1d_fn

    projected_lbd = packed_to_padded(projected_states, seq_lens).cuda()
    projected_bld = projected_lbd.transpose(0, 1).contiguous()
    _, xbc_bld, _ = split_projected_states(mixer, projected_bld)
    xbc_bdl = xbc_bld.transpose(1, 2).contiguous()
    xbc_dtype = xbc_bdl.dtype
    if compute_dtype is not None:
        xbc_bdl = xbc_bdl.to(compute_dtype)

    with torch.no_grad():
        xbc_out_bdl = causal_conv1d_fn(
            x=xbc_bdl,
            weight=conv1d_weight_2d(mixer, compute_dtype),
            bias=conv1d_bias(mixer, compute_dtype),
            activation=mixer.activation,
            seq_idx=None,
        )
    xbc_out_bdl = xbc_out_bdl.to(xbc_dtype)
    xbc_out_lbd = xbc_out_bdl.transpose(1, 2).transpose(0, 1).contiguous()
    return padded_to_packed(xbc_out_lbd, seq_lens)


def run_varlen_conv_prefill(mixer, projected_states, seq_lens, conv_dtype=None):
    from megatron.core.ssm.ops.causal_conv1d_varlen import causal_conv1d_varlen_fn

    projected = projected_states.cuda().contiguous()
    _, xbc, _ = split_projected_states(mixer, projected)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=projected.device)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=projected.device),
            torch.cumsum(lens, dim=0),
        ]
    )
    conv_shape, _ = mixer.mamba_state_shapes_per_request()
    if conv_dtype is None:
        conv_dtype = mixer.conv1d.weight.dtype
    conv_state = torch.zeros(
        len(seq_lens),
        *conv_shape,
        dtype=conv_dtype,
        device=projected.device,
    )
    initial_states = conv_state[:, :, 1:]
    xbc_dtype = xbc.dtype

    with torch.no_grad():
        xbc_out = causal_conv1d_varlen_fn(
            x=xbc.to(conv_dtype).contiguous(),
            weight=conv1d_weight_2d(mixer, conv_dtype),
            bias=conv1d_bias(mixer, conv_dtype),
            cu_seqlens=cu_seqlens,
            initial_states=initial_states,
            activation=mixer.activation,
        )
    return xbc_out.to(xbc_dtype).contiguous()


class NormPreInputCapture:
    def __init__(self, norm_module):
        self.values = []
        self.handle = norm_module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):  # noqa: ARG002
        self.values.append(args[0].detach().clone())

    def close(self):
        self.handle.remove()

    def only_value(self):
        if len(self.values) != 1:
            raise RuntimeError(f"expected one norm pre-input, saw {len(self.values)}")
        return self.values[0]


def load_megatron_model(model_ref):
    bridge = AutoBridge.from_hf_pretrained(
        model_ref,
        trust_remote_code=is_safe_repo(trust_remote_code=True, hf_path=model_ref),
    )
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.pipeline_dtype = torch.bfloat16
    provider.gradient_accumulation_fusion = False
    if hasattr(provider, "use_mamba_mem_eff_path"):
        provider.use_mamba_mem_eff_path = False
        print_rank_0("[replay] use_mamba_mem_eff_path=False")
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    model_list = provider.provide_distributed_model(wrap_with_ddp=False)
    model = model_list[0].cuda().eval()
    disable_mtp_for_inference(model)
    return unwrap(model)


def run_static_prefill(mixer, projected_states, seq_lens):
    padded = packed_to_padded(projected_states, seq_lens).cuda()
    capture = NormPreInputCapture(mixer.norm)
    try:
        with torch.no_grad():
            normed = mixer._ssm_prefill(padded, conv_state=None, ssm_state=None)
        y = capture.only_value()
    finally:
        capture.close()
    return padded_to_packed(y, seq_lens), padded_to_packed(normed, seq_lens)


def run_varlen_prefill(mixer, projected_states, seq_lens):
    packed = projected_states.cuda().unsqueeze(1).contiguous()
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=packed.device)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=packed.device),
            torch.cumsum(lens, dim=0),
        ]
    )
    seq_idx = torch.repeat_interleave(
        torch.arange(len(seq_lens), dtype=torch.int32, device=packed.device),
        lens.to(torch.int64),
    ).unsqueeze(0)
    batch_indices = torch.arange(len(seq_lens), dtype=torch.int32, device=packed.device)
    conv_shape, ssm_shape = mixer.mamba_state_shapes_per_request()
    conv_state = torch.zeros(
        len(seq_lens),
        *conv_shape,
        dtype=mixer.conv1d.weight.dtype,
        device=packed.device,
    )
    ssm_state = torch.zeros(
        len(seq_lens), *ssm_shape, dtype=packed.dtype, device=packed.device
    )

    capture = NormPreInputCapture(mixer.norm)
    try:
        with torch.no_grad():
            normed = mixer._ssm_prefill(
                packed,
                conv_state,
                ssm_state,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                batch_indices=batch_indices,
            )
        y = capture.only_value()
    finally:
        capture.close()
    return y.squeeze(1).contiguous(), normed.squeeze(1).contiguous()


def run_megatron_norm(mixer, y, z, seq_lens):
    y_padded = packed_to_padded(y, seq_lens).cuda()
    z_padded = packed_to_padded(z, seq_lens).cuda()
    with torch.no_grad():
        normed = mixer.norm(y_padded, z_padded)
    return padded_to_packed(normed, seq_lens)


def reference_rms_norm_gated(
    y,
    z,
    weight,
    eps,
    group_size,
    *,
    round_gated=False,
    round_norm_before_weight=False,
):
    input_dtype = y.dtype
    y = y.cuda()
    z = z.cuda()
    weight = weight.cuda()
    gated = y.float() * torch.nn.functional.silu(z.float())
    if round_gated:
        gated = gated.to(input_dtype).float()

    *prefix_dims, hidden_dim = gated.shape
    group_count = hidden_dim // group_size
    grouped = gated.reshape(*prefix_dims, group_count, group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normed = grouped * torch.rsqrt(variance + eps)
    normed = normed.reshape(*prefix_dims, hidden_dim)
    if round_norm_before_weight:
        return (weight * normed.to(input_dtype)).to(input_dtype)
    return (normed * weight.float()).to(input_dtype)


def main():
    args = parse_args()
    model_ref = resolve_model_ref(args.model)
    vllm = torch.load(args.vllm_capture, map_location="cpu", weights_only=False)
    seq_lens = [int(v) for v in vllm["seq_lens"]]
    projected_states, projected_selector = load_tensor(
        vllm, args.layer, "mixer.in_proj", ("output", "item0", "first"), outputs=True
    )
    vllm_y, y_selector = load_tensor(
        vllm, args.layer, "mixer.norm", ("arg0", "first"), outputs=False
    )
    vllm_z, z_selector = load_tensor(
        vllm, args.layer, "mixer.norm", ("arg1",), outputs=False
    )
    vllm_normed, normed_selector = load_tensor(
        vllm, args.layer, "mixer.norm", ("output", "item0", "first"), outputs=True
    )
    projected_states = projected_states.reshape(sum(seq_lens), -1).contiguous()
    vllm_y = vllm_y.reshape(sum(seq_lens), -1).contiguous()
    vllm_z = vllm_z.reshape(sum(seq_lens), -1).contiguous()
    vllm_normed = vllm_normed.reshape(sum(seq_lens), -1).contiguous()

    print(
        f"[replay] seq_lens={seq_lens} projected={tuple(projected_states.shape)} "
        f"selectors projected={projected_selector} y={y_selector} z={z_selector} "
        f"normed={normed_selector}"
    )

    model = load_megatron_model(model_ref)
    mixer = find_decoder_layers(model)[args.layer].mixer
    norm_group_size = getattr(mixer.norm, "group_size", vllm_y.shape[-1])
    norm_eps = getattr(mixer.norm, "eps", getattr(mixer.norm, "variance_epsilon", 1e-5))
    print(f"[replay] norm_group_size={norm_group_size} norm_eps={norm_eps}")

    conv_call = first_mamba_call(vllm, "mamba_conv1d_calls")
    if conv_call is not None:
        vllm_conv = conv_call.get("output_token_major")
        if vllm_conv is None:
            conv_output = conv_call.get("output")
            if isinstance(conv_output, torch.Tensor) and conv_output.dim() == 2:
                vllm_conv = conv_output.transpose(0, 1).contiguous()
        if vllm_conv is not None:
            vllm_conv = vllm_conv.reshape(sum(seq_lens), -1).contiguous()
            static_conv = run_static_conv_prefill(mixer, projected_states, seq_lens)
            print_stats("static conv bf16 vs vLLM", static_conv, vllm_conv)
            static_conv_fp32 = run_static_conv_prefill(
                mixer, projected_states, seq_lens, torch.float32
            )
            print_stats("static conv fp32 vs vLLM", static_conv_fp32, vllm_conv)
            varlen_conv = run_varlen_conv_prefill(mixer, projected_states, seq_lens)
            print_stats("varlen conv bf16 vs vLLM", varlen_conv, vllm_conv)
            varlen_conv_fp32 = run_varlen_conv_prefill(
                mixer, projected_states, seq_lens, torch.float32
            )
            print_stats("varlen conv fp32 vs vLLM", varlen_conv_fp32, vllm_conv)
            print_stats("varlen bf16 vs static bf16", varlen_conv, static_conv)
            print_stats(
                "varlen fp32 vs static fp32", varlen_conv_fp32, static_conv_fp32
            )
        else:
            print("[replay] vLLM Mamba conv capture has no output tensor")
    else:
        print("[replay] no vLLM Mamba conv capture found")

    scan_call = first_mamba_call(vllm, "mamba_scan_calls")
    if scan_call is not None and scan_call.get("out_after") is not None:
        vllm_scan = scan_call["out_after"].reshape(sum(seq_lens), -1).contiguous()
        print_stats("vLLM scan capture vs norm y", vllm_scan, vllm_y)
    elif scan_call is None:
        print("[replay] no vLLM Mamba scan capture found")

    static_y, static_normed = run_static_prefill(mixer, projected_states, seq_lens)
    print_stats("static y vs vLLM y", static_y, vllm_y)
    print_stats("static normed vs vLLM normed", static_normed, vllm_normed)
    megatron_norm_vllm_yz = run_megatron_norm(mixer, vllm_y, vllm_z, seq_lens)
    print_stats("Megatron norm(vLLM y,z)", megatron_norm_vllm_yz, vllm_normed)
    print_stats(
        "Megatron norm(vLLM y,z) vs static", megatron_norm_vllm_yz, static_normed
    )
    ref_norm = reference_rms_norm_gated(
        vllm_y,
        vllm_z,
        mixer.norm.weight,
        norm_eps,
        norm_group_size,
    )
    print_stats("ref norm fp32-weight vs vLLM", ref_norm, vllm_normed)
    print_stats("ref norm fp32-weight vs Megatron", ref_norm, megatron_norm_vllm_yz)
    ref_norm_native = reference_rms_norm_gated(
        vllm_y,
        vllm_z,
        mixer.norm.weight,
        norm_eps,
        norm_group_size,
        round_norm_before_weight=True,
    )
    print_stats("ref norm native-round vs vLLM", ref_norm_native, vllm_normed)
    print_stats(
        "ref norm native-round vs Megatron", ref_norm_native, megatron_norm_vllm_yz
    )
    ref_norm_gated_round = reference_rms_norm_gated(
        vllm_y,
        vllm_z,
        mixer.norm.weight,
        norm_eps,
        norm_group_size,
        round_gated=True,
    )
    print_stats("ref norm gated-round vs vLLM", ref_norm_gated_round, vllm_normed)
    print_stats(
        "ref norm gated-round vs Megatron", ref_norm_gated_round, megatron_norm_vllm_yz
    )

    varlen_y, varlen_normed = run_varlen_prefill(mixer, projected_states, seq_lens)
    print_stats("varlen y vs vLLM y", varlen_y, vllm_y)
    print_stats("varlen normed vs vLLM normed", varlen_normed, vllm_normed)
    print_stats("varlen y vs static y", varlen_y, static_y)
    print_stats("varlen normed vs static normed", varlen_normed, static_normed)

    if args.megatron_capture and os.path.exists(args.megatron_capture):
        megatron = torch.load(
            args.megatron_capture, map_location="cpu", weights_only=False
        )
        megatron_y, _ = load_tensor(
            megatron, args.layer, "mixer.norm", ("arg0", "first"), outputs=False
        )
        megatron_y = megatron_y.reshape(sum(seq_lens), -1).contiguous()
        print_stats("static y vs saved Megatron y", static_y, megatron_y)
        print_stats("varlen y vs saved Megatron y", varlen_y, megatron_y)


if __name__ == "__main__":
    main()
