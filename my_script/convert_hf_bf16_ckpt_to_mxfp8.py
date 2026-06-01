"""This script converts a Hugging Face checkpoint format. Two modes are supported: BF16 to deepseek FP8 and BF16 to MXFP8."""

import os
import argparse
import torch
import safetensors
from transformers import AutoModelForCausalLM
from typing import Dict
from huggingface_hub import save_torch_state_dict
import json
from safetensors.torch import safe_open
from os.path import exists

def cast_tensor_to_fp8_blockwise(data: torch.Tensor):
    assert len(data.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = 128
    block_size0 = 128
    shape_before_padding = data.shape
    # pad data to make its shape a multiple of weight_block_size with the last element of data
    if data.shape[1] % block_size1 != 0 or data.shape[0] % block_size0 != 0:
        pad1 = (
            0
            if data.shape[1] % block_size1 == 0
            else block_size1 - data.shape[1] % block_size1
        )
        pad0 = (
            0
            if data.shape[0] % block_size0 == 0
            else block_size0 - data.shape[0] % block_size0
        )
        print(
            f"Padding data from {data.shape} to {(data.shape[0] + pad0, data.shape[1] + pad1)}"
        )
        data = torch.nn.functional.pad(
            data, (0, pad1, 0, pad0), mode="constant", value=data[-1, -1]
        )

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    shape_after_padding = data.shape
    blk_m, blk_n = data.shape[0] // block_size0, data.shape[1] // block_size1

    data = data.reshape(blk_m, block_size0, blk_n, block_size1)

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data = data.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data = data.to(torch.float32).contiguous().flatten(start_dim=2)

    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data), dim=-1, keepdim=True)

    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)

    descale_fp = torch.reciprocal(scale_fp)
    descale_fp = descale_fp.reshape(blk_m, blk_n)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(shape_after_padding)
    )

    # remove the padding
    if data.shape != shape_before_padding:
        fp_data = fp_data[: shape_before_padding[0], : shape_before_padding[1]]

    # Convert to target format, but still in original precision container
    return fp_data, descale_fp


def cast_tensor_to_mxfp8(data: torch.Tensor):
    assert len(data.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = 32
    block_size0 = 1
    shape_before_padding = data.shape
    # pad data to make its shape a multiple of weight_block_size with the last element of data
    assert data.shape[1] % block_size1 == 0 and data.shape[0] % block_size0 == 0, "Data shape must be a multiple of tile size [1, 32]"

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    shape_after_padding = data.shape
    blk_m, blk_n = data.shape[0] // block_size0, data.shape[1] // block_size1

    data = data.reshape(blk_m, block_size0, blk_n, block_size1)

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data = data.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data = data.to(torch.float32).contiguous().flatten(start_dim=2)

    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data), dim=-1, keepdim=True)

    # Calculate scales
    descale = max_abs / max_dtype
    exponent = torch.ceil(torch.log2(descale))
    # Post process exponent to be in range of -127 to 127 and to be E8M0 biased
    exponent = torch.clamp(exponent, min=-127, max=127) + 127
    # Convert to uint8 container
    exponent = exponent.to(torch.uint8)
    # Calculate descale_fp to apply to data_hp
    scale_fp = torch.where(
        # If exponent is 0, descale_fp is 1.0 rather than 2^127
        exponent == 0,
        1.0,
        torch.exp2(127 - exponent.to(torch.float32)),
    )
    exponent = exponent.reshape(blk_m, blk_n)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(shape_after_padding)
    )

    # remove the padding
    if data.shape != shape_before_padding:
        fp_data = fp_data[: shape_before_padding[0], : shape_before_padding[1]]

    # Convert to target format, but still in original precision container
    return fp_data, exponent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-hf-path", 
        type=str,
        required=True
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--input-dtype", type=str, choices=["bfloat16"], default="bfloat16", help="The dtype of the input checkpoint. Supported values: bfloat16.")
    parser.add_argument("--output-dtype", type=str, choices=["deepseek-fp8", "mxfp8"], default="mxfp8", help="The dtype of the output checkpoint. Supported values: deepseek-fp8, mxfp8.")
    return parser.parse_args()

def need_quantize(hf_param_name: str, tensor: torch.Tensor, output_dtype: str, mxfp8_quantize_only_expert_weights: bool) -> bool:
    """Check if a tensor needs to be quantized."""
    if (
        output_dtype == "mxfp8" 
        and mxfp8_quantize_only_expert_weights
        and not (".experts." in hf_param_name)
    ):
        return False
    if not hf_param_name.endswith(".weight"):
        return False
    if ".gate." in hf_param_name:
        return False
    if "lm_head" in hf_param_name:
        return False
    if "router." in hf_param_name:
        return False
    if "layernorm" in hf_param_name:
        return False
    if "embedding" in hf_param_name:
        return False
    if tensor.ndim != 2:
        return False
    if "_proj." not in hf_param_name:
        return False

    return True

def maybe_quantize_tensor(
    hf_param_name: str,
    tensor: torch.Tensor,
    output_dtype: str,
    mxfp8_quantize_only_expert_weights: bool,
) -> Dict[str, torch.Tensor]:
    """Quantize a tensor to the specified dtype."""

    if not need_quantize(hf_param_name, tensor, output_dtype, mxfp8_quantize_only_expert_weights):
        return {
            hf_param_name: tensor,
        }

    tensor = tensor.to('cuda')
    if output_dtype == "deepseek-fp8":
        data, scale = cast_tensor_to_fp8_blockwise(tensor)
        data = data.cpu()
        scale = scale.cpu()
        assert data.dtype == torch.float8_e4m3fn, "Data dtype should be float8_e4m3fn"
        assert scale.dtype == torch.float32, "Scale dtype should be float32"
        return {
            hf_param_name: data,
            hf_param_name + "_scale": scale,
        }
    elif output_dtype == "mxfp8":
        data, scale = cast_tensor_to_mxfp8(tensor)
        data = data.cpu()
        scale = scale.cpu()
        assert data.dtype == torch.float8_e4m3fn, "Data dtype should be float8_e4m3fn"
        assert scale.dtype == torch.uint8, "Scale dtype should be uint8"
        return {
            hf_param_name: data,
            hf_param_name + "_scale": scale,
        }
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")

def convert_config_to_mxfp8(output_dir: str, input_hf_path: str):
    """Convert the config.json to mxfp8."""
    config = json.load(open(os.path.join(input_hf_path, "config.json")))
    
    config["quantization_config"] = {
        "quant_algo": "MXFP8",
        "quant_method": "modelopt",
    }

    # load the safetensors config.json, add all conv1d layers to the ignore list
    # this is hacky and only for nano/super-v3
    if exists(os.path.join(input_hf_path, "model.safetensors.index.json")):
        safetensors_config = json.load(open(os.path.join(input_hf_path, "model.safetensors.index.json")))
        param_names = safetensors_config["weight_map"].keys()
    else:
        with safe_open(os.path.join(input_hf_path, "model.safetensors"), framework="pt") as f:
            param_names = f.keys()
    ignore_list = set()
    for key in param_names:
        if "conv1d" in key:
            ignore_list.add(key.removesuffix(".weight").removesuffix(".bias"))
    config["quantization_config"]["ignore"] = list(ignore_list) 

    json.dump(config, open(os.path.join(output_dir, "config.json"), "w"), indent=4)

def main():
    args = parse_args()

    # get the state dict from the hf checkpoint
    print(f"Loading model from {args.input_hf_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.input_hf_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    state_dict = model.state_dict()
    print(f"Loaded {len(state_dict)} tensors from checkpoint.")

    # create a new state dict for output hf checkpoint
    new_state_dict = {}

    # for each tensor in the state dict, move it to cuda, quantize, and save it in the new state dict
    print(f"Quantizing tensors to {args.output_dtype}...")
    for name, tensor in state_dict.items():
        print(f"Processing {name} with shape {tensor.shape}...")
        quantized_tensors = maybe_quantize_tensor(name, tensor, args.output_dtype, False)
        for key in quantized_tensors.keys():
            print(f"Saving {key} with shape {quantized_tensors[key].shape}...")
        new_state_dict.update(quantized_tensors) # Move back to CPU for saving

    # save the new state dict to the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    save_torch_state_dict(
        new_state_dict, 
        save_directory=args.output_dir,
        max_shard_size="5GB",
    )

    if args.output_dtype == "mxfp8":
        convert_config_to_mxfp8(args.output_dir, args.input_hf_path)
    else:
        pass
    print("Conversion complete!")
    print("!!!Note: Please manually copy the tokenizer related files from old checkpoint folder to new one")


if __name__ == "__main__":
    main()