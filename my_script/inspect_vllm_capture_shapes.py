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

"""Print selected vLLM debug capture tensor shapes."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", required=True)
    parser.add_argument("--nemo-glob", required=True)
    parser.add_argument(
        "--modules",
        nargs="+",
        default=[
            "<layer>",
            "self_attn.rotary_emb",
            "self_attn.attn",
            "self_attn.o_proj",
        ],
    )
    return parser.parse_args()


def load_capture(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "capture" in payload:
        return payload["capture"]
    return payload


def layer0_calls(capture: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    calls_by_layer = capture["module_input_calls_by_layer"]
    return calls_by_layer.get(0) or calls_by_layer.get("0")


def tensor_summary(tensor: torch.Tensor) -> tuple[Any, str, Any]:
    values = ""
    if tensor.ndim == 1:
        values = tensor[:32].tolist()
    return tuple(tensor.shape), str(tensor.dtype), values


def summarize_call(call: dict[str, Any]) -> dict[str, Any]:
    args = [
        tensor_summary(value)
        for value in call.get("args", ())
        if isinstance(value, torch.Tensor)
    ]
    kwargs = {
        key: tensor_summary(value)
        for key, value in call.get("kwargs", {}).items()
        if isinstance(value, torch.Tensor)
    }
    return {"args": args, "kwargs": kwargs}


def main() -> None:
    args = parse_args()
    standalone_capture = load_capture(args.standalone)
    standalone_calls = layer0_calls(standalone_capture)
    print("standalone modules:", list(standalone_calls))
    for module in args.modules:
        print(f"STANDALONE module={module}")
        for idx, call in enumerate(standalone_calls.get(module, [])):
            print(f"  call={idx} {summarize_call(call)}")

    for nemo_file in sorted(glob.glob(args.nemo_glob)):
        nemo_capture = load_capture(nemo_file)
        nemo_calls = layer0_calls(nemo_capture)
        print(f"NEMO file={Path(nemo_file).name}")
        for module in args.modules:
            print(f"  module={module}")
            for idx, call in enumerate(nemo_calls.get(module, [])):
                print(f"    call={idx} {summarize_call(call)}")


if __name__ == "__main__":
    main()
