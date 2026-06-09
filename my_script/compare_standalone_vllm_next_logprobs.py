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

"""Compare standalone vLLM next-token logprobs against NeMo vLLM rollout logprobs."""

from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", required=True)
    parser.add_argument("--token-dump", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    standalone = torch.load(args.standalone, map_location="cpu", weights_only=False)
    token_dump = torch.load(args.token_dump, map_location="cpu", weights_only=False)

    next_token_logprobs = standalone["next_token_logprobs"].to(torch.float32)
    generation_logprobs = token_dump["offline_generation_logprobs"]
    metadata = token_dump["offline_metadata"]

    rows = []
    for idx, item in enumerate(metadata):
        sample_idx = int(item["sample_idx"])
        token_id = int(item["target_token_id"])
        standalone_logprob = float(next_token_logprobs[sample_idx, token_id].item())
        nemo_logprob = float(generation_logprobs[idx])
        rows.append((sample_idx, token_id, standalone_logprob, nemo_logprob))

    diffs = torch.tensor(
        [
            standalone_logprob - nemo_logprob
            for _, _, standalone_logprob, nemo_logprob in rows
        ],
        dtype=torch.float32,
    )
    abs_diff = diffs.abs()
    rel_diff = abs_diff / torch.maximum(
        torch.tensor([max(abs(row[2]), abs(row[3])) for row in rows]),
        torch.full_like(abs_diff, 1e-12),
    )
    print(
        "summary "
        f"tokens={len(rows)} "
        f"mean_abs={float(abs_diff.mean().item()):.8e} "
        f"max_abs={float(abs_diff.max().item()):.8e} "
        f"mean_rel={float(rel_diff.mean().item()):.8e} "
        f"max_rel={float(rel_diff.max().item()):.8e} "
        f"mean_signed={float(diffs.mean().item()):.8e}"
    )
    for sample_idx, token_id, standalone_logprob, nemo_logprob in rows:
        diff = standalone_logprob - nemo_logprob
        print(
            f"sample={sample_idx} token={token_id} "
            f"standalone={standalone_logprob:.8e} "
            f"nemo={nemo_logprob:.8e} diff={diff:.8e}"
        )


if __name__ == "__main__":
    main()
