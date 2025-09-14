import sys, json
import torch
from nemo_rl.algorithms.utils import (
    masked_mean
)

def get_logprob_and_mask(json_path):
    with open(json_path, "r") as f:
        data = json.loads(f.readline())
    data = data['logprobs'][0]
    data = torch.tensor(data)
    mask = (data != 0)
    return data, mask

if __name__ == "__main__":
    logprob1, mask1 = get_logprob_and_mask(sys.argv[1])
    logprob2, mask2 = get_logprob_and_mask(sys.argv[2])
    assert torch.all(mask1 == mask2)
    lp_error = torch.abs(logprob1 - logprob2)
    mult_prob_error = torch.exp(lp_error*mask1)
    avg_mult_prob_error = masked_mean(mult_prob_error, mask1).item()
    print(f"Avg Mult prob error: {avg_mult_prob_error}")
    print(f"LP error: {lp_error}")
    print(f"Mult prob error: {mult_prob_error}")