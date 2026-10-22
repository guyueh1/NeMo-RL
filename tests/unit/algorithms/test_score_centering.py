"""Score-centering weights and masking against full-distribution references."""

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.score_centering import compute_score_centering_term
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.mark.parametrize("mode", ["off", "is", "tis", "icepop"])
@pytest.mark.parametrize("head_probs", [(0.1, 0.6), (0.3, 0.65)])
def test_score_centering_matches_full_distribution_gradient(mode, head_probs):
    logits = torch.tensor([0.6, -0.2, 0.1, -0.7], requires_grad=True)
    logp = logits.log_softmax(-1)
    q_head = torch.tensor(head_probs, requires_grad=True)
    q = torch.cat(
        [q_head.detach(), (1 - q_head.detach().sum()) * logp[2:].softmax(-1).detach()]
    )
    correction, _ = compute_score_centering_term(
        logp[:2].reshape(1, 1, 2),
        sampler_topk_logprobs=q_head.log().reshape(1, 1, 2),
        importance_sampling_mode=mode,
        is_bounds=(0.5, 2.0) if mode in ("tis", "icepop") else None,
        token_mask=torch.ones(1, 1),
    )
    ratios = logp.detach().exp() / q
    if mode == "off":
        weights = torch.ones_like(ratios)
    elif mode == "is":
        weights = ratios
    elif mode == "tis":
        weights = ratios.clamp(0.5, 2.0)
    else:
        weights = torch.where((ratios >= 0.5) & (ratios <= 2.0), ratios, 0.0)
    expected = (q * weights * logp).sum()
    torch.testing.assert_close(
        torch.autograd.grad(correction.sum(), logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        atol=1e-6,
        rtol=1e-5,
    )
    assert q_head.grad is None
    if mode == "is":
        torch.testing.assert_close(correction, torch.zeros_like(correction))


@pytest.mark.parametrize("mode", ["off", "is", "tis", "icepop"])
@pytest.mark.parametrize("force_on_policy", [True, False])
def test_score_centering_masks_nonfinite_positions_and_keeps_icepop_correction(
    mode, force_on_policy
):
    lp = torch.tensor([0.6, -0.2, 0.1, -0.7]).log_softmax(-1)
    token_lp = torch.tensor([[lp[0], float("nan")]], requires_grad=True)
    head_lp = torch.tensor(
        [[[lp[0], lp[1]], [float("nan"), float("inf")]]], requires_grad=True
    )
    advantage = torch.tensor([[0.0, 1.0, float("nan")]], requires_grad=True)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[0, 0, 0]]),
            "token_mask": torch.tensor([[0.0, 1.0, 0.0]]),
            "sample_mask": torch.ones(1),
            "advantages": advantage,
            # Ratio 2 lies beyond the PPO clip range. The sampled-token gradient
            # must stay clipped when force_on_policy_ratio is disabled.
            "prev_logprobs": torch.tensor(
                [[0.0, lp[0] - torch.log(torch.tensor(2.0)), float("nan")]]
            ),
            "generation_logprobs": torch.tensor([[0.0, -2.3025851, float("nan")]]),
            "sampler_topk_logprobs": torch.tensor(
                [[[0.0, 0.0], [0.1, 0.6], [float("nan"), float("nan")]]]
            ).log(),
        }
    )
    cfg = ClippedPGLossConfig(
        score_centering=True,
        reference_policy_kl_penalty=0,
        force_on_policy_ratio=force_on_policy,
        use_importance_sampling_correction=mode != "off",
        truncated_importance_sampling_type=mode if mode in ("tis", "icepop") else None,
        truncated_importance_sampling_ratio=2.0 if mode in ("tis", "icepop") else None,
        truncated_importance_sampling_ratio_min=0.5
        if mode in ("tis", "icepop")
        else None,
    )
    loss, _ = ClippedPGLossFn(cfg)(
        next_token_logprobs=token_lp,
        policy_logprobs_at_sampler_topk=head_lp,
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(1.0),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(token_lp.grad).all() and torch.isfinite(head_lp.grad).all()
    assert token_lp.grad[0, 1] == 0 and torch.all(head_lp.grad[0, 1] == 0)
    assert advantage.grad is None
    if not force_on_policy:
        assert token_lp.grad[0, 0] == 0
    if mode == "icepop":
        # p(sampled)/q(sampled) exceeds the upper bound: the sampled loss is
        # zero, but the head/tail correction still supplies a gradient.
        assert token_lp.grad[0, 0] == 0
        assert head_lp.grad[0, 0].abs().sum() > 0
