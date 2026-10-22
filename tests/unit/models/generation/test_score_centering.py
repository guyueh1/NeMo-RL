"""Contract of score-centering request preparation and sampler extraction."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.score_centering import (
    build_score_centering_request_overrides,
    extract_score_centering_samples,
    validate_score_centering_request,
)


def test_score_centering_request_and_sample_contract():
    # Gym merges these fields into existing request overrides. Sampling controls
    # such as temperature must survive, while the requested logprob count wins.
    request = {"temperature": 1.0, "top_logprobs": 0}
    request.update(build_score_centering_request_overrides(num_candidates=2))
    assert request["temperature"] == 1.0
    validate_score_centering_request(SimpleNamespace(**request), num_candidates=2)

    output = SimpleNamespace(
        index=0,
        logprobs=[
            {
                # The generated token is outside the head and arrives first.
                9: SimpleNamespace(rank=7, logprob=-3.0),
                1: SimpleNamespace(rank=1, logprob=-0.3),
                2: SimpleNamespace(rank=2, logprob=-1.8),
            },
            {
                4: SimpleNamespace(rank=2, logprob=-10000.0),
                3: SimpleNamespace(rank=1, logprob=-0.1),
            },
        ],
    )
    # Staging receives CPU tensors even if the serving caller has selected a
    # different default device.
    with torch.device("meta"):
        sampler_ids, sampler_logprobs = extract_score_centering_samples(
            output, num_candidates=request["top_logprobs"]
        )

    assert sampler_ids.dtype == torch.int32
    assert sampler_logprobs.dtype == torch.float32
    assert sampler_ids.device.type == sampler_logprobs.device.type == "cpu"
    assert sampler_ids.tolist() == [[1, 2], [3, 4]]
    torch.testing.assert_close(
        sampler_logprobs, torch.tensor([[-0.3, -1.8], [-0.1, -9999.0]])
    )
    # The retained logprobs describe the full sampler distribution, not a
    # distribution renormalized over the chosen candidates.
    assert torch.all(sampler_logprobs.exp().sum(-1) < 1)


@pytest.mark.parametrize("logprobs", [None, []])
def test_score_centering_empty_completion(logprobs):
    sampler_ids, sampler_logprobs = extract_score_centering_samples(
        SimpleNamespace(index=0, logprobs=logprobs), num_candidates=2
    )
    assert sampler_ids.shape == sampler_logprobs.shape == (0, 2)
    assert sampler_ids.dtype == torch.int32
    assert sampler_logprobs.dtype == torch.float32


def test_score_centering_rejects_incomplete_candidates():
    output = SimpleNamespace(
        index=0, logprobs=[{1: SimpleNamespace(rank=1, logprob=-0.1)}]
    )
    with pytest.raises(RuntimeError, match="fewer than top_k"):
        extract_score_centering_samples(output, num_candidates=2)


@pytest.mark.parametrize(
    "request_fields",
    [{"logprobs": False, "top_logprobs": 2}, {"logprobs": True, "top_logprobs": 0}],
)
def test_score_centering_rejects_inconsistent_request(request_fields):
    with pytest.raises(ValueError, match="top_logprobs=2 over HTTP"):
        validate_score_centering_request(
            SimpleNamespace(**request_fields), num_candidates=2
        )


def test_score_centering_rejects_zero_candidates():
    with pytest.raises(ValueError, match="candidate count must be positive"):
        build_score_centering_request_overrides(num_candidates=0)
