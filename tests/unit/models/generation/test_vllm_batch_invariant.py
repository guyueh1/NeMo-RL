import torch

from nemo_rl.models.generation.vllm import batch_invariant
from nemo_rl.models.true_on_policy import G_MXFP8_MATMUL_BI_BACKEND_ENV


def test_true_on_policy_mxfp8_cublas_backend_installs_cublas(monkeypatch):
    calls = []

    monkeypatch.setattr(
        batch_invariant,
        "install_true_on_policy_patch_components",
        lambda model, components: {"bf16": components},
    )
    monkeypatch.setattr(
        batch_invariant,
        "install_mxfp8_bi_cublas_patch",
        lambda model: calls.append("cublas") or {"patched": True},
    )
    monkeypatch.setattr(
        batch_invariant,
        "install_mxfp8_bi_matmul_patch",
        lambda model: calls.append("native") or {"patched": True},
    )
    monkeypatch.setenv(G_MXFP8_MATMUL_BI_BACKEND_ENV, "cublas")

    result = batch_invariant.install_true_on_policy_patches(
        torch.nn.Module(),
        bf16_true_on_policy=True,
        mxfp8_matmul_batch_invariant=True,
    )

    assert calls == ["cublas"]
    assert result["mxfp8_matmul_backend"] == "cublas"
    assert result["mxfp8_matmul"] == {"patched": True}
