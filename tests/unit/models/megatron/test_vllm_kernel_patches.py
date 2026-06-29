from nemo_rl.models.megatron import vllm_kernel_patches
from nemo_rl.models.true_on_policy import G_MXFP8_MATMUL_BI_BACKEND_ENV


def test_bf16_true_on_policy_installs_megatron_sdpa_and_te_rmsnorm(monkeypatch):
    calls = []

    def fake_install_vllm_style_sdpa():
        calls.append("sdpa")
        return {"patched": True, "already_installed": False, "backend": "vllm_fa2"}

    def fake_install_te_batch_invariant_rmsnorm_patch():
        calls.append("te_rmsnorm")
        return 6

    monkeypatch.setattr(
        vllm_kernel_patches,
        "install_vllm_style_sdpa",
        fake_install_vllm_style_sdpa,
    )
    monkeypatch.setattr(
        vllm_kernel_patches,
        "install_te_batch_invariant_rmsnorm_patch",
        fake_install_te_batch_invariant_rmsnorm_patch,
    )

    result = vllm_kernel_patches.install_true_on_policy_patches(
        bf16_true_on_policy=True,
        mxfp8_matmul_batch_invariant=False,
        mxfp8_active=False,
    )

    assert calls == ["sdpa", "te_rmsnorm"]
    assert result == {
        "vllm_style_sdpa": {
            "patched": True,
            "already_installed": False,
            "backend": "vllm_fa2",
        },
        "te_batch_invariant_rmsnorm_entrypoints": 6,
        "bf16_true_on_policy": "megatron_true_on_policy_patches",
    }


def test_mxfp8_cublas_backend_uses_te_passthrough(monkeypatch):
    calls = []

    monkeypatch.setattr(
        vllm_kernel_patches,
        "install_megatron_true_on_policy_patches",
        lambda: {},
    )
    monkeypatch.setattr(
        vllm_kernel_patches,
        "install_bi_mxfp8_matmul_cublas",
        lambda: calls.append("cublas"),
    )
    monkeypatch.setattr(
        vllm_kernel_patches,
        "install_bi_mxfp8_matmul",
        lambda: calls.append("native"),
    )
    monkeypatch.setenv(G_MXFP8_MATMUL_BI_BACKEND_ENV, "cublas")

    result = vllm_kernel_patches.install_true_on_policy_patches(
        bf16_true_on_policy=True,
        mxfp8_matmul_batch_invariant=True,
        mxfp8_active=True,
    )

    assert calls == ["cublas"]
    assert result == {
        "bf16_true_on_policy": "megatron_true_on_policy_patches",
        "mxfp8_matmul_batch_invariant": "cublas",
    }
