import pytest

from nemo_rl.models import true_on_policy
from nemo_rl.models.true_on_policy import (
    G_MXFP8_MATMUL_BI_BACKEND_ENV,
    G_TE_CUBLAS_WORKSPACE_SIZE_BYTES_ENV,
    get_mxfp8_matmul_bi_backend,
)


@pytest.mark.parametrize(
    ("raw_backend", "expected_backend"),
    [
        ("qdq", "qdq"),
        ("native", "native"),
        ("native-fp8", "native"),
        ("cublas", "cublas"),
        ("cublaslt", "cublas"),
        ("te", "cublas"),
        ("transformer-engine", "cublas"),
    ],
)
def test_get_mxfp8_matmul_bi_backend_aliases(
    monkeypatch,
    raw_backend,
    expected_backend,
):
    monkeypatch.setenv(G_MXFP8_MATMUL_BI_BACKEND_ENV, raw_backend)

    assert get_mxfp8_matmul_bi_backend() == expected_backend


def test_get_mxfp8_matmul_bi_backend_defaults_to_qdq(monkeypatch):
    monkeypatch.delenv(G_MXFP8_MATMUL_BI_BACKEND_ENV, raising=False)

    assert get_mxfp8_matmul_bi_backend() == "qdq"


def test_get_mxfp8_matmul_bi_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv(G_MXFP8_MATMUL_BI_BACKEND_ENV, "surprise")

    with pytest.raises(ValueError, match=G_MXFP8_MATMUL_BI_BACKEND_ENV):
        get_mxfp8_matmul_bi_backend()


def test_install_te_cublas_workspace_limit_from_env_absent(monkeypatch):
    monkeypatch.delenv(G_TE_CUBLAS_WORKSPACE_SIZE_BYTES_ENV, raising=False)

    assert true_on_policy.install_te_cublas_workspace_limit_from_env() == {
        "patched": False,
        "workspace_limit_bytes": None,
        "cache_cleared": False,
    }


def test_install_te_cublas_workspace_limit_from_env(monkeypatch):
    calls = []

    class FakeGemmModule:
        @staticmethod
        def get_cublas_workspace_size_bytes():
            return 1024

    def get_cublas_workspace():
        return None

    get_cublas_workspace.cache_clear = lambda: calls.append("cache_clear")
    FakeGemmModule.get_cublas_workspace = get_cublas_workspace

    monkeypatch.setenv(G_TE_CUBLAS_WORKSPACE_SIZE_BYTES_ENV, "4")
    monkeypatch.setattr(
        true_on_policy.importlib,
        "import_module",
        lambda name: FakeGemmModule,
    )

    assert true_on_policy.install_te_cublas_workspace_limit_from_env() == {
        "patched": True,
        "workspace_limit_bytes": 4,
        "cache_cleared": True,
    }
    assert FakeGemmModule.get_cublas_workspace_size_bytes() == 4
    assert calls == ["cache_clear"]
