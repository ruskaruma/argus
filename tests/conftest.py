import pytest

_torch_available: bool | None = None
_cuda_available: bool | None = None


def _check_torch() -> bool:
    global _torch_available
    if _torch_available is None:
        try:
            import torch  # noqa: F401

            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


def _check_cuda() -> bool:
    global _cuda_available
    if _cuda_available is None:
        if not _check_torch():
            _cuda_available = False
        else:
            import torch

            _cuda_available = torch.cuda.is_available()
    return _cuda_available


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_torch: requires PyTorch")
    config.addinivalue_line("markers", "requires_cuda: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: slow benchmark tests")


@pytest.fixture
def tracer():
    """Fresh tracer instance for each test."""
    from argus import Tracer

    return Tracer()


@pytest.fixture(scope="session")
def tiny_model():
    """Session-scoped tiny GPT2 model — downloaded once per test run."""
    if not _check_torch():
        pytest.skip("PyTorch not installed")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    input_ids = tokenizer.encode("Hello", return_tensors="pt")
    return model, input_ids


def pytest_collection_modifyitems(items):
    for item in items:
        if "requires_torch" in item.keywords and not _check_torch():
            item.add_marker(pytest.mark.skip(reason="PyTorch not installed"))
        if "requires_cuda" in item.keywords and not _check_cuda():
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))
