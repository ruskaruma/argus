import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_torch: requires PyTorch")
    config.addinivalue_line("markers", "requires_cuda: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: slow benchmark tests")


@pytest.fixture
def tracer():
    """Fresh tracer instance for each test."""
    from argus import Tracer

    return Tracer()


def pytest_collection_modifyitems(items):
    for item in items:
        if "requires_torch" in item.keywords:
            try:
                import torch  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="PyTorch not installed"))
        if "requires_cuda" in item.keywords:
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError
            except (ImportError, RuntimeError):
                item.add_marker(pytest.mark.skip(reason="CUDA not available"))
