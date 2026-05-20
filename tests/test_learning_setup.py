import torch


def test_torch_imports_and_creates_cpu_tensor():
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert tensor.device.type == "cpu"
    assert tensor.tolist() == [1.0, 2.0]


def test_learning_package_imports():
    import learning

    assert learning is not None

