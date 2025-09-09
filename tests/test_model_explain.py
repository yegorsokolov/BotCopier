import pytest

torch = pytest.importorskip("torch")

from scripts.explain import integrated_gradients
from scripts.train_target_clone import TabTransformer


def test_integrated_gradients_shapes():
    model = TabTransformer(num_features=3, dim=8, heads=2, depth=1, ff_dim=16, dropout=0.0)
    model.eval()
    batch = torch.randn(4, 3)
    attrs = integrated_gradients(model, batch, steps=8)
    assert attrs.shape == batch.shape
    assert torch.any(attrs != 0)
