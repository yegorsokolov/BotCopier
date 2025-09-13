import numpy as np
import pytest

torch = pytest.importorskip("torch")

from botcopier.models.registry import TabTransformer, fit_tab_transformer


def test_grad_clip_prevents_nan():
    X = np.random.randn(64, 4).astype(np.float32) * 1e4
    y = np.random.randint(0, 2, size=64).astype(np.float32)

    model = TabTransformer(4)
    opt = torch.optim.SGD(model.parameters(), lr=1e3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    X_t = torch.tensor(X)
    y_t = torch.tensor(y).unsqueeze(-1)
    has_nan = False
    for _ in range(10):
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        if torch.isnan(loss):
            has_nan = True
            break
        loss.backward()
        opt.step()
    assert has_nan

    _, pred_fn = fit_tab_transformer(X, y, epochs=10, lr=1e3, grad_clip=1.0)
    preds = pred_fn(X)
    assert np.isfinite(preds).all()
