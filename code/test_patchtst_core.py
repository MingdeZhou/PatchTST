from argparse import Namespace

import torch

from models.PatchTST import Model


def make_config(**overrides):
    values = dict(
        enc_in=3,
        seq_len=32,
        pred_len=8,
        patch_len=8,
        stride=4,
        d_model=16,
        n_heads=4,
        e_layers=2,
        d_ff=32,
        dropout=0.1,
        fc_dropout=0.1,
        head_dropout=0.0,
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,
    )
    values.update(overrides)
    return Namespace(**values)


def test_forward_backward():
    torch.manual_seed(7)
    model = Model(make_config())
    x = torch.randn(4, 32, 3)
    y = torch.randn(4, 8, 3)
    pred = model(x)
    assert pred.shape == (4, 8, 3)
    assert torch.isfinite(pred).all()
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


if __name__ == "__main__":
    test_forward_backward()
    print("PatchTST core smoke test passed")
