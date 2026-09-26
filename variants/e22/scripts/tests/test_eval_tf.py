"""Unit test for teacher_forced_corpus with a hand-written XOR model."""

import numpy as np
import torch

from scripts.cawm.eval import teacher_forced_corpus


class _XorModel:
    """Predicts frame 8+j as the bitwise XOR of the 8 TRUE input frames."""

    def __call__(self, frames_pm1):
        bits = (frames_pm1[:, 0] > 0).float()  # (B,16,H,W)
        preds = [bits[:, j:j + 8].sum(dim=1) % 2 for j in range(8)]
        return torch.stack(preds, dim=1) * 2.0 - 1.0  # logits, sign = bit


def test_teacher_forced_xor():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 2, size=(4, 16, 8, 8), dtype=np.uint8)
    # Make trajectory 0 self-consistent so the XOR model is exact on it.
    for j in range(8):
        frames[0, 8 + j] = np.bitwise_xor.reduce(frames[0, j:j + 8], axis=0)
    # Reference metrics computed by hand in numpy.
    correct = np.zeros((4, 8, 8, 8), dtype=bool)
    for j in range(8):
        pred = np.bitwise_xor.reduce(frames[:, j:j + 8], axis=1)
        correct[:, j] = pred == frames[:, 8 + j]
    exp_pixel = correct.mean()
    exp_seq = correct.reshape(4, 8, -1).all(axis=-1).mean()

    out = teacher_forced_corpus(_XorModel(), {"frames": frames}, batch=2)
    assert abs(out["tf_pixel_acc"] - exp_pixel) < 1e-9
    assert abs(out["tf_seq_acc"] - exp_seq) < 1e-9
    assert out["n"] == 4
    assert out["n_steps"] == 8
    # Trajectory 0 contributes 8 perfect (trajectory, j) pairs out of 32.
    assert out["tf_seq_acc"] >= 8 / 32
