"""Pixel lab smoke tests (run: python3 -m pixel.tests_pixel from scripts/)."""
import numpy as np
import torch

from . import ca as ca_mod
from . import render as render_mod
from .models_pixel import build
from .stream import PixelStream, make_eval_batches


def test_ca_blinker():
    table = ca_mod.table_from_rule(ca_mod.RULE_GOL)
    g = np.zeros((8, 8), np.uint8)
    g[4, 3:6] = 1
    g2 = ca_mod.step(g, table)
    assert g2[3, 4] == 1 and g2[4, 4] == 1 and g2[5, 4] == 1
    assert g2.sum() == 3
    g3 = ca_mod.step(g2, table)
    assert (g3 == g).all()
    print("ca blinker ok")


def test_activity():
    rng = np.random.default_rng(0)
    st = ca_mod.trajectory(ca_mod.table_from_rule(ca_mod.RULE_B2S3), rng, 16, 0.3)
    diffs = [(st[t + 1] != st[t]).mean() for t in range(15)]
    assert np.mean(diffs) > 0.05, f"rule too static: {np.mean(diffs)}"
    print(f"b2s3 activity ok (mean cell-change {np.mean(diffs):.3f})")


def test_render_roundtrip():
    rng = np.random.default_rng(1)
    st = ca_mod.trajectory(ca_mod.table_from_rule(ca_mod.RULE_GOL), rng, 4)
    vid = render_mod.render_states(st, s=8, offset=(24, 40), frame=128,
                                   hold=2, border=1)
    assert vid.shape == (8, 1, 128, 128)
    back = render_mod.frame_to_cells(vid[0, 0], 8, (24, 40))
    assert (back == st[0]).all()
    print("render roundtrip ok")


def test_sc_rejection():
    rng = np.random.default_rng(2)
    table = ca_mod.table_from_rule(12345)
    st = ca_mod.trajectory(table, rng, 16, 0.4)
    if ca_mod.is_self_consistent(st):
        seen = set()
        for t in range(7):
            seen.update(np.unique(ca_mod.slots_of(st[t])).tolist())
        for t in range(7, 15):
            assert set(np.unique(ca_mod.slots_of(st[t])).tolist()) <= seen
    print("sc check ok")


def test_stream_and_models():
    for k_max, model_name, s, frame in ((1, "pcnn", 8, 128),
                                        (2, "pcnn", 16, 128),
                                        (2, "part", 16, 128)):
        ds = PixelStream(level="L1", s=s, frame=frame,
                         offsets=[((frame - 8 * s) // 2,) * 2],
                         k_max=k_max, batch=4, seed=0)
        b = next(iter(ds))
        model = build(model_name, s=s, frame=frame)
        out, lat = model(b["hist"])
        assert out.shape == b["fut"][:, 0].shape, (model_name, out.shape)
        n = sum(p.numel() for p in model.parameters())
        print(f"{model_name} s={s} kmax={k_max}: out {tuple(out.shape)}, "
              f"params {n:,}")


def test_overfit():
    torch.manual_seed(0)
    # frame 64 with s=8: grid fills the frame and the encoder latent lands
    # exactly on the 8x8 cell grid; small enough for a CPU smoke.
    ds = PixelStream(level="L1", s=8, frame=64, offsets=[(0, 0)],
                     k_max=1, batch=8, seed=0)
    b = next(iter(ds))
    model = build("pcnn", s=8, frame=64)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    hist, fut = b["hist"], b["fut"]
    first = last = None
    for i in range(80):
        window = hist
        ls = []
        for j in range(8):
            logits, _ = model(window)
            ls.append(torch.nn.functional.binary_cross_entropy_with_logits(
                logits, (fut[:, j] + 1) / 2))
            window = torch.cat([window[:, 1:], fut[:, j:j + 1]], 1)
        loss = torch.stack(ls).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    print(f"overfit: {first:.4f} -> {last:.4f}")
    # CPU smoke only checks direction; the real overfit check is on GPU.
    assert last < first * 0.75
    print("overfit ok")


def test_eval_runs():
    ds = make_eval_batches(level="L1", s=8, frame=64, offsets=[(0, 0)],
                           k_values=(1,), n_traj=2)
    model = build("pcnn", s=8, frame=64)
    from .train_pixel import evaluate
    per_k = evaluate(model, ds, 8, 8, 8, "cpu")
    assert per_k[0]["k"] == 1
    print("eval pipeline ok", per_k[0])


if __name__ == "__main__":
    test_ca_blinker()
    test_activity()
    test_render_roundtrip()
    test_sc_rejection()
    test_stream_and_models()
    test_overfit()
    test_eval_runs()
    print("ALL PIXEL TESTS PASS")
