"""E3.5 sc_stream + E4.5 twohop frozen temperature: default-off bit-compat and
new-path invariants (registered 2026-08-30 evening)."""

import numpy as np
import torch

from cawm import rules as R
from cawm.data import (ChunkedStreamDataset, sample_batch,
                       unpredictable_cells_mask)
from cawm.models.art_twohop import ArtTwoHop


def test_chunked_default_off_bitcompat():
    """sc_filter default OFF: chunk 0 equals the legacy direct sample_batch."""
    ds = ChunkedStreamDataset(42, half="train", grid=8, n_chunks=2, chunk_size=32)
    b = ds[0]
    _, frames, covs = sample_batch(range(32), 42, R.rule_split()[0], grid=8)
    assert np.array_equal(b["frames"].numpy(), frames)
    assert np.array_equal(b["cov"].numpy(), covs)


def test_sc_stream_invariant_and_deterministic():
    """sc_filter=True: every batch is fully self-consistent (mask rate 0) and
    chunk contents are a deterministic function of the chunk index."""
    ds = ChunkedStreamDataset(42, half="train", grid=8, n_chunks=3,
                              chunk_size=64, sc_filter=True)
    b0, b0b, b1 = ds[0], ds[0], ds[1]
    assert torch.equal(b0["frames"], b0b["frames"])          # deterministic
    assert not torch.equal(b0["frames"], b1["frames"])       # chunks differ
    for b in (b0, b1):
        m = unpredictable_cells_mask(b["frames"].numpy(), b["cov"].numpy(), None)
        assert m.sum() == 0                                   # SC invariant


def test_sc_stream_l4_prior_slot():
    """L4 stream: the prior slot counts as predictable in the filter."""
    ds = ChunkedStreamDataset(42, half="train", grid=8, n_chunks=1,
                              chunk_size=32, l4=(9, 1, 1.0), sc_filter=True,
                              prior_slots=(9,))
    b = ds[0]
    prior = np.zeros(18, dtype=bool)
    prior[9] = True
    m = unpredictable_cells_mask(b["frames"].numpy(), b["cov"].numpy(), prior)
    assert m.sum() == 0


def test_twohop_temp_modes():
    """Default 'learned' is bit-compatible (trainable param, init 1.0);
    'frozen20' is a non-trainable buffer at 20.0 with the same forward path."""
    torch.manual_seed(0)
    m_l = ArtTwoHop(grid=8)
    assert isinstance(m_l.temp, torch.nn.Parameter) and m_l.temp.item() == 1.0
    torch.manual_seed(0)
    m_f = ArtTwoHop(grid=8, temp_mode="frozen20")
    assert not isinstance(m_f.temp, torch.nn.Parameter)
    assert m_f.temp.item() == 20.0
    assert m_f.trainable_param_count() == m_l.trainable_param_count() - 1
    x = torch.randn(2, 1, 16, 8, 8).sign()
    with torch.no_grad():
        assert m_f(x).shape == (2, 8, 8, 8)
        r = m_f.rollout(x[:, :, :8])
    assert r.shape == (2, 8, 8, 8)
