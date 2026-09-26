"""Render CA state sequences into video frames.

Spin convention (same as the mainline): black cell = +1, white cell = -1.
Background is -1 as well, so the arena edge is exactly the zero boundary
of the CA; an optional thin border line marks the arena for the model.

Geometry of a rendered video: (s, oy, ox) with the grid occupying
[y: y+8s, x: x+8s]. Everything is generated with known geometry, so eval
can de-render pixels back to cells.
"""
import numpy as np


def allowed_offsets(frame: int, grid: int, s: int):
    """Offsets that keep the grid in-frame and lattice-aligned (multiples of s)."""
    hi = frame - grid * s
    return [(oy, ox) for oy in range(0, hi + 1, s) for ox in range(0, hi + 1, s)]


def render_states(states, s=8, offset=(0, 0), frame=128, hold=1,
                  border=1, border_val=0.0):
    """states (T,H,W) {0,1} -> (T*hold, 1, frame, frame) float32 in [-1, 1]."""
    T, H, W = states.shape
    oy, ox = offset
    vid = np.full((T, 1, frame, frame), -1.0, np.float32)
    cells = states.astype(np.float32) * 2 - 1
    cells = np.repeat(np.repeat(cells, s, axis=1), s, axis=2)
    vid[:, 0, oy:oy + H * s, ox:ox + W * s] = cells
    if border > 0:
        b = border
        y0, y1 = max(oy - b, 0), min(oy + H * s + b, frame)
        x0, x1 = max(ox - b, 0), min(ox + W * s + b, frame)
        vid[:, 0, y0:y1, x0:x0 + b] = border_val
        vid[:, 0, y0:y1, x1 - b:x1] = border_val
        vid[:, 0, y0:y0 + b, x0:x1] = border_val
        vid[:, 0, y1 - b:y1, x0:x1] = border_val
    if hold > 1:
        vid = np.repeat(vid, hold, axis=0)
    return vid


def frame_to_cells(img, s, offset, grid=8):
    """img (frame, frame) in [-1,1] -> (grid, grid) uint8 by block averaging."""
    oy, ox = offset
    region = img[oy:oy + grid * s, ox:ox + grid * s]
    blocks = region.reshape(grid, s, grid, s).mean(axis=(1, 3))
    return (blocks > 0).astype(np.uint8)
