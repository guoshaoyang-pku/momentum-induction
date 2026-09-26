"""ArtTemporal: a bounded temporal-context extension of ArtTwoHop.

The model keeps the entire ArtTwoHop machinery (emergence arm, d=64,
learned temperature, query detector, two unmasked attention blocks,
head) and only replaces the *memory front end*: instead of encoding a
single frame per token, every one of the 8 prefix frames becomes the
anchor of a 3-frame temporal window ``[prev, current, next]``.
Unobserved neighbours (before frame 0 and after frame 7) are filled
with numeric zeros -- there is *no* temporal wrap-around.

Four temporal modes gate the input slices of the window multiplicatively:

    center     (0, 1, 0)   -- main mode
    forward    (0, 1, 1)   -- main mode
    backward   (1, 1, 0)   -- secondary mode
    symmetric  (1, 1, 1)   -- secondary mode

crossed with positional-encoding modes ``learned`` / ``none`` this gives
8 variants (2 main x 2 PE, 2 secondary x 2 PE).

Equal allocation guarantee
--------------------------
All 8 variants allocate *identical* state shapes and parameter counts:

* the parent is always constructed with ``pos='learned'`` so that
  ``tok_pos`` and ``q_pos`` are always allocated; ``self.pos`` is set to
  the requested value only AFTER the constructor, so with the same seed
  every variant has an identical initial ``state_dict()`` and identical
  subsequent RNG state;
* the temporal mode is realised purely by multiplicative gating of the
  input slices at forward time -- no buffers whose state differs across
  modes are registered.

Note that *equal allocated parameters* is NOT the same as *equal active
degrees of freedom*: in ``center`` mode the temporal-neighbour slices of
the first Conv3d kernel never receive gradient (their input is always
zero), and with ``pos='none'`` the positional parameters are allocated
but never used.  The equality is an accounting/experimental-control
property, not a statement about effective capacity.

Circular spatial padding is done exclusively via
``cawm.models.constructive_cnn._circ3d`` (spatial dims only; the
temporal dim of the 3-frame window is consumed by the (3,3,3) kernel).
"""

import torch
import torch.nn as nn

from .art_twohop import ArtTwoHop

from .constructive_cnn import _circ3d

MODEL_NAME = "art_temporal"

TEMPORAL_MODES = ("center", "forward", "backward", "symmetric")

# Multiplicative gates over the (prev, current, next) input slices.
_TEMPORAL_MASKS = {
    "center": (0.0, 1.0, 0.0),
    "forward": (0.0, 1.0, 1.0),
    "backward": (1.0, 1.0, 0.0),
    "symmetric": (1.0, 1.0, 1.0),
}


class ArtTemporal(ArtTwoHop):
    """ArtTwoHop with a temporal 3-frame-window memory front end.

    Parameters
    ----------
    grid : int
        Spatial grid size (H = W = grid).
    temporal : str
        One of ``center``, ``forward``, ``backward``, ``symmetric``.
    pos : str
        ``learned`` or ``none``.  Positional parameters are ALWAYS
        allocated (parent constructed with ``pos='learned'``); this flag
        only gates their use at forward time.
    """

    def __init__(self, grid=8, temporal="center", pos="learned",
                 head="concat", temp_mode="learned", d=64):
        if temporal not in _TEMPORAL_MASKS:
            raise ValueError(
                "temporal must be one of %r, got %r" % (TEMPORAL_MODES, temporal))
        if pos not in ("learned", "none"):
            raise ValueError("pos must be 'learned' or 'none', got %r" % (pos,))

        # Always allocate positional parameters so that every variant has
        # identical state shapes / parameter counts and consumes the RNG
        # stream identically.
        super().__init__(grid=grid, arm="emergence", head=head,
                         temp_mode=temp_mode, d=d, pos="learned")

        self.name = MODEL_NAME
        self.temporal = temporal

        # Memory front end: temporal window (B*8, 1, 3, H, W) -> 36-d codes.
        # Conv3d(1,3,(3,3,3)) consumes the temporal dim (3 -> 1); spatial
        # dims are circularly padded via _circ3d before the call.
        self.mem_front = nn.Sequential(
            nn.Conv3d(1, 3, (3, 3, 3)),
            nn.Conv3d(3, 27, 1),
            nn.ReLU(),
            nn.Conv3d(27, 36, 1),
            nn.ReLU(),
        )

        # Replace the parent's Linear(18, d) token projection with one
        # that accepts the 36-d temporal memory codes.
        self.tok_proj = nn.Linear(36, d)

        # Set the *effective* positional mode only AFTER all parameter
        # allocation, so initial state_dict and subsequent RNG draws are
        # identical across all 8 variants for a fixed seed.  The parent's
        # _query gates q_pos on self.pos at forward time; _tokens below
        # gates tok_pos likewise.
        self.pos = pos

    # ------------------------------------------------------------------
    # Memory front end (useful for probes)
    # ------------------------------------------------------------------
    def _memory_windows(self, frames):
        """Build gated 3-frame windows around each of the 8 anchor frames.

        frames : (B, 1, 8, H, W)
        returns: (B*8, 1, 3, H, W) with slices (prev, current, next).
        Unobserved neighbours are numeric zero (no temporal wrap); the
        temporal mode's mask multiplicatively gates the input slices.
        """
        B, C, T, H, W = frames.shape
        if C != 1 or T != 8:
            raise ValueError("expected frames of shape (B, 1, 8, H, W), got %r"
                             % (tuple(frames.shape),))
        zero = frames.new_zeros(B, 1, 1, H, W)
        padded = torch.cat([zero, frames, zero], dim=2)  # (B, 1, 10, H, W)
        wins = torch.stack([padded[:, :, t:t + 3] for t in range(T)], dim=1)
        wins = wins.reshape(B * T, 1, 3, H, W)
        mask = frames.new_tensor(_TEMPORAL_MASKS[self.temporal]).view(1, 1, 3, 1, 1)
        return wins * mask

    def _memory_codes(self, frames):
        """Run the memory front end.

        frames : (B, 1, 8, H, W)
        returns: (B, 8*H*W, 36) codes, frame-major then cell-major, which
        matches the parent's 8*ncells token / tok_pos ordering.
        """
        B, _, T, H, W = frames.shape
        wins = self._memory_windows(frames)          # (B*8, 1, 3, H, W)
        h = self.mem_front(_circ3d(wins, 1))          # (B*8, 36, 1, H, W)
        h = h.squeeze(2)                              # (B*8, 36, H, W)
        h = h.reshape(B, T, 36, H * W).permute(0, 1, 3, 2)  # (B, 8, H*W, 36)
        return h.reshape(B, T * H * W, 36)

    # ------------------------------------------------------------------
    # Token construction (parent forward / rollout are reused as-is)
    # ------------------------------------------------------------------
    def _tokens(self, prefix):
        """prefix : (B, 1, 8, H, W) -> (B, 8*H*W, d) memory tokens."""
        tok = self.tok_proj(self._memory_codes(prefix))
        if self.pos == "learned":
            tok = tok + self.tok_pos
        for blk in self.blocks:
            tok = blk(tok)
        return tok
