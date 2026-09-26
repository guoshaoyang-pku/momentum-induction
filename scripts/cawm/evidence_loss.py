"""E19 training-only evidence strata; never used by model inference."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def evidence_groups(frames, prefix_cov, mode='relation'):
    """Return (B,8,H,W) group IDs using original prefix coverage only.

    frames are binary teacher-forcing training inputs. A label uses its
    source frame, never its outcome or later coverage. L4B SC required.
    relation: direct=0, partner-only=1; slot_relation: 18 slots per group.
    """
    if mode not in ('relation', 'slot_relation', 'slot'):
        raise ValueError(mode)
    src = frames[:, 7:15].float()
    b, t, h, w = src.shape
    kernel = torch.ones(1, 1, 3, 3, device=src.device)
    kernel[0, 0, 1, 1] = 0
    # Disable autocast so binary counts remain exact on all backends.
    with torch.autocast(device_type=src.device.type, enabled=False):
        nbr = F.conv2d(F.pad(src.reshape(-1, 1, h, w), (1,1,1,1), mode='circular'), kernel)
    slot = (9 * src.long() + nbr.reshape(b,t,h,w).long())
    if mode == 'slot':
        return slot
    direct = prefix_cov.bool().gather(1, slot.flatten(1)).reshape_as(slot)
    partner = torch.arange(18, device=src.device)
    partner[3], partner[12], partner[4], partner[13] = 12, 3, 13, 4
    paired = partner[slot] != slot
    pcov = prefix_cov.bool().gather(1, partner[slot].flatten(1)).reshape_as(slot)
    derived = ~direct & paired & pcov
    if not torch.all(direct | derived):
        raise ValueError('Evidence-balanced loss requires answerable L4B SC targets')
    return derived.long() if mode == 'relation' else derived.long()*18 + slot


def evidence_bce(logits, targets, groups, mode='relation'):
    """Equal mean over present evidence groups, then present slots if asked.

    Normalize AFTER frontier slicing; absent groups/slots add no phantom loss.
    Accumulation is fp32 under mixed precision. No random-number consumption.
    """
    if logits.shape != targets.shape or groups.shape != targets.shape:
        raise ValueError('logits, targets and groups must have identical shapes')
    n = 2 if mode == 'relation' else 36 if mode == 'slot_relation' else 18 if mode == 'slot' else 0
    if not n:
        raise ValueError(mode)
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
    ids = groups.flatten()
    sums = torch.zeros(n, device=logits.device).scatter_add(0, ids, bce.flatten())
    counts = torch.zeros(n, device=logits.device).scatter_add(0, ids, torch.ones_like(bce).flatten())
    means = sums / counts.clamp_min(1)
    present = counts > 0
    if mode == 'slot_relation':
        means = (means.reshape(2,18) * present.reshape(2,18)).sum(1) / present.reshape(2,18).sum(1).clamp_min(1)
        present = present.reshape(2,18).any(1)
    return (means * present).sum() / present.sum().clamp_min(1)
