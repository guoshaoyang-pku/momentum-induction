"""L4 evaluation must preserve constrained-family and exercised semantics."""
import numpy as np
from cawm.data import build_eval_corpus
from diagnose_structured import corpus_audit
from eval_lifegpt import load_evaluation_corpus


def test_lifegpt_l4_standard_loader_uses_pinned_l4_family(tmp_path):
    path=tmp_path/'l4b_zs_sc_g8.npz'
    build_eval_corpus(str(path),n=2048,master_seed=45,half='zs',grid=8,
                      self_consistent=True,l4v2='l4b')
    c=load_evaluation_corpus('l4b_zs_sc',8,str(tmp_path))
    assert c['frames'].shape==(2048,16,8,8)
    corpus_audit(c,l4=True,exercised=False,half='zs')
    # A normal SC corpus must not be accepted as an all-exercised corpus.
    try:
        corpus_audit(c,l4=True,exercised=True,half='zs')
    except AssertionError:
        pass
    else:
        raise AssertionError('ordinary SC mistaken for exercised L4')


def test_lifegpt_l4_loader_refuses_absent_frozen_corpus(tmp_path):
    try:
        load_evaluation_corpus('l4b_zs_scx',8,str(tmp_path))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('missing frozen corpus silently regenerated')
