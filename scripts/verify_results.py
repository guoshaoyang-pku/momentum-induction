#!/usr/bin/env python3
"""Verify bundled compact results without access to the research workspace."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    figures = json.loads((ROOT/'results/FIGURES.json').read_text())['figures']
    for row in figures:
        for extension in ('pdf','png'):
            path = ROOT/'results/figures'/f"{row['name']}.{extension}"
            assert digest(path) == row[f'{extension}_sha256'], path
    rows = list(csv.DictReader((ROOT/'results/data/e21_selected_weights.csv').open()))
    assert len(rows) == 6 and {(r['model'],r['seed']) for r in rows} == {(m,str(s)) for m in ('art','kv') for s in (42,43,44)}
    corpus = ROOT/'data/eval_corpora/l4b_zs_scx_g8.npz'
    for row in rows:
        assert row['corpus_sha256'] == digest(corpus)
        assert digest(ROOT/row['checkpoint']) == row['checkpoint_sha256']
        assert 0 <= float(row['seq_acc']) <= 1 and int(row['n']) == 768
    depth = json.loads((ROOT/'results/data/e21_depth_summary.json').read_text())
    for row in rows:
        key = f"{'art' if row['model']=='art' else 'kv'}/{'plain' if row['loss']=='plain' else 'slot_relation'}/r4"
        assert math.isclose(float(row['seq_acc']),depth[key]['test'][int(row['seed'])-42],abs_tol=1e-12)
    comparison = list(csv.DictReader((ROOT/'results/data/e11_same_weights_8_calls.csv').open()))
    assert len(comparison)==12
    for world, expected in {'Game of Life':(0.4225260416666667,0.9993489583333334),'Billiards':(0.9921875,1.0)}.items():
        for sampler, target in zip(('uniform','frame_ar_noise'),expected):
            selected=[r for r in comparison if r['world']==world and r['sampler']==sampler]
            assert len(selected)==3 and {int(r['model_seed']) for r in selected}=={42,43,44}
            assert all(int(r['calls_per_rollout'])==8 for r in selected)
            assert math.isclose(mean(float(r['seq_acc']) for r in selected),target,abs_tol=1e-12)
    e22=list(csv.DictReader((ROOT/'results/data/e22_rows.csv').open()))
    assert len(e22)==24 and all(r['Audit status']=='audited' for r in e22)
    plotted=list(csv.DictReader((ROOT/'results/data/two_metrics_l3_seeds.csv').open()))
    assert len(plotted)==30 and len({r['model'] for r in plotted})==10
    for model in {r['model'] for r in plotted}:
        selected=[r for r in plotted if r['model']==model]
        assert {int(r['model_seed']) for r in selected}=={42,43,44}
        assert all(int(r['n_worlds'])==2048 and 0<=float(r['pixel_acc'])<=1 and 0<=float(r['seq_acc'])<=1 for r in selected)
        expected_mode='rollout' if model in ('standard diffusion','causal-freezing diffusion') else 'teacher_forced'
        assert {r['pixel_mode'] for r in selected}=={expected_mode}
    print(f'verified {len(figures)} figures, {len(rows)} checkpoint records, {len(comparison)} sampler rows, {len(e22)} E22 rows, {len(plotted)} Figure 2 seed rows')

if __name__=='__main__':
    verify()
