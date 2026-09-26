"""E13 fixed-final-checkpoint evaluation; stores predictions for audit."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

from cawm.train import build_model, model_kwargs, get_corpus, code_fingerprint
from cawm.eval import rollout_corpus, metrics
from cawm.models.lifegpt import LifeGPT


def sample_token(logits, temperature=0.0, generator=None):
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    if temperature < 0:
        raise ValueError('temperature must be nonnegative')
    return torch.multinomial((logits / temperature).softmax(-1), 1,
                             generator=generator)


@torch.no_grad()
def lifegpt_cached_generate(model, prefix, length, temperature=0.0, generator=None):
    """Untruncated ancestral sampling; same model/cache as registered greedy."""
    assert not model.training
    b, offset = prefix.shape
    shape = (b, model.nhead, offset+length, model.head_dim)
    cache = [(torch.empty(shape, device=prefix.device, dtype=model.embed.weight.dtype),
              torch.empty(shape, device=prefix.device, dtype=model.embed.weight.dtype))
             for _ in model.blocks]
    logits = model.token_logits(prefix, cache=cache)
    outputs = []
    for i in range(length):
        token = sample_token(logits[:, -1], temperature, generator)
        outputs.append(token)
        if i+1 < length:
            logits = model.token_logits(token, cache=cache, offset=offset+i)
    return torch.cat(outputs, 1)


def binary_trajectory_log_probability(logit, target, temperature=1.0):
    """Log P(entire true continuation | prefix), using true-path conditionals.

    This product is the expected exact-trajectory indicator under ancestral
    sampling; it is not a teacher-forced pixel accuracy or a sampled score.
    Double precision avoids losing tiny error probabilities at confident tokens.
    """
    assert temperature > 0
    signed = logit.double() * (target.double()*2-1) / temperature
    return torch.nn.functional.logsigmoid(signed).flatten(1).sum(1)


@torch.no_grad()
def standard_cached_generate(model, prefix, length):
    """Exact standardTF learned-position/cell inference, with per-layer KV cache.

    No changed weights/attention/position convention. Tested against the
    registered _encode slow path token logits before using it for evaluation.
    """
    assert model.pos_enc == 'learned' and model.tokenizer == 'cell'
    b, offset = prefix.shape
    heads = model.nhead
    dh = model.dmodel // heads
    cache = [(torch.empty(b, heads, offset+length, dh, device=prefix.device),
              torch.empty(b, heads, offset+length, dh, device=prefix.device))
             for _ in model.blocks]
    def logits(tokens, start):
        n = tokens.shape[1]
        x = model.embed(tokens) + model.pos[:, start:start+n]
        for block, (kc, vc) in zip(model.blocks, cache):
            h = block.ln1(x)
            q, k, v = torch.nn.functional.linear(h, block.attn.in_proj_weight,
                    block.attn.in_proj_bias).chunk(3, -1)
            q, k, v = [z.reshape(b, n, heads, dh).transpose(1, 2) for z in (q, k, v)]
            kc[:, :, start:start+n] = k
            vc[:, :, start:start+n] = v
            y = torch.nn.functional.scaled_dot_product_attention(
                q, kc[:, :, :start+n], vc[:, :, :start+n], is_causal=n > 1)
            x = x + block.attn.out_proj(y.transpose(1, 2).reshape(b, n, -1))
            x = x + block.ffn(block.ln2(x))
        return model.head(model.lnf(x)).squeeze(-1)
    z = logits(prefix, 0)
    outputs = []
    for i in range(length):
        token = (z[:, -1:] > 0).long()
        outputs.append(token)
        if i+1 < length:
            z = logits(token, offset+i)
    return torch.cat(outputs, 1)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_evaluation_corpus(name, grid, cache_dir='data/eval_corpora'):
    if name.startswith('l4b_'):
        # L4 corpora are already frozen. Never silently generate a new strict
        # corpus with L23 defaults or a different rare-event rejection stream.
        path = Path(cache_dir)/f'{name}_g{grid}.npz'
        with np.load(path) as z:
            corpus = {k:z[k] for k in z.files}
        expected = 768 if name == 'l4b_zs_scx' else 2048
        assert len(corpus['frames']) == expected
        from diagnose_structured import corpus_audit
        corpus_audit(corpus, l4=True, exercised=name.endswith('scx'),
                     half='train' if '_val_' in name else 'zs')
        return corpus
    return get_corpus(cache_dir, f'{name}_g{grid}', n=2048,
                      master_seed=44 if name == 'l3_zs_sc' else 43,
                      half='zs' if name == 'l3_zs_sc' else 'train',
                      grid=grid, self_consistent=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--official-csv')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--out', required=True)
    p.add_argument('--corpus', choices=['l3_zs_sc', 'l2_val_sc',
                   'l4b_zs_scx', 'l4b_zs_sc', 'l4b_val_sc'], default='l3_zs_sc')
    p.add_argument('--limit', type=int, default=0, help='diagnostic only; zero=full')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--sampling-seed', type=int, default=0)
    a = p.parse_args()
    if a.temperature < 0:
        p.error('temperature must be nonnegative')
    start = time.time()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=sha(a.checkpoint),
                  cawm_sha256=code_fingerprint(), evaluator_sha256=sha(__file__),
                  precision='fp32', temperature=a.temperature, limit=a.limit,
                  sampling_seed=a.sampling_seed, batch=a.batch,
                  sampling='argmax' if a.temperature == 0 else 'full_softmax_ancestral',
                  hostname=platform.node(), torch_version=str(torch.__version__),
                  device=a.device)
    generator = torch.Generator(device=a.device).manual_seed(a.sampling_seed)
    if a.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        result['gpu_name'] = torch.cuda.get_device_name()
    if a.official_csv:
        model = LifeGPT(vocab=256).to(a.device).eval()
        model.load_official(ck)
        rows = list(csv.DictReader(open(a.official_csv)))
        per_world = []
        for i, row in enumerate(rows):
            prompt = '@PredictNextState<' + row['State 1'] + '>'
            ids = torch.tensor([list(prompt.encode())], device=a.device)
            with torch.no_grad():
                pred = lifegpt_cached_generate(model, ids, 1029, a.temperature,
                                               generator)[0].cpu().tolist()
            text = bytes(pred).decode('utf8', errors='replace')
            try:
                cell_text = text.split('[', 1)[1].split(']', 1)[0]
            except IndexError:
                cell_text = ''
            truth = row['State 2']
            valid = len(cell_text) == len(truth) and set(cell_text) <= {'0', '1'}
            # Invalid/missing characters remain errors, never skipped.
            correct = sum(j < len(cell_text) and c == cell_text[j]
                          for j, c in enumerate(truth))
            per_world.append(dict(index=i, valid=valid, pixel_acc=correct/len(truth),
                                  frame_exact=valid and cell_text == truth,
                                  generated=text, target=truth))
            print(i, valid, correct/len(truth), flush=True)
        result.update(task='official_32x32_Conway', n=len(rows),
                      params=model.total_param_count(), corpus_sha256=sha(a.official_csv),
                      pixel_acc=float(np.mean([x['pixel_acc'] for x in per_world])),
                      frame_acc=float(np.mean([x['frame_exact'] for x in per_world])),
                      valid_outputs=sum(x['valid'] for x in per_world), worlds=per_world)
    else:
        args = ck['args']
        assert ck['step'] == args['steps'], 'formal evaluation requires final step'
        if a.corpus.startswith('l4b_'):
            assert args['task']=='L4' and args.get('l4v2')=='l4b'
        model = build_model(args['seed'], model=args['model'],
                            **model_kwargs(args)).to(a.device).eval()
        weights = 'model_ema' if 'model_ema' in ck else 'model'
        model.load_state_dict(ck[weights], strict=True)
        if a.temperature > 0:
            assert args['model'] == 'lifegpt', 'sampling follow-up is LifeGPT only'
            def sampled_rollout(prefix_pm1):
                b = prefix_pm1.shape[0]
                tokens = ((prefix_pm1[:, 0]+1)/2).long().reshape(b, -1)
                return lifegpt_cached_generate(model, tokens, 8*args['grid']**2,
                        a.temperature, generator).reshape(
                        b, 8, args['grid'], args['grid']).to(torch.uint8)
            model.rollout = sampled_rollout
        if args['model'] == 'art_vanilla':
            def cached_rollout(prefix_pm1):
                b = prefix_pm1.shape[0]
                tokens = ((prefix_pm1[:, 0]+1)/2).long().reshape(b, -1)
                return standard_cached_generate(model, tokens, 8*args['grid']**2).reshape(
                    b, 8, args['grid'], args['grid']).to(torch.uint8)
            model.rollout = cached_rollout
        corpus_name = f'{a.corpus}_g{args["grid"]}'
        corpus = load_evaluation_corpus(a.corpus, args['grid'])
        result['corpus_array_hashes'] = {k:hashlib.sha256(v.tobytes()).hexdigest()
                                         for k,v in corpus.items()}
        result['sampling_generator_state_sha256'] = hashlib.sha256(
            generator.get_state().cpu().numpy().tobytes()).hexdigest()
        if a.limit:
            corpus = {k: v[:a.limit] if hasattr(v, 'shape') and len(v.shape) else v
                      for k, v in corpus.items()}
        pred = rollout_corpus(model, corpus, device=a.device, batch=a.batch, bf16=False)
        target = corpus['frames'][:, 8:]
        # Independent NumPy exact reconstruction, asserted against shared metric.
        eq = pred == target
        exact = eq.reshape(len(pred), -1).all(1)
        reported = metrics(pred, corpus)
        assert reported['seq_acc'] == float(exact.mean())
        tf_correct, count = 0, 0
        log_prob = []
        with torch.no_grad():
            for i in range(0, len(pred), a.batch):
                fr = torch.from_numpy(corpus['frames'][i:i+a.batch]).to(a.device).float()
                logit = model((fr*2-1).unsqueeze(1))
                tf_correct += ((logit > 0) == fr[:, 8:]).sum().item()
                count += logit.numel()
                if a.temperature > 0:
                    log_prob.append(binary_trajectory_log_probability(
                        logit, fr[:, 8:], a.temperature).cpu().numpy())
        pred_path = out.with_suffix('.npz')
        extra = {}
        if log_prob:
            extra['trajectory_log_probability'] = np.concatenate(log_prob)
            result['expected_seq_acc'] = float(np.exp(extra['trajectory_log_probability']).mean())
        np.savez_compressed(pred_path, pred=pred, target=target, exact=exact, **extra)
        result.update(task=args['task'], corpus=corpus_name, checkpoint_step=ck['step'],
                      seed=args['seed'], params=model.total_param_count(), args=args,
                      weights=weights, metrics=reported, tf_pixel_acc=tf_correct/count,
                      frame_exact=eq.all(axis=(2, 3)).mean(0).tolist(),
                      prediction_file=str(pred_path), prediction_sha256=sha(pred_path),
                      corpus_sha256=sha(Path('data/eval_corpora')/(corpus_name+'.npz')))
    if a.device.startswith('cuda'):
        result['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
    result['wall_time_s'] = time.time()-start
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)
    print(json.dumps({k:v for k,v in result.items() if k not in ('worlds', 'args')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
