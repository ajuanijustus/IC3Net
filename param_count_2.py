#!/usr/bin/env python3
"""Count parameters in IC3Net-style checkpoints.

Only state dicts are read, so no environment or model code is needed.

Usage:
    python count_params.py RUNS_DIR [MORE_DIRS_OR_FILES ...] [--csv out.csv] [--breakdown]

Checkpoint layout assumed (IC3Net main.py): torch.save({'policy_net': state_dict,
'log': ..., 'trainer': ...}, path). Top-level state dicts and the keys 'model' /
'state_dict' / 'net' are also accepted. A leading 'module.' (DataParallel) is stripped.

Environment, scenario and condition are inferred from the file path (heuristic, see
infer_* below). Unrecognised parts are shown as '?'; edit the helpers if your folder
names differ.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import torch

EXTS = ('.pt', '.pth', '.ckpt', '.tar')

PAPER_NAMES = {
    'ic': 'IC',
    'iric': 'IRIC',
    'commnet': 'CommNet',
    'phase1': 'IC3Net',
    'phase2': 'Gated-Shared',
    'phase3': 'Matrix-Shared',
    'phase4': 'Matrix-Separate',
    'phase5': 'Matrix-Separate',  # pooled with phase4
}
ENVS = ('traffic_junction', 'predator_prey', 'starcraft')


# ----------------------------------------------------------------- loading
def _find_tensor_dict(obj):
    """Recursively return the first non-empty dict whose values are all tensors."""
    if isinstance(obj, dict):
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
        for v in obj.values():
            r = _find_tensor_dict(v)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            r = _find_tensor_dict(v)
            if r is not None:
                return r
    return None


def load_state_dict(path):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # older torch without the weights_only argument
        ckpt = torch.load(path, map_location='cpu')

    sd = None
    if isinstance(ckpt, dict):
        for key in ('policy_net', 'model', 'state_dict', 'net'):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        if sd is None and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            sd = ckpt
    elif hasattr(ckpt, 'state_dict'):
        sd = ckpt.state_dict()

    if sd is None:  # fallback: first nested dict whose values are all tensors
        sd = _find_tensor_dict(ckpt)

    if sd is None:
        raise ValueError('no state dict found (top-level keys: %s)'
                         % (list(ckpt.keys())[:8] if isinstance(ckpt, dict) else type(ckpt)))
    return {(k[7:] if k.startswith('module.') else k): v
            for k, v in sd.items() if torch.is_tensor(v)}


def find_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in names:
                    if n.lower().endswith(EXTS):
                        files.append(os.path.join(root, n))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print('WARNING: not found: %s' % p, file=sys.stderr)
    return sorted(files)


# ---------------------------------------------------------------- counting
def group_of(key):
    parts = key.split('.')
    if parts[0] in ('comm_policy', 'env_policy') and len(parts) > 2:
        return '.'.join(parts[:2])
    return parts[0]


def summarise(sd):
    total = sum(v.numel() for v in sd.values())
    value = sum(v.numel() for k, v in sd.items() if 'value_head' in k)
    groups = defaultdict(int)
    for k, v in sd.items():
        groups[group_of(k)] += v.numel()
    # LSTMCell recurrent weights: shape (4*hid, hid)
    lstm_keys = sorted(k for k in sd if k.endswith('weight_hh'))
    hid = sd[lstm_keys[0]].shape[1] if lstm_keys else None
    return dict(total=total, inference=total - value, value=value,
                groups=dict(groups), lstm_cores=len(lstm_keys), hid=hid)


# ----------------------------------------------------------- path labelling
def infer_env(path):
    p = path.lower()
    for e in ENVS:
        if e in p:
            return e
    if 'traffic' in p:
        return 'traffic_junction'
    if 'predator' in p or 'prey' in p:
        return 'predator_prey'
    if 'starcraft' in p or 'torchcraft' in p:
        return 'starcraft'
    return '?'


def infer_scenario(path):
    m = re.search(r'(easy|medium|hard|explore\w*|combat\w*)', path.lower())
    return m.group(1) if m else '?'


def infer_condition(path):
    tokens = [t for t in re.split(r'[\\/\-\.\s_]+', path.lower()) if t]
    for t in tokens:
        if re.fullmatch(r'phase[1-5]', t):
            return t
    for t in tokens:
        if t in ('commnet', 'iric', 'ic'):
            return t
        if t == 'ic3net':
            return 'phase1'
    return '?'


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='checkpoint files or directories to search')
    ap.add_argument('--csv', help='write one row per checkpoint to this CSV')
    ap.add_argument('--breakdown', action='store_true',
                    help='print per-module parameter counts for one example of each configuration')
    args = ap.parse_args()

    files = find_files(args.paths)
    if not files:
        sys.exit('No checkpoint files found.')

    rows, examples = [], {}
    for f in files:
        try:
            sd = load_state_dict(f)
        except Exception as e:
            print('SKIP %s: %s' % (f, e), file=sys.stderr)
            continue
        s = summarise(sd)
        cond = infer_condition(f)
        row = dict(path=f, env=infer_env(f), scenario=infer_scenario(f), phase_key=cond,
                   condition=PAPER_NAMES.get(cond, '?'), total=s['total'],
                   inference=s['inference'], value_heads=s['value'],
                   lstm_cores=s['lstm_cores'], hid=s['hid'])
        rows.append(row)
        examples.setdefault((row['env'], row['scenario'], row['condition'], row['total']),
                            (f, s['groups']))

    # ---- per-file table
    print('\nPer checkpoint (%d files)' % len(rows))
    print('%-8s %-10s %-16s %10s %10s %6s %5s  %s' %
          ('env', 'scenario', 'condition', 'total', 'inference', 'cores', 'hid', 'path'))
    for r in rows:
        print('%-8s %-10s %-16s %10d %10d %6d %5s  %s' %
              (r['env'][:8], r['scenario'][:10], r['condition'], r['total'], r['inference'],
               r['lstm_cores'], r['hid'], r['path']))

    # ---- grouped summary
    grouped = defaultdict(int)
    for r in rows:
        grouped[(r['env'], r['scenario'], r['condition'], r['phase_key'], r['total'],
                 r['inference'], r['lstm_cores'], r['hid'])] += 1

    ref = {}
    for (env, scen, cond, _pk, total, _inf, _c, _h) in grouped:
        if cond == 'Matrix-Shared':
            ref[(env, scen)] = total

    print('\nDistinct configurations (ratio is total params relative to Matrix-Shared in the same scenario)')
    print('%-16s %-10s %-16s %-7s %10s %10s %6s %5s %7s %6s' %
          ('env', 'scenario', 'condition', 'phase', 'total', 'inference', 'cores', 'hid',
           'ratio', 'files'))
    for key in sorted(grouped):
        env, scen, cond, pk, total, inf, cores, hid = key
        base = ref.get((env, scen))
        ratio = '%.3f' % (total / base) if base else '-'
        print('%-16s %-10s %-16s %-7s %10d %10d %6d %5s %7s %6d' %
              (env, scen, cond, pk, total, inf, cores, hid, ratio, grouped[key]))

    if args.breakdown:
        print('\nPer-module breakdown (one example per configuration)')
        for (env, scen, cond, total), (f, groups) in sorted(examples.items()):
            print('\n%s / %s / %s  total=%d  (%s)' % (env, scen, cond, total, f))
            for g, n in sorted(groups.items(), key=lambda kv: -kv[1]):
                print('    %-32s %10d  (%.1f%%)' % (g, n, 100.0 * n / total))

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print('\nWrote %s' % args.csv)

    unlabeled = [r['path'] for r in rows if '?' in (r['env'], r['condition'])]
    if unlabeled:
        print('\nWARNING: %d checkpoints had an unrecognised env or condition in the path, e.g. %s'
              % (len(unlabeled), unlabeled[0]), file=sys.stderr)


if __name__ == '__main__':
    main()