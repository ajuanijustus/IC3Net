#!/usr/bin/env python
"""Export training logs from a directory of checkpoints to JSON and CSV.

    python scripts/export_results.py <checkpoint_dir> <out_dir> [--final-k 50]

Writes
    <out_dir>/runs/<label>.json   one file per run: label, scenario, arch, seed, finished,
                                  args, meta, and every logged metric as a per-epoch list
    <out_dir>/summary.csv         one row per run: label, scenario, arch, seed, epochs,
                                  finished, final success / reward / steps (mean of the
                                  last K epochs), success AUC (mean over all epochs), git SHA

Per-agent metrics (reward, comm_action) are averaged over agents. Success stays a fraction in
[0, 1]. Snapshots (<label>_500 ...), module files (_comm.pt, _env.pt) and .done/.tmp
files are skipped.

Runs on the legacy stack (torch 0.4) or with a modern torch; it only needs this repo on the
path, because the logs are pickled as utils.LogField.
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, REPO)

import utils  # noqa: E402,F401  (needed to unpickle utils.LogField)

LABEL_RE = re.compile(r'^(?P<scenario>[a-z]+-[a-z0-9]+)__(?P<arch>[a-z0-9_]+)__s(?P<seed>\d+)$')
SKIP_RE = re.compile(r'(_\d+|_comm\.pt|_env\.pt|\.done|\.tmp|\.log|\.json|\.csv)$')


def load_checkpoint(path):
    return utils.torch_load(path)


def to_series(field):
    values = field.data if hasattr(field, 'data') else field
    out = []
    for v in values:
        arr = np.asarray(v.numpy() if hasattr(v, 'numpy') else v, dtype=float)
        out.append(float(arr.mean()) if arr.size else float('nan'))
    return out


def tail_mean(series, k):
    seg = [x for x in series[-k:] if x == x]
    return float(np.mean(seg)) if seg else float('nan')


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--final-k', type=int, default=50, help='epochs averaged for final values (default 50)')
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out_dir, 'runs'), exist_ok=True)
    rows = []
    for name in sorted(os.listdir(a.checkpoint_dir)):
        path = os.path.join(a.checkpoint_dir, name)
        if not os.path.isfile(path) or SKIP_RE.search(name):
            continue
        try:
            ckpt = load_checkpoint(path)
        except Exception as e:  # not a checkpoint
            print('skip {}: {}'.format(name, e))
            continue
        if not isinstance(ckpt, dict) or 'log' not in ckpt:
            continue

        meta = ckpt.get('meta', {}) or {}
        label = meta.get('run_label') or name
        m = LABEL_RE.match(label)
        scenario, arch, seed = (m.group('scenario'), m.group('arch'), int(m.group('seed'))) if m else (None, None, None)
        if seed is None and 'args' in ckpt:
            seed = ckpt['args'].get('seed')

        metrics = {k: to_series(v) for k, v in ckpt['log'].items() if k != 'epoch'}
        epochs = len(ckpt['log']['epoch'].data)
        finished = os.path.exists(path + '.done')

        record = {
            'label': label, 'scenario': scenario, 'arch': arch, 'seed': seed,
            'epochs': epochs, 'finished': finished,
            'args': jsonable(ckpt.get('args', {})), 'meta': jsonable(meta),
            'metrics': metrics,
        }
        with open(os.path.join(a.out_dir, 'runs', label + '.json'), 'w') as f:
            json.dump(record, f)

        success = metrics.get('success', [])
        rows.append({
            'label': label, 'scenario': scenario, 'arch': arch, 'seed': seed,
            'epochs': epochs, 'finished': finished,
            'final_success': tail_mean(success, a.final_k),
            'final_reward': tail_mean(metrics.get('reward', []), a.final_k),
            'final_steps': tail_mean(metrics.get('steps_taken', []), a.final_k),
            'final_comm_density': tail_mean(metrics.get('comm_density', []), a.final_k),
            'success_auc': float(np.nanmean(success)) if success else float('nan'),
            'git_sha': meta.get('git_sha'),
        })
        print('{:45s} {:5d} epochs  {}'.format(label, epochs, 'done' if finished else 'running/incomplete'))

    if rows:
        with open(os.path.join(a.out_dir, 'summary.csv'), 'w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print('{} runs exported to {}'.format(len(rows), a.out_dir))


if __name__ == '__main__':
    main()
