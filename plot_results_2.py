#!/usr/bin/env python3
"""Plot aggregated learning curves and write summary CSVs for the AAMAS runs.

For every (environment, scenario):
  * loads each base checkpoint once and reads success / reward / steps logs,
  * pools phase5 into phase4 (identical architecture) and ic3net into phase1,
  * truncates all curves to the shortest condition budget in that scenario
    (the shortest condition's longest run), so conditions are compared at equal epochs,
  * saves a 3-panel figure and appends rows to three CSVs.

Outputs (in --save-dir):
  <env>_<scenario>_metrics.png
  summary_by_condition.csv   one row per (env, scenario, condition)
  summary_per_seed.csv       one row per run (use for CIs, noise floor, effect sizes)
  summary_curves.csv         learning curves every --curve-step epochs

Statistics are computed at the common budget; the full-budget final success is kept
alongside for transparency. State the truncation in Section 5 and in figure captions.

Usage:
  python plot_and_summarise.py
  python plot_and_summarise.py --final-k 50 --smooth 25 --common-epochs 1500
"""
import argparse
import csv
import os
import re
import warnings

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION, REGEX & LEGENDS
# ==========================================
MODELS_ROOT = "trained_models"
MODELS_DIRS = ["trained_models_aug_all_1", "trained_models_aug_all_2", "trained_models_aug_all_3",
               "trained_models_aug_all_temp_1", "trained_models_aug_all_temp_2", "trained_models_aug_all_temp_3",
               "trained_models_aug_all_sc_1", "trained_models_aug_all_sc_2", "trained_models_aug_all_sc_3"]
SAVE_DIR = "plots_aamas_sc_selected"

MODELS_DIRS = ["trained_models_aug_all_1", "trained_models_aug_all_2", "trained_models_aug_all_3",
               "trained_models_aug_all_temp_1", "trained_models_aug_all_temp_2", "trained_models_aug_all_temp_3"]
# MODELS_DIRS = ["trained_models_aug_all_sc_1", "trained_models_aug_all_sc_2", "trained_models_aug_all_sc_3"]
SAVE_DIR = "plots_aamas_sc_selected"

KEY_MAPPING = {
    'success': 'success',
    'reward': 'reward',
    'steps': 'steps_taken',
}

# Raw run key -> canonical condition (phase5 is the same architecture as phase4)
CANONICAL = {'ic3net': 'phase1', 'phase5': 'phase4'}

# CONDITION_ORDER = ['ic', 'iric', 'commnet', 'phase1', 'phase2', 'phase3', 'phase4']
CONDITION_ORDER = ['commnet', 'phase1', 'phase2', 'phase3', 'phase4']


LEGEND_MAPPING = {
    'ic': 'IC',
    'iric': 'IRIC',
    'commnet': 'CommNet',
    'phase1': 'IC3Net',
    'phase2': 'Gated-Shared',
    'phase3': 'Matrix-Shared',
    'phase4': 'Matrix-Separate',
}

ALGO_COLORS = {
    'commnet': '#5785c1',
    'phase1': '#dc8d6d',
    'iric': '#78b38a',
    'ic': '#ba71af',
    'phase2': '#e6b800',
    'phase3': '#df5454',
    'phase4': '#00670c',
}

model_pattern = re.compile(
    r"^(?P<env>predator_prey|traffic_junction|starcraft)_"
    r"(?P<algo>ic3net|commnet|iric|ic)_"
    r"(?P<param>.+?)_"
    r"(?:phase(?:(?P<phase>\d+)|na|none|null)_)?"
    r"(?P<job_id>job\d+_\d+_\d+)"
    r"(?:_(?P<epoch>\d+))?"
    r"(?:\..*)?$"
)

NAN = float('nan')


# ==========================================
# 2. SCAN CHECKPOINTS (base checkpoints only)
# ==========================================
def scan_checkpoints(root, dirs):
    raw = {}
    for d in dirs:
        full_dir = os.path.join(root, d)
        if not os.path.exists(full_dir):
            print(f"Warning: Directory '{full_dir}' not found. Skipping.")
            continue
        print(f"Scanning directory: {full_dir}")
        for item in os.listdir(full_dir):
            if item.endswith('_env.pt') or item.endswith('_comm.pt'):
                continue  # weights-only copies saved for phase 5
            m = model_pattern.match(item)
            if not m:
                continue
            meta = m.groupdict()
            if meta['epoch'] is not None:
                continue  # only the base checkpoint carries the full log
            algo = f"phase{meta['phase']}" if meta['phase'] else meta['algo']
            (raw.setdefault(meta['env'], {}).setdefault(meta['param'], {})
                .setdefault(algo, {}))[meta['job_id']] = os.path.join(full_dir, item)
    return raw


def organise(raw):
    out = {}
    for env, pd in raw.items():
        for param, ad in pd.items():
            for impl, jobs in ad.items():
                cond = CANONICAL.get(impl, impl)
                lst = out.setdefault(env, {}).setdefault(param, {}).setdefault(cond, [])
                for job_id, path in sorted(jobs.items()):
                    lst.append(dict(impl=impl, job_id=job_id, path=path))
    return out


# ==========================================
# 3. LOG PARSING
# ==========================================
def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # older torch has no weights_only argument
        return torch.load(path, map_location='cpu')


def extract_metric(log_data, metric_key):
    actual_key = KEY_MAPPING[metric_key]
    raw_field = log_data.get(actual_key, None)
    if raw_field is None:
        return None
    if hasattr(raw_field, 'data'):
        raw_values = raw_field.data
    elif isinstance(raw_field, tuple) and len(raw_field) == 4 and isinstance(raw_field[0], list):
        raw_values = raw_field[0]
    else:
        raw_values = raw_field
    if not isinstance(raw_values, list):
        return None
    cleaned = []
    for item in raw_values:
        if hasattr(item, 'numpy') or isinstance(item, (list, tuple, np.ndarray)):
            arr = item.numpy() if hasattr(item, 'numpy') else np.array(item)
            cleaned.append(0.0 if arr.size == 0 else float(np.mean(arr)))
        elif isinstance(item, (bool, int, float, np.number)):
            cleaned.append(float(item))
        else:
            try:
                cleaned.append(float(item))
            except Exception:
                cleaned.append(0.0)
    return cleaned


def load_run(entry):
    """Load one checkpoint once and return dict of metric arrays (success scaled to %)."""
    try:
        ckpt = load_checkpoint(entry['path'])
    except Exception as e:
        print(f"  Warning: Could not read {entry['path']}. Skipping. Error: {e}")
        return None
    log = ckpt.get('log') if isinstance(ckpt, dict) else None
    if log is None:
        return None
    data = {}
    for metric in KEY_MAPPING:
        vec = extract_metric(log, metric)
        data[metric] = None if not vec else np.asarray(vec, dtype=np.float64)
    if data['success'] is None:
        return None
    data['success'] = data['success'] * 100.0
    n = len(data['success'])
    for metric in ('reward', 'steps'):
        if data[metric] is None:
            data[metric] = np.full(n, NAN)
    return data


# ==========================================
# 4. STATISTICS HELPERS
# ==========================================
def trailing_mean(x, w):
    x = np.asarray(x, dtype=np.float64)
    if w <= 1 or len(x) == 0:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    n = np.arange(1, len(x) + 1)
    lo = np.maximum(0, n - w)
    return (c[n] - c[lo]) / (n - lo)


def tail_mean(x, upto, k):
    seg = np.asarray(x)[:upto]
    if len(seg) == 0:
        return NAN
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return float(np.nanmean(seg[-k:]))


def conv_epoch(curve, final, w, frac, floor):
    """First epoch (1-indexed, >= w) at which the trailing-mean curve reaches frac*final."""
    if not np.isfinite(final) or final < floor or len(curve) < w:
        return NAN
    sm = trailing_mean(curve, w)
    hit = np.where(sm[w - 1:] >= frac * final)[0]
    return float(hit[0] + w) if len(hit) else NAN


def late_gain(curve, window):
    """Mean of the last half-window minus the mean of the half-window before it."""
    if len(curve) < window or window < 2:
        return NAN
    half = window // 2
    return float(np.mean(curve[-half:]) - np.mean(curve[-window:-half]))


def agg(values):
    v = np.array([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return dict(mean=NAN, std=NAN, median=NAN, min=NAN, max=NAN, q25=NAN, q75=NAN, n=0)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else NAN,
                median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), n=len(v))


def padded_matrix(arrs, length):
    m = np.full((len(arrs), length), NAN, dtype=np.float64)
    for i, a in enumerate(arrs):
        n = min(len(a), length)
        m[i, :n] = a[:n]
    return m


def mean_std(m):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean = np.nanmean(m, axis=0)
        std = np.nanstd(m, axis=0, ddof=1)
    n = np.sum(~np.isnan(m), axis=0)
    std = np.where(n > 1, std, 0.0)
    return mean, std, n


def fmt(x, nd=3):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


# ==========================================
# 5. PER-SCENARIO PROCESSING
# ==========================================
def ordered_conditions(conds):
    return [c for c in CONDITION_ORDER if c in conds] # + [c for c in conds if c not in CONDITION_ORDER]


def run_stats(run, common, args):
    s = run['success']
    length = len(s)
    st = dict(epochs_run=length, complete=length >= common,
              final_success_full=tail_mean(s, length, args.final_k))
    keys = ('final_success_common', 'final_reward_common', 'final_steps_common',
            'conv_epoch', 'mean_success', 'late_gain')
    st.update({k: NAN for k in keys})
    if st['complete']:
        sc = s[:common]
        st['final_success_common'] = tail_mean(sc, common, args.final_k)
        st['final_reward_common'] = tail_mean(run['reward'], common, args.final_k)
        st['final_steps_common'] = tail_mean(run['steps'], common, args.final_k)
        st['conv_epoch'] = conv_epoch(sc, st['final_success_common'], args.smooth,
                                      args.conv_frac, args.conv_floor)
        st['mean_success'] = float(np.mean(sc))
        st['late_gain'] = late_gain(sc, args.late_window)
    return st


def plot_scenario(env, param, conds, runs_by_cond, common, plot_len, truncated, args, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['success', 'reward', 'steps']
    titles = ['Success Rate (%)', 'Total Reward', 'Steps Taken']
    plotted = False
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for cond in conds:
            arrs = [r[metric] for r in runs_by_cond[cond]]
            if args.plot_smooth > 1:
                arrs = [trailing_mean(a, args.plot_smooth) for a in arrs]
            m = padded_matrix(arrs, plot_len)
            if np.all(np.isnan(m)):
                continue
            plotted = True
            mean, std, _ = mean_std(m)
            x = np.arange(1, plot_len + 1)
            color = ALGO_COLORS.get(cond, '#7f7f7f')
            ax.plot(x, mean, label=LEGEND_MAPPING.get(cond, cond.upper()), color=color, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(titles[idx], fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', frameon=True)
    if plotted:
        env_title = env.replace('_', ' ').title()
        param_title = f"Map: {param}" if env == 'starcraft' else (param.upper() if param == 'na' else param.title())
        plt.suptitle(f"Aggregated Paper Results: {env_title} ({param_title})",
                     fontsize=16, fontweight='bold', y=1.02)
        # if truncated and not args.no_budget_note:
        #     fig.text(0.5, -0.02, f"Curves truncated to {plot_len} epochs "
        #                          f"(shortest training budget among conditions in this scenario).",
        #              ha='center', fontsize=10, style='italic')
        plt.tight_layout()
        path = os.path.join(save_dir, f"{env}_{param}_metrics.png")
        plt.savefig(path, bbox_inches='tight', dpi=200)
        print(f"  Saved plot: {path}")
    plt.close(fig)


def process_scenario(env, param, cond_entries, args, save_dir, per_seed_rows, cond_rows, curve_rows):
    runs_by_cond, entries_by_cond = {}, {}
    for cond, entries in cond_entries.items():
        loaded, kept = [], []
        for e in entries:
            r = load_run(e)
            if r is not None:
                loaded.append(r)
                kept.append(e)
        if loaded:
            runs_by_cond[cond] = loaded
            entries_by_cond[cond] = kept
    if not runs_by_cond:
        return
    conds = ordered_conditions(list(runs_by_cond))

    # ---- budgets
    cond_max = {c: max(len(r['success']) for r in runs_by_cond[c]) for c in conds}
    cond_min = {c: min(len(r['success']) for r in runs_by_cond[c]) for c in conds}
    shortest = min(cond_max, key=cond_max.get)
    common = cond_max[shortest]
    if args.common_epochs:
        common = min(common, args.common_epochs)
    max_len = max(cond_max.values())
    truncated = common < max_len
    plot_len = common if not args.no_truncate else max_len

    print(f"\n== {env} / {param} ==")
    print(f"  {'condition':<16}{'runs':>5}{'epochs_min':>12}{'epochs_max':>12}")
    for c in conds:
        print(f"  {LEGEND_MAPPING.get(c, c):<16}{len(runs_by_cond[c]):>5}{cond_min[c]:>12}{cond_max[c]:>12}")
    print(f"  common budget: {common} epochs (set by {LEGEND_MAPPING.get(shortest, shortest)})")
    for c in conds:
        short = [e['job_id'] for e, r in zip(entries_by_cond[c], runs_by_cond[c]) if len(r['success']) < common]
        if short:
            print(f"  WARNING: {LEGEND_MAPPING.get(c, c)} has runs shorter than the common budget "
                  f"(excluded from common-budget statistics): {short}")

    # ---- plot
    plot_scenario(env, param, conds, runs_by_cond, common, plot_len, truncated, args, save_dir)

    # ---- per-seed and per-condition rows
    for c in conds:
        name = LEGEND_MAPPING.get(c, c.upper())
        stats = []
        for e, r in zip(entries_by_cond[c], runs_by_cond[c]):
            st = run_stats(r, common, args)
            stats.append((e, st))
            per_seed_rows.append(dict(
                env=env, scenario=param, condition_key=c, condition=name, impl_key=e['impl'],
                job_id=e['job_id'], epochs_run=st['epochs_run'], complete_at_common=int(st['complete']),
                common_epochs=common,
                final_success_common=fmt(st['final_success_common']),
                final_success_full=fmt(st['final_success_full']),
                final_reward_common=fmt(st['final_reward_common']),
                final_steps_common=fmt(st['final_steps_common']),
                conv_epoch=fmt(st['conv_epoch'], 0),
                mean_success_over_training=fmt(st['mean_success']),
                late_gain=fmt(st['late_gain']), path=e['path']))

        sts = [s for _, s in stats]
        fs, ff = agg([s['final_success_common'] for s in sts]), agg([s['final_success_full'] for s in sts])
        fr, fp = agg([s['final_reward_common'] for s in sts]), agg([s['final_steps_common'] for s in sts])
        cv, ms, lg = agg([s['conv_epoch'] for s in sts]), agg([s['mean_success'] for s in sts]), agg([s['late_gain'] for s in sts])
        eligible = sum(1 for s in sts if np.isfinite(s['final_success_common']) and s['final_success_common'] >= args.conv_floor)
        impl_means = {}
        for e, s in stats:
            if np.isfinite(s['final_success_common']):
                impl_means.setdefault(e['impl'], []).append(s['final_success_common'])
        impl_str = '|'.join(f"{k}:n={len(v)},mean={np.mean(v):.2f}" for k, v in sorted(impl_means.items()))
        cond_rows.append(dict(
            env=env, scenario=param, condition_key=c, condition=name, n_runs=len(sts), n_complete=fs['n'],
            epochs_min=cond_min[c], epochs_max=cond_max[c], common_epochs=common,
            final_success_common_mean=fmt(fs['mean']), final_success_common_std=fmt(fs['std']),
            final_success_common_median=fmt(fs['median']), final_success_common_min=fmt(fs['min']),
            final_success_common_max=fmt(fs['max']),
            final_success_full_mean=fmt(ff['mean']), final_success_full_std=fmt(ff['std']),
            final_reward_common_mean=fmt(fr['mean']), final_reward_common_std=fmt(fr['std']),
            final_steps_common_mean=fmt(fp['mean']), final_steps_common_std=fmt(fp['std']),
            conv_n_eligible=eligible, conv_n_reached=cv['n'],
            conv_epoch_median=fmt(cv['median'], 0), conv_epoch_q25=fmt(cv['q25'], 0), conv_epoch_q75=fmt(cv['q75'], 0),
            mean_success_over_training_mean=fmt(ms['mean']), mean_success_over_training_std=fmt(ms['std']),
            late_gain_mean=fmt(lg['mean']), impl_breakdown=impl_str))

        # ---- learning curves (trailing mean per seed, then across seeds)
        sm = [trailing_mean(r['success'][:common], args.curve_win) for r in runs_by_cond[c]]
        rw = [trailing_mean(r['reward'][:common], args.curve_win) for r in runs_by_cond[c]]
        sp = [trailing_mean(r['steps'][:common], args.curve_win) for r in runs_by_cond[c]]
        ms_, ss_, ns_ = mean_std(padded_matrix(sm, common))
        mr_, sr_, _ = mean_std(padded_matrix(rw, common))
        mp_, _, _ = mean_std(padded_matrix(sp, common))
        epochs = list(range(args.curve_step, common + 1, args.curve_step))
        if not epochs or epochs[-1] != common:
            epochs.append(common)
        for ep in epochs:
            i = ep - 1
            curve_rows.append(dict(env=env, scenario=param, condition_key=c, condition=name, epoch=ep,
                                   n_seeds=int(ns_[i]), success_mean=fmt(ms_[i]), success_std=fmt(ss_[i]),
                                   reward_mean=fmt(mr_[i]), reward_std=fmt(sr_[i]), steps_mean=fmt(mp_[i])))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


# ==========================================
# 6. MAIN
# ==========================================
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=MODELS_ROOT)
    ap.add_argument('--dirs', nargs='*', default=MODELS_DIRS)
    ap.add_argument('--save-dir', default=SAVE_DIR)
    ap.add_argument('--envs', nargs='*', default=None, help='restrict to these environments')
    ap.add_argument('--final-k', type=int, default=50, help='final value = mean of last K epochs (default 50)')
    ap.add_argument('--smooth', type=int, default=25, help='trailing window for convergence detection (default 25)')
    ap.add_argument('--conv-frac', type=float, default=0.9, help='convergence = reach this fraction of final success')
    ap.add_argument('--conv-floor', type=float, default=10.0, help='skip convergence if final success (%%) is below this')
    ap.add_argument('--late-window', type=int, default=250, help='window for late-gain (second half minus first half)')
    ap.add_argument('--curve-step', type=int, default=50, help='epoch spacing in summary_curves.csv')
    ap.add_argument('--curve-win', type=int, default=25, help='trailing mean window for summary_curves.csv')
    ap.add_argument('--plot-smooth', type=int, default=1, help='trailing mean window for plotted curves (1 = none)')
    ap.add_argument('--common-epochs', type=int, default=None, help='cap the common budget at this many epochs')
    ap.add_argument('--no-truncate', action='store_true', help='plot full-length curves (CSVs stay at the common budget)')
    ap.add_argument('--no-budget-note', action='store_true', help='omit the truncation note under the figure')
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    raw = scan_checkpoints(args.root, args.dirs)
    dataset = organise(raw)
    total = sum(len(v) for p in dataset.values() for c in p.values() for v in c.values())
    print(f"\nFound {total} unique base runs across all directories.")
    print(f"Settings: final_k={args.final_k}, smooth={args.smooth}, conv_frac={args.conv_frac}, "
          f"conv_floor={args.conv_floor}, late_window={args.late_window}")

    per_seed_rows, cond_rows, curve_rows = [], [], []
    for env in sorted(dataset):
        if args.envs and env not in args.envs:
            continue
        for param in sorted(dataset[env]):
            process_scenario(env, param, dataset[env][param], args, args.save_dir,
                             per_seed_rows, cond_rows, curve_rows)

    print()
    write_csv(os.path.join(args.save_dir, 'summary_by_condition.csv'), cond_rows)
    write_csv(os.path.join(args.save_dir, 'summary_per_seed.csv'), per_seed_rows)
    write_csv(os.path.join(args.save_dir, 'summary_curves.csv'), curve_rows)
    print("\nDone.")


if __name__ == '__main__':
    main()