#!/usr/bin/env python
"""Record reference fixtures on the legacy stack, for exact-equivalence tests of the modern port.

    python scripts/record_fixtures.py <out_dir> [--seeds 0 1 2] [--episodes 3] [--hid 32]

Must run on the legacy stack (Python 3.6, torch 0.4, gym 0.9.6, numpy 1.13). Writes plain
numpy arrays only (no pickled objects), so any later numpy can read them:

  <out_dir>/env/<scenario>__seed<k>.npz
      Per episode e: obs (T+1, n, d), actions (T, n), rewards (T, n), done (T,), and every
      numeric info array (e.g. alive_mask, is_completed, car_loc, predator_locs), plus the
      terminal reward, final success and add_rate. Actions come from a separate RandomState,
      so the env's own use of the global numpy RNG is exactly what training sees. TJ episodes
      reset at several curriculum epochs.

  <out_dir>/model/<scenario>__<arch>.npz
      One training batch from the real Trainer for each architecture: the initial state dict
      (sd__*), and per forward pass the inputs (state, hidden, alive mask, phase-1 gate), the
      outputs (log-probs per head, values, next hidden) and the executed gates. Also the
      recorded actions, rewards, masks and the raw gradients from compute_grad.

  <out_dir>/manifest.json
      Library versions, git SHA and settings.
"""
import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO)

import gym  # noqa: E402
import ic3net_envs  # noqa: E402,F401  (registers the envs)
import data  # noqa: E402
from arguments import build_parser, prepare_args, finalize_args, build_policy  # noqa: E402
from trainer import Trainer  # noqa: E402

torch.set_default_tensor_type('torch.DoubleTensor')

ENV_IDS = {'predator_prey': 'PredatorPrey-v0', 'traffic_junction': 'TrafficJunction-v0'}
ENV_SCENARIOS = ['pp-easy', 'pp-medium', 'pp-hard', 'tj-easy', 'tj-medium', 'tj-hard']
MODEL_SCENARIOS = ['pp-easy', 'tj-easy']
TJ_RESET_EPOCHS = [0, 400, 800, 1300]  # before, during and after the curriculum

_spec = importlib.util.spec_from_file_location('make_configs', os.path.join(REPO, 'configs', 'make_configs.py'))
make_configs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_configs)
ARCHS = [(a, f) for a, f, _ in make_configs.ARCHS] + [('matrix_separate_phase4', '--ic3net --phase 4')]


def parse_args(flags):
    """Parse a flag string exactly as main.py would (training + env arguments)."""
    parser = build_parser()
    env_name = flags.split('--env_name')[1].split()[0]
    gym.make(ENV_IDS[env_name]).init_args(parser)
    args = parser.parse_args(flags.split())
    return prepare_args(args)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compact(arr):
    """Store integer-valued arrays as int16 to keep fixtures small."""
    arr = np.asarray(arr)
    if arr.dtype.kind == 'f' and arr.size and np.all(arr == np.round(arr)) and np.abs(arr).max() < 32000:
        return arr.astype(np.int16)
    return arr


def to_np(x):
    if isinstance(x, (list, tuple)):
        return [to_np(v) for v in x]
    if hasattr(x, 'detach'):
        return x.detach().numpy().copy()
    return np.asarray(x).copy()


def record_env(scen, seed, episodes, out_dir):
    base, flags, _ = make_configs.SCENARIOS[scen]
    args = parse_args('{} {} --seed {}'.format(base, flags, seed))
    seed_all(seed)
    env = data.init(args.env_name, args)
    act_rng = np.random.RandomState(10000 + seed)
    n, nact = args.nagents, env.num_actions
    reset_epochs = TJ_RESET_EPOCHS if args.env_name == 'traffic_junction' else [0]

    arrays = {}
    for e in range(episodes):
        epoch = reset_epochs[e % len(reset_epochs)]
        obs = [to_np(env.reset(epoch))[0]]
        actions, rewards, dones, infos = [], [], [], {}
        for t in range(args.max_steps):
            a = act_rng.randint(nact, size=n)
            next_obs, reward, done, info = env.step([a])
            done = bool(done) or t == args.max_steps - 1
            obs.append(to_np(next_obs)[0])
            actions.append(a)
            rewards.append(np.asarray(reward, dtype=float))
            dones.append(done)
            for k, v in info.items():
                v = np.asarray(v)
                if v.dtype.kind in 'biuf':
                    infos.setdefault(k, []).append(v.copy())
            if done:
                break
        terminal = np.asarray(env.reward_terminal(), dtype=float)
        stat = env.get_stat()
        p = 'ep{}_'.format(e)
        arrays[p + 'obs'] = compact(np.stack(obs))
        arrays[p + 'actions'] = np.stack(actions).astype(np.int16)
        arrays[p + 'rewards'] = np.stack(rewards)
        arrays[p + 'done'] = np.array(dones)
        arrays[p + 'terminal_reward'] = terminal
        arrays[p + 'reset_epoch'] = np.array(epoch)
        arrays[p + 'success'] = np.array(float(stat.get('success', np.nan)))
        if 'add_rate' in stat:
            arrays[p + 'add_rate'] = np.array(float(stat['add_rate']))
        for k, vs in infos.items():
            try:
                arrays[p + 'info_' + k] = compact(np.stack(vs))
            except ValueError:
                pass  # ragged; skip
    meta = {'scenario': scen, 'seed': seed, 'episodes': episodes, 'flags': '{} {}'.format(base, flags),
            'nagents': n, 'num_actions': int(nact), 'observation_dim': int(env.observation_dim),
            'action_seed': 10000 + seed, 'max_steps': args.max_steps}
    arrays['meta'] = np.array(json.dumps(meta))
    path = os.path.join(out_dir, 'env', '{}__seed{}.npz'.format(scen, seed))
    np.savez_compressed(path, **arrays)
    return path


def record_model(scen, arch, arch_flags, hid, out_dir):
    base, flags, _ = make_configs.SCENARIOS[scen]
    args = parse_args('{} {} {} --hid_size {} --recurrent --detach_gap 10 --nprocesses 1 '
                      '--batch_size 60 --epoch_size 1 --seed 0'.format(base, flags, arch_flags, hid))
    # Same order as main.py: seed, env for argument shapes, finalise, build, trainer env.
    seed_all(args.seed)
    finalize_args(args, data.init(args.env_name, args, False))
    policy_net = build_policy(args)
    state_dict = {k: to_np(v) for k, v in policy_net.state_dict().items()}

    steps = []
    orig_forward = policy_net.forward

    def recording_forward(x, info={}):
        out = orig_forward(x, info)
        rec = {}
        if args.recurrent:
            rec['state'] = to_np(x[0])
            hid_in = x[1]
            if isinstance(hid_in, (list, tuple)):
                rec['h_in'], rec['c_in'] = to_np(hid_in[0]), to_np(hid_in[1])
            else:
                rec['h_in'] = to_np(hid_in)
            action_out, value, hid_out = out
            if isinstance(hid_out, (list, tuple)):
                rec['h_out'], rec['c_out'] = to_np(hid_out[0]), to_np(hid_out[1])
            else:
                rec['h_out'] = to_np(hid_out)
        else:
            rec['state'] = to_np(x)
            action_out, value = out
        for i, lp in enumerate(action_out):
            rec['logp_{}'.format(i)] = to_np(lp)
        if isinstance(value, (list, tuple)):
            rec['value_env'], rec['value_comm'] = to_np(value[0]), to_np(value[1])
        else:
            rec['value'] = to_np(value)
        rec['info_alive'] = np.asarray(info.get('alive_mask', np.ones(args.nagents)), dtype=float)
        if args.phase == 1 and args.hard_attn and args.commnet:
            # phase 1 applies the gate passed in via info (chosen at the previous step)
            rec['info_comm_action'] = np.asarray(info['comm_action'], dtype=float)
        gates = getattr(policy_net, 'last_comm_action', None)
        if gates is not None:
            rec['gates'] = np.stack([to_np(g) for g in gates])
        steps.append(rec)
        return out

    policy_net.forward = recording_forward
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))
    batch, stat = trainer.run_batch(0)
    trainer.optimizer.zero_grad()
    grad_stat = trainer.compute_grad(batch)

    arrays = {'sd__' + k: v for k, v in state_dict.items()}
    for key in steps[0]:
        arrays['step__' + key] = np.stack([s[key] for s in steps])
    arrays['batch__actions'] = np.array([np.stack([np.asarray(h) for h in a]) for a in batch.action])
    arrays['batch__rewards'] = np.array(batch.reward, dtype=float)
    arrays['batch__episode_mask'] = np.array(batch.episode_mask, dtype=float)
    arrays['batch__episode_mini_mask'] = np.array(batch.episode_mini_mask, dtype=float)
    arrays['batch__alive_mask'] = np.array([m['alive_mask'] for m in batch.misc], dtype=float)
    for name, p in policy_net.named_parameters():
        if p.grad is not None:
            arrays['grad__' + name] = to_np(p.grad)
    meta = {'scenario': scen, 'arch': arch, 'arch_flags': arch_flags,
            'args': {k: (v if isinstance(v, (bool, int, float, str, list, type(None))) else str(v))
                     for k, v in vars(args).items()},
            'num_steps': len(steps), 'grad_stat': {k: float(v) for k, v in grad_stat.items()},
            'note': 'grads are raw sums from compute_grad, before division by num_steps'}
    arrays['meta'] = np.array(json.dumps(meta))
    path = os.path.join(out_dir, 'model', '{}__{}.npz'.format(scen, arch))
    np.savez_compressed(path, **arrays)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--hid', type=int, default=32, help='hidden size for model fixtures (default 32)')
    a = ap.parse_args()
    for sub in ('env', 'model'):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)

    for scen in ENV_SCENARIOS:
        for seed in a.seeds:
            print('env  ', record_env(scen, seed, a.episodes, a.out_dir), flush=True)
    for scen in MODEL_SCENARIOS:
        for arch, flags in ARCHS:
            print('model', record_model(scen, arch, flags, a.hid, a.out_dir), flush=True)

    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
    except Exception:
        sha = None
    manifest = {'git_sha': sha, 'python': sys.version.split()[0], 'torch': torch.__version__,
                'numpy': np.__version__, 'gym': gym.__version__, 'seeds': a.seeds,
                'episodes': a.episodes, 'hid': a.hid, 'env_scenarios': ENV_SCENARIOS,
                'model_scenarios': MODEL_SCENARIOS, 'archs': [x[0] for x in ARCHS]}
    with open(os.path.join(a.out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('Fixtures written to', a.out_dir)


if __name__ == '__main__':
    main()
