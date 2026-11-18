# import ctypes, sys, os

# # Force load ZeroMQ with global symbol visibility for dynamic extensions
# try:
#     # Try Conda environment libzmq first
#     ctypes.CDLL(os.path.join(sys.prefix, "lib", "libzmq.dylib"), mode=ctypes.RTLD_GLOBAL)
# except OSError:
#     try:
#         # Fallback to Homebrew libzmq
#         ctypes.CDLL("/opt/homebrew/lib/libzmq.dylib", mode=ctypes.RTLD_GLOBAL)
#     except OSError:
#         ctypes.CDLL("libzmq.dylib", mode=ctypes.RTLD_GLOBAL)

import os
import sys
import time
import random
import signal
import subprocess

import numpy as np
import torch
import visdom
import data
from models import *
from comm import CommNetMLP
from utils import *
from arguments import build_parser, prepare_args, finalize_args, build_policy
from trainer import Trainer
from multi_processing import MultiProcessTrainer

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = build_parser()

init_args_for_env(parser)
args = parser.parse_args()

# Seed every RNG before anything (envs included) draws from them.
if args.seed == -1:
    args.seed = np.random.randint(0, 10000)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_git_info():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir,
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip() != ''
        return sha, dirty
    except Exception:
        return None, None


git_sha, git_dirty = get_git_info()

prepare_args(args)

env = data.init(args.env_name, args, False)

finalize_args(args, env)
num_inputs = args.num_inputs

print(args)
print('Seed: {}  git: {}{}'.format(args.seed, git_sha, ' (dirty)' if git_dirty else ''))

policy_net = build_policy(args)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
print('Sharing parameters among threads, but not gradients.')
for p in policy_net.parameters():
    p.data.share_memory_()

# Resuming: read the existing checkpoint now, because the workers' seeds depend on
# where training resumes (they must not replay the random streams of the first part).
resume_state = None
args.resume_epoch = 0
if args.resume and args.save != '' and os.path.exists(args.save):
    resume_state = torch_load(args.save)
    args.resume_epoch = len(resume_state['log']['epoch'].data)

if args.nprocesses > 1:
    print('Initialising MultiProcessTrainer.')
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    print('Initialising Trainer.')
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()

log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['comm_density'] = LogField(list(), True, 'epoch', 'comm_pairs')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
# dual-pathway models (phases 2-5) report their losses per pathway
log['env_action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['comm_action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['env_value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['comm_value_loss'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env)

print('Initialisations complete.')

def run(num_epochs):
    print('Starting training interation.')
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            s = trainer.train_batch(ep)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)

        print('Epoch {}\tReward {}\tTime {:.2f}s'.format(
                epoch, stat['reward'], epoch_time
        ), flush=True)

        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.2f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'comm_density' in stat.keys():
            print('Comm-Density: {:.3f}'.format(stat['comm_density']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))

        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        # Snapshot named by the global epoch count, so names stay right after a resume.
        if args.save_every and args.save != '' and epoch % args.save_every == 0:
            save(args.save + '_' + str(epoch))

        if args.save != '':
            save(args.save)

def atomic_save(obj, path):
    # Write to a temp file then rename, so a job killed mid-write never leaves a
    # truncated checkpoint behind.
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save(path):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    d['args'] = dict(vars(args))
    d['meta'] = {
        'seed': args.seed,
        'git_sha': git_sha,
        'git_dirty': git_dirty,
        'worker_seeds': getattr(trainer, 'worker_seeds', []),
        'torch_version': torch.__version__,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'run_label': os.environ.get('RUN_LABEL'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
        'resumed_at_epochs': resumed_at_epochs,
    }
    # Main-process RNG states, restored on --resume. (Workers are re-seeded instead.)
    d['rng'] = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'random': random.getstate()}
    atomic_save(d, path)

    if args.phase == 5:
        atomic_save(policy_net.env_policy.state_dict(), path + '_env.pt')
        atomic_save(policy_net.comm_policy.state_dict(), path + '_comm.pt')

def load(path):
    d = torch_load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

resumed_at_epochs = []
if resume_state is not None:
    policy_net.load_state_dict(resume_state['policy_net'])
    log.update(resume_state['log'])
    trainer.load_state_dict(resume_state['trainer'])
    resumed_at_epochs = list(resume_state.get('meta', {}).get('resumed_at_epochs', [])) + [args.resume_epoch]
    if 'rng' in resume_state:
        torch.set_rng_state(resume_state['rng']['torch'])
        np.random.set_state(resume_state['rng']['numpy'])
        random.setstate(resume_state['rng']['random'])
    print('Resuming {} from epoch {}'.format(args.save, args.resume_epoch))
    del resume_state

if args.load != '':
    load(args.load)

run(max(args.num_epochs - args.resume_epoch, 0))

if args.display:
    env.end_display()

if args.save != '':
    save(args.save)

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
