"""Command-line arguments and the argument post-processing shared by main.py and scripts.

Kept separate from main.py (which runs training on import) so that tools such as
scripts/record_fixtures.py build models with exactly the same settings as training.
"""
import argparse

from action_utils import parse_action_args
from comm import CommNetMLP
from models import MLP, RNN, Random


def build_parser():
    """The training argument parser (environment-specific arguments are added separately)."""
    parser = argparse.ArgumentParser(description='PyTorch RL trainer')
    # training
    # note: number of steps per epoch = epoch_size X batch_size x nprocesses
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--epoch_size', type=int, default=10,
                        help='number of update iterations in an epoch')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='number of steps before each update (per thread)')
    parser.add_argument('--nprocesses', type=int, default=16,
                        help='How many processes to run')
    # model
    parser.add_argument('--hid_size', default=64, type=int,
                        help='hidden layer size')
    parser.add_argument('--recurrent', action='store_true', default=False,
                        help='make the model recurrent in time')
    # optimization
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='discount factor')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='gae (remove?)')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed. Pass -1 for random seed') # TODO: works in thread?
    parser.add_argument('--normalize_rewards', action='store_true', default=False,
                        help='normalize rewards in each batch')
    parser.add_argument('--lrate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--entr', type=float, default=0,
                        help='entropy regularization coeff')
    parser.add_argument('--value_coeff', type=float, default=0.01,
                        help='coeff for value loss term')
    # environment
    parser.add_argument('--env_name', default="Cartpole",
                        help='name of the environment to run')
    parser.add_argument('--max_steps', default=20, type=int,
                        help='force to end the game after this many steps')
    parser.add_argument('--nactions', default='1', type=str,
                        help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
    parser.add_argument('--action_scale', default=1.0, type=float,
                        help='scale action output from model')
    # other
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot training progress')
    parser.add_argument('--plot_env', default='main', type=str,
                        help='plot env name')
    parser.add_argument('--save', default='', type=str,
                        help='save the model after training')
    parser.add_argument('--save_every', default=0, type=int,
                        help='save the model after every n_th epoch')
    parser.add_argument('--load', default='', type=str,
                        help='load the model')
    parser.add_argument('--display', action="store_true", default=False,
                        help='Display environment state')


    parser.add_argument('--random', action='store_true', default=False,
                        help="enable random model")

    # CommNet specific args
    parser.add_argument('--commnet', action='store_true', default=False,
                        help="enable commnet model")
    parser.add_argument('--ic3net', action='store_true', default=False,
                        help="enable commnet model")
    parser.add_argument('--nagents', type=int, default=1,
                        help="Number of agents (used in multiagent)")
    parser.add_argument('--comm_mode', type=str, default='avg',
                        help="Type of mode for communication tensor calculation [avg|sum]")
    parser.add_argument('--comm_passes', type=int, default=1,
                        help="Number of comm passes per step over the model")
    parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                        help="Whether communication should be there")
    parser.add_argument('--mean_ratio', default=1.0, type=float,
                        help='how much coooperative to do? 1.0 means fully cooperative')
    parser.add_argument('--rnn_type', default='MLP', type=str,
                        help='type of rnn to use. [LSTM|MLP]')
    parser.add_argument('--detach_gap', default=10000, type=int,
                        help='detach hidden state and cell state for rnns at this interval.'
                        + ' Default 10000 (very high)')
    parser.add_argument('--comm_init', default='uniform', type=str,
                        help='how to initialise comm weights [uniform|zeros]')
    parser.add_argument('--hard_attn', default=False, action='store_true',
                        help='Whether to use hard attention: action - talk|silent')
    parser.add_argument('--comm_action_one', default=False, action='store_true',
                        help='Whether to always talk, sanity check for hard attention.')
    parser.add_argument('--advantages_per_action', default=False, action='store_true',
                        help='Whether to multipy log porb for each chosen action with advantages')
    parser.add_argument('--share_weights', default=False, action='store_true',
                        help='Share weights for hops')

    # Development
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Ablation phase: 1=Vanilla, 2=Decoupled Binary, 3=Matrix Routing, 4=Representation Split, 5=Standalone Policies")
    parser.add_argument('--legacy_gate_resample', action='store_true', default=False,
                        help='Reproduce the pre-fix gate bug in phases 2-5: re-sample the gate heads after '
                             'the model has used its own gate sample, and keep self gates in the loss. '
                             'Only for reproducing legacy results.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='If the --save checkpoint already exists, load it and train only the '
                             'remaining epochs up to --num_epochs.')

    return parser


def prepare_args(args):
    """Post-process parsed arguments before the environment is created."""
    if args.ic3net:
        args.commnet = 1
        args.hard_attn = 1
        args.mean_ratio = 0

        # For TJ set comm action to 1 as specified in paper to showcase
        # importance of individual rewards even in cooperative games
        if args.env_name == "traffic_junction":
            args.comm_action_one = True
    # Enemy comm
    args.nfriendly = args.nagents
    if hasattr(args, 'enemy_comm') and args.enemy_comm:
        if hasattr(args, 'nenemies'):
            args.nagents += args.nenemies
        else:
            raise RuntimeError("Env. needs to pass argument 'nenemy'.")
    return args


def finalize_args(args, env):
    """Post-process arguments that depend on the environment (action heads, recurrence)."""
    num_inputs = env.observation_dim
    args.num_actions = env.num_actions

    # Multi-action
    if not isinstance(args.num_actions, (list, tuple)): # single action case
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.num_inputs = num_inputs

    # Hard attention
    if args.hard_attn and args.commnet:
        if args.phase in [3, 4, 5]:
            # Both Phase 3 and Phase 4 utilize the N x N targeted matrix routing architecture
            args.num_actions = [*args.num_actions, *([2] * args.nagents)]
            args.dim_actions = env.dim_actions + args.nagents
        else:
            # Original (and Phase 2) implementation: Each agent has a single binary head for communication
            args.num_actions = [*args.num_actions, 2]
            args.dim_actions = env.dim_actions + 1

    # Recurrence
    if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
        args.recurrent = True
        args.rnn_type = 'LSTM'


    parse_action_args(args)
    return args


def build_policy(args):
    """Instantiate the policy network selected by the arguments."""
    num_inputs = args.num_inputs
    if args.commnet:
        policy_net = CommNetMLP(args, num_inputs)
    elif args.random:
        policy_net = Random(args, num_inputs)
    elif args.recurrent:
        policy_net = RNN(args, num_inputs)
    else:
        policy_net = MLP(args, num_inputs)
    return policy_net
