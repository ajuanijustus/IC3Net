"""Phase 4 and phase 5 must compute the same function.

Phase 4 (one network with separate comm/action branches) and phase 5 (two standalone
modules, IsolatedCommPolicy + IsolatedEnvPolicy) are meant to be the same
architecture, Matrix-Separate. Only phase 5 can save the two modules separately, so
the corrected runs use phase 5 alone. This test justifies that choice: with the same
weights and the same RNG state, both give identical outputs over several steps.

Run:  python -m pytest tests/test_phase_equivalence.py   or   python tests/test_phase_equivalence.py
"""
import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from comm import CommNetMLP  # noqa: E402

torch.set_default_tensor_type('torch.DoubleTensor')

N_AGENTS = 4
HID = 16
N_INPUTS = 7


def make_args(phase):
    num_actions = [5] + [2] * N_AGENTS
    return Namespace(
        nagents=N_AGENTS, hid_size=HID, comm_passes=1, recurrent=True, rnn_type='LSTM',
        continuous=False, phase=phase, comm_mask_zero=False, comm_init='uniform', batch_size=1,
        num_actions=num_actions, naction_heads=num_actions, dim_actions=len(num_actions),
        hard_attn=True, commnet=True, comm_mode='avg', legacy_gate_resample=False,
    )


def phase4_to_phase5_key(key):
    """Map a phase-4 state_dict key to the matching phase-5 key."""
    if key.startswith('comm_') or key.startswith('C_modules'):
        return 'comm_policy.' + key
    return 'env_policy.' + key


def test_phase4_and_phase5_are_identical():
    torch.manual_seed(0)
    p4 = CommNetMLP(make_args(4), N_INPUTS)
    p5 = CommNetMLP(make_args(5), N_INPUTS)

    mapped = {phase4_to_phase5_key(k): v for k, v in p4.state_dict().items()}
    assert set(mapped) == set(p5.state_dict()), 'parameter sets differ: {}'.format(
        set(mapped) ^ set(p5.state_dict()))
    p5.load_state_dict(mapped)

    h4 = p4.init_hidden(1)
    h5 = p5.init_hidden(1)
    for step in range(5):
        x = torch.randn(1, N_AGENTS, N_INPUTS)
        alive = torch.ones(N_AGENTS).numpy()
        alive[step % N_AGENTS] = 0  # exercise the alive mask too
        info = {'alive_mask': alive}

        torch.manual_seed(100 + step)
        a4, v4, h4 = p4([x, h4], info)
        torch.manual_seed(100 + step)
        a5, v5, h5 = p5([x, h5], info)

        for o4, o5 in zip(a4, a5):
            assert (o4 - o5).abs().max().item() == 0
        for o4, o5 in zip(v4, v5):
            assert (o4 - o5).abs().max().item() == 0
        for o4, o5 in zip(h4, h5):
            assert (o4 - o5).abs().max().item() == 0
        for g4, g5 in zip(p4.last_comm_action, p5.last_comm_action):
            assert torch.equal(g4, g5)


if __name__ == '__main__':
    test_phase4_and_phase5_are_identical()
    print('Phase 4 and phase 5 give identical outputs.')
