"""Gate consistency tests for the dual-pathway models (phases 2-5).

The REINFORCE update for the gates is only correct if the gate actions recorded for
training are the same gates that masked the messages during the forward pass.

For each phase this test:
  1. runs one forward pass and selects actions the way the trainer does,
  2. rebuilds the message each agent should have received from the *recorded* gates,
  3. compares it with the input the action pathway actually received (via a hook).

With the fix, they match on every trial. With --legacy_gate_resample (the pre-fix
behaviour) they must disagree on some trials, which demonstrates the old bug.

Run:  python -m pytest tests/test_gates.py      or      python tests/test_gates.py
"""
import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from comm_2 import CommNetMLP  # noqa: E402
from action_utils import select_action  # noqa: E402

torch.set_default_tensor_type('torch.DoubleTensor')

N_AGENTS = 4
HID = 16
N_INPUTS = 7
N_ENV_ACTIONS = 5
N_TRIALS = 30


def make_args(phase, legacy):
    n = N_AGENTS
    if phase == 2:
        num_actions = [N_ENV_ACTIONS, 2]
    else:
        num_actions = [N_ENV_ACTIONS] + [2] * n
    return Namespace(
        nagents=n, nfriendly=n, hid_size=HID, comm_passes=1, recurrent=True, rnn_type='LSTM',
        continuous=False, phase=phase, comm_mask_zero=False, comm_init='uniform', batch_size=1,
        num_actions=num_actions, naction_heads=num_actions, dim_actions=len(num_actions),
        hard_attn=True, commnet=True, comm_mode='avg', legacy_gate_resample=legacy,
    )


def parts(model, phase):
    """Return (C module, action-pathway encoder, action-pathway LSTM) for a phase."""
    if phase == 5:
        return model.comm_policy.C_modules[0], model.env_policy.env_encoder, model.env_policy.env_f_module
    enc = model.env_encoder if phase == 4 else model.encoder
    return model.C_modules[0], enc, model.env_f_module


def recorded_gate_matrix(action, phase, n):
    """gate[i, j] = 1 if agent i opens its link to agent j, from the recorded actions."""
    heads = [a.view(-1) for a in action]  # each (n,)
    if phase == 2:
        g = heads[-1].double()
        return g.view(n, 1).expand(n, n)
    return torch.stack(heads[-n:], dim=1).double()  # column r = head r


def expected_action_input(model, phase, x, prev_h, gate):
    n = N_AGENTS
    c_mod, enc, _ = parts(model, phase)
    comm_h = prev_h.chunk(2, dim=-1)[0]  # (n, HID)
    offdiag = torch.ones(n, n) - torch.eye(n)
    links = gate * gate.t() * offdiag  # mutual consent, no self links
    msg = links.t().mm(comm_h) / (n - 1)  # msg_j = sum_i links[i, j] h_i / (n - 1)
    return (enc(x.view(n, -1)) + c_mod(msg)).detach()


def run_trials(phase, legacy):
    torch.manual_seed(1234 + phase)
    args = make_args(phase, legacy)
    model = CommNetMLP(args, N_INPUTS)
    _, _, lstm = parts(model, phase)
    captured = []
    lstm.register_forward_hook(lambda m, inp, out: captured.append(inp[0].detach().clone()))

    mismatches = 0
    for _ in range(N_TRIALS):
        x = torch.randn(1, N_AGENTS, N_INPUTS)
        prev_h = torch.randn(N_AGENTS, 2 * HID)
        prev_c = torch.randn(N_AGENTS, 2 * HID)
        captured.clear()
        action_out, _, _ = model([x, (prev_h, prev_c)], {})
        presampled = None if legacy else model.last_comm_action
        action = select_action(args, action_out, presampled)

        gate = recorded_gate_matrix(action, phase, N_AGENTS)
        expected = expected_action_input(model, phase, x, prev_h, gate)
        if (captured[-1] - expected).abs().max().item() > 1e-10:
            mismatches += 1
    return mismatches


def test_recorded_gates_match_executed_gates():
    for phase in [2, 3, 4, 5]:
        mismatches = run_trials(phase, legacy=False)
        assert mismatches == 0, 'phase {}: {} / {} trials used gates that were not recorded'.format(
            phase, mismatches, N_TRIALS)


def test_legacy_flag_reproduces_resampling_bug():
    for phase in [2, 3, 4, 5]:
        mismatches = run_trials(phase, legacy=True)
        assert mismatches > 0, 'phase {}: legacy mode unexpectedly matched on every trial'.format(phase)


def test_presampled_heads_keep_shape_and_values():
    args = make_args(3, legacy=False)
    torch.manual_seed(0)
    model = CommNetMLP(args, N_INPUTS)
    x = torch.randn(1, N_AGENTS, N_INPUTS)
    h = model.init_hidden(1)
    action_out, _, _ = model([x, h], {})
    resampled = select_action(args, action_out)
    given = select_action(args, action_out, model.last_comm_action)
    assert given.shape == resampled.shape
    for r, g in enumerate(model.last_comm_action):
        assert torch.equal(given[1 + r].view(-1), g.view(-1))


if __name__ == '__main__':
    for phase in [2, 3, 4, 5]:
        fixed = run_trials(phase, legacy=False)
        legacy = run_trials(phase, legacy=True)
        print('phase {}: mismatching trials  fixed {:2d}/{}   legacy {:2d}/{}'.format(
            phase, fixed, N_TRIALS, legacy, N_TRIALS))
    test_recorded_gates_match_executed_gates()
    test_legacy_flag_reproduces_resampling_bug()
    test_presampled_heads_keep_shape_and_values()
    print('All gate tests passed.')
