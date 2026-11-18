#!/usr/bin/env python
"""Generate the run tables for the corrected (post-fix) experiments.

Each output line is   <run label> TAB <arguments for main.py> TAB <estimated hours>
and is read by launch/run_array.sh (one SLURM array task per line). Labels look like
    tj-hard__matrix_separate__s03
and become the checkpoint file name, so every run has a fixed, unique name.

Protocol (agreed 2026-11-18):
  * 7 architectures; Matrix-Separate runs as phase 5 only (identical to phase 4, see
    tests/test_phase_equivalence.py, and it also saves the comm/action modules).
  * The same epoch budget for every architecture within a scenario.
  * Explicit seeds. Seeds 1-10 first; seeds 11-20 for TJ/PP when the cluster is idle.
  * SC explore 75/75 goes in its own table, submitted last.

Lines are sorted longest-first (by measured legacy epoch times), so the slowest runs
start early and the whole array finishes sooner.

Usage:  python configs/make_configs.py          (writes the .tsv files next to this script)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ARCHS = [
    # label            flags                   key into SEC_PER_EPOCH
    ('ic', '', 'ic'),
    ('iric', '--mean_ratio 0', 'iric'),
    ('commnet', '--commnet', 'commnet'),
    ('ic3net', '--ic3net --phase 1', 'phase1'),
    ('gated_shared', '--ic3net --phase 2', 'phase2'),
    ('matrix_shared', '--ic3net --phase 3', 'phase3'),
    ('matrix_separate', '--ic3net --phase 5', 'phase5'),
]

COMMON = '--nprocesses 16 --hid_size 128 --detach_gap 10 --recurrent'

PP = '--env_name predator_prey --lrate 0.001'
TJ = ('--env_name traffic_junction --lrate 0.001 --vision 0 '
      '--curr_start 250 --curr_end 1250')
SC = ('--env_name starcraft --nagents 10 --lrate 0.002 --max_steps 60 --frame_skip 8 '
      '--rnn_type LSTM --explore_vision 10 --step_size 16')

# name: (env base, scenario flags, epochs)
SCENARIOS = {
    'pp-easy': (PP, '--nagents 3 --dim 5 --max_steps 20 --vision 0', 1500),
    'pp-medium': (PP, '--nagents 5 --dim 10 --max_steps 40 --vision 1', 1500),
    'pp-hard': (PP, '--nagents 10 --dim 20 --max_steps 80 --vision 1', 2500),
    'tj-easy': (TJ, '--nagents 5 --dim 6 --max_steps 20 --add_rate_min 0.1 --add_rate_max 0.3 '
                    '--difficulty easy', 1500),
    'tj-medium': (TJ, '--nagents 10 --dim 14 --max_steps 40 --add_rate_min 0.02 --add_rate_max 0.05 '
                      '--difficulty medium', 1500),
    'tj-hard': (TJ, '--nagents 20 --dim 18 --max_steps 80 --add_rate_min 0.02 --add_rate_max 0.05 '
                    '--difficulty hard', 2500),
    'sc-combat': (SC, '--task_type combat --nenemies 3 --our_unit_type 0 --enemy_unit_type 65 '
                      '--init_range_end 150', 1000),
    'sc-explore50': (SC, '--task_type explore --nenemies 1 --our_unit_type 34 --enemy_unit_type 34 '
                         '--init_range_end 150 --stay_near_enemy', 1000),
    'sc-explore75': (SC, '--task_type explore --nenemies 1 --our_unit_type 34 --enemy_unit_type 34 '
                         '--init_range_end 175 --stay_near_enemy', 1000),
}

# Median seconds per epoch measured from the legacy BEAR logs (epoch_times.txt,
# 2026-11-18). TJ medium/hard were measured on the easy road layout, so the real
# layouts may be somewhat slower.
SEC_PER_EPOCH = {
    'pp-easy': dict(ic=2, iric=4, commnet=8, phase1=10, phase2=14, phase3=16, phase5=14),
    'pp-medium': dict(ic=6, iric=6, commnet=12, phase1=13, phase2=26, phase3=28, phase5=30),
    'pp-hard': dict(ic=33, iric=32, commnet=22, phase1=52, phase2=78, phase3=87, phase5=97),
    'tj-easy': dict(ic=4, iric=5, commnet=10, phase1=13, phase2=18, phase3=21, phase5=24),
    'tj-medium': dict(ic=7, iric=6, commnet=13, phase1=17, phase2=26, phase3=29, phase5=27),
    'tj-hard': dict(ic=9, iric=12, commnet=28, phase1=32, phase2=43, phase3=52, phase5=50),
    'sc-combat': dict(ic=55, iric=56, commnet=58, phase1=65, phase2=69, phase3=72, phase5=70),
    'sc-explore50': dict(ic=60, iric=60, commnet=66, phase1=64, phase2=72, phase3=74, phase5=74),
    'sc-explore75': dict(ic=64, iric=66, commnet=70, phase1=68, phase2=78, phase3=81, phase5=82),
}

TABLES = [
    # file name, scenarios, seeds
    ('corrected_sc.tsv', ['sc-combat', 'sc-explore50'], range(1, 11)),
    ('corrected_tjpp.tsv', ['pp-easy', 'pp-medium', 'pp-hard', 'tj-easy', 'tj-medium', 'tj-hard'], range(1, 11)),
    ('corrected_sc_explore75.tsv', ['sc-explore75'], range(1, 11)),
    ('corrected_tjpp_seeds11-20.tsv', ['pp-easy', 'pp-medium', 'pp-hard', 'tj-easy', 'tj-medium', 'tj-hard'],
     range(11, 21)),
]


def build(scenarios, seeds):
    rows = []
    for seed in seeds:
        for scen in scenarios:
            base, flags, epochs = SCENARIOS[scen]
            for arch, arch_flags, key in ARCHS:
                label = '{}__{}__s{:02d}'.format(scen, arch, seed)
                args = ' '.join(x for x in [base, flags, COMMON, arch_flags,
                                            '--num_epochs {}'.format(epochs), '--seed {}'.format(seed)] if x)
                hours = SEC_PER_EPOCH[scen][key] * epochs / 3600.0
                rows.append((label, args, hours, seed))
    # Longest first; ties broken by seed so early tasks cover all architectures.
    rows.sort(key=lambda r: (-round(r[2], 1), r[3], r[0]))
    return rows


def makespan(hours, slots):
    """Greedy list-scheduling estimate of wall-clock time for `slots` concurrent jobs."""
    free = [0.0] * slots
    for h in hours:  # already longest-first
        i = free.index(min(free))
        free[i] += h
    return max(free)


def main():
    for fname, scenarios, seeds in TABLES:
        rows = build(scenarios, seeds)
        path = os.path.join(HERE, fname)
        with open(path, 'w') as f:
            f.write('# label\targs\test_hours   (generated by configs/make_configs.py; do not edit by hand)\n')
            for label, args, hours, _ in rows:
                f.write('{}\t{}\t{:.1f}\n'.format(label, args, hours))
        hours = [r[2] for r in rows]
        print('{:34s} {:4d} runs  {:7.0f} job-hours ({:5.1f} job-days)  longest {:5.1f} h  '
              'wall-clock ~{:5.1f} / {:5.1f} / {:5.1f} days at 20 / 40 / 84 jobs'.format(
                  fname, len(rows), sum(hours), sum(hours) / 24, max(hours),
                  makespan(hours, 20) / 24, makespan(hours, 40) / 24, makespan(hours, 84) / 24))


if __name__ == '__main__':
    main()
