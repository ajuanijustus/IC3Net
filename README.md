# Modular communication and action policies (IC3Net-based)

Research code for multi-agent reinforcement learning with communication. It is built on the
[IC3Net](https://github.com/IC3Net/IC3Net) reference implementation, and adds architectures in which
the communication policy and the action policy are separate pathways: scalar gating, pairwise
(matrix) routing, and fully separate, independently saved modules.

It runs on Predator-Prey, Traffic Junction and StarCraft: Brood War (TorchCraft) on a fixed legacy
stack: Python 3.6, torch 0.4, gym 0.9.6. This repository is kept as the reproducibility artifact for
that stack. A successor codebase on a modern stack is in development.

## Layout

| Path | What |
|---|---|
| `main.py` | Training entry point (logging, checkpoints, `--resume`) |
| `arguments.py` | Command-line arguments and their post-processing; builds the policy |
| `comm.py` | All architectures; `--phase` selects the dual-pathway variants |
| `models.py` | Non-communicating baselines (IC/IRIC: MLP or recurrent) |
| `trainer.py`, `multi_processing.py` | REINFORCE trainer with a learned baseline; parallel rollout workers |
| `action_utils.py`, `env_wrappers.py`, `data.py`, `utils.py` | Action sampling, env wrapper, env factory, logging helpers |
| `configs/` | Run tables and their generator (`make_configs.py`) |
| `launch/run_array.sh` | SLURM array launcher (one table line per task; skip/resume) |
| `scripts/` | Results export and bridge fixtures |
| `tests/` | Unit tests and the full stack check (`run_stack_checks.sh`, `bear_checks.sh`) |
| `setup/` | BlueBEAR environment and TorchCraft build scripts |
| `docs/` | Setup (BlueBEAR, macOS) and running experiments |
| `ic3net-envs/`, `gym-starcraft/` | Environment packages (Predator-Prey, Traffic Junction; StarCraft) |

## Architectures

| Label | Routing | Pathways |
|---|---|---|
| IC / IRIC | none | single (team / individual reward) |
| CommNet | broadcast mean | single |
| IC3Net | scalar gate | single |
| Gated-Shared | scalar gate | dual, shared encoder |
| Matrix-Shared | pairwise mutual-consent gates | dual, shared encoder |
| Matrix-Separate | pairwise mutual-consent gates | dual, separate modules (saved separately) |

Command-line flags for each are listed in [docs/experiments.md](docs/experiments.md#architectures).

## Quick start

Setup: [BlueBEAR](docs/setup_bear.md) or [macOS](docs/setup_macos.md). Then:

```bash
python main.py --env_name predator_prey --nagents 3 --dim 5 --max_steps 20 --vision 0 \
  --nprocesses 4 --hid_size 128 --detach_gap 10 --recurrent --lrate 0.001 \
  --ic3net --phase 5 --num_epochs 100 --seed 1 --save trained_models/pp_easy_demo
```

Full experiment grids are run from the tables in `configs/` with `launch/run_array.sh`. See
[docs/experiments.md](docs/experiments.md).

## Tests

```bash
python tests/test_gates.py               # gates used for messages are the gates trained on
python tests/test_phase_equivalence.py   # phase 4 and phase 5 compute the same function
bash tests/run_stack_checks.sh           # smoke, TJ layouts, determinism, resume, legacy parity
```

On BlueBEAR: `sbatch tests/bear_checks.sh`.

## Reproducibility notes

Compared with the earlier state of this code (tag `legacy-aamas2027`), the current code fixes:
- **Gate training (dual-pathway models):** the gate actions used for the REINFORCE update are now the
  ones that masked the messages. Previously the trainer re-sampled them, so the gates got no learning
  signal. Each agent's always-masked gate to itself is no longer in the loss.
- **Traffic Junction difficulty:** `--difficulty` now reaches the environment, so medium and hard use
  their real road layouts.
- **Seeding:** `random`, numpy and torch are seeded in every process, with independent per-worker
  seeds. The StarCraft envs place units with Python's `random`, which was previously unseeded.

`--legacy_gate_resample` reproduces the old gate behaviour exactly, for comparison. Checkpoints record
their arguments, seeds and git SHA. The corrected experiment runs use tag `corrected-v1`.

## Credits

This code builds on IC3Net by Amanpreet Singh, Tushar Jain and Sainbayar Sukhbaatar (MIT licence). The
environment packages come from [apsdehal/ic3net-envs](https://github.com/apsdehal/ic3net-envs) and
[apsdehal/gym-starcraft](https://github.com/apsdehal/gym-starcraft). If you use this code, please cite
IC3Net:

```
@inproceedings{singh2019learning,
  title={Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks},
  author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

## Licence

MIT. See [LICENSE](LICENSE).
