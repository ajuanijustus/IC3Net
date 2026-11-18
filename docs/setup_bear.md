# Setup on BlueBEAR

Everything runs on CPU. The stack is fixed for reproducibility: Python 3.6.15, torch 0.4.0,
gym 0.9.6, numpy 1.13.3 (TorchCraft does not build on anything newer).

## 1. Python environment (Predator-Prey, Traffic Junction)

From the repo root:

```bash
mkdir -p slurm_logs
sbatch setup/bear_env.sh
```

This creates the conda env (default `/rds/projects/b/baberc-human-agent-teaming/Aju/$USER_conda_envs/ic3`,
override with `IC3_CONDA_ENV`). It installs `requirements.txt` and torch 0.4.0 from
download.pytorch.org, checked against its SHA-256, and installs `ic3net-envs` in develop mode.

## 2. StarCraft (TorchCraft)

```bash
sbatch setup/bear_starcraft.sh
```

This builds OpenBW, BWAPI and BWEnv into `TORCHCRAFT_DIR` (default
`/rds/projects/b/baberc-human-agent-teaming/Aju/TorchCraft`). It installs the TorchCraft Python bindings and
`gym-starcraft` into the same env.

You also need:
- The StarCraft 1.16.1 MPQ files (`BrooDat.mpq`, `BroodWar.mpq`, `patch_rt.mpq`, `StarCraft.mpq`,
  `StarDat.mpq`) in one directory. Point `OPENBW_MPQ_PATH` at it.
  `launch/run_array.sh` defaults to `/rds/projects/b/baberc-human-agent-teaming/Aju/sc_mpq`.
- `gym-starcraft/gym_starcraft/envs/config.yml` pointing at your `bwapi_install` and `BWEnv.so`.

## 3. Check the install

```bash
sbatch tests/bear_checks.sh
```

This runs the gate and equivalence unit tests, smoke runs of all architectures on PP and TJ, the TJ
layout check, determinism, resume and exact parity with the legacy code. The output goes to
`slurm_logs/checks_<jobid>.out`, ending with `0 check(s) failed.`

## Tips

- BEAR login shells reset on timeout. Put paths you use often in `~/.bashrc`, e.g.
  `export OUT=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/trained_models/corrected_v1`.
- If RDS drops executable bits, `git config core.fileMode false` stops git reporting them as changes.
