#!/bin/bash
# Run tests/run_stack_checks.sh on BEAR as one short job.
#
#   cd /rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net
#   sbatch tests/bear_checks.sh            # all checks
#   sbatch tests/bear_checks.sh parity     # or one of: gates | smoke | layout | determinism | parity
#
# Output goes to slurm_logs/checks_<jobid>.out

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logs/checks_%j.out
#SBATCH --error=slurm_logs/checks_%j.out

module purge; module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/ic3"
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs"
mamba activate "${CONDA_ENV_PATH}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "git: $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "python: $(which python)"

TMPDIR="/scratch/${USER}" bash tests/run_stack_checks.sh "${1:-all}"
