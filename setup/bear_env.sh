#!/bin/bash
# Create the Python 3.6 / torch 0.4 conda environment on BlueBEAR and install the
# Predator-Prey and Traffic Junction environments. For StarCraft, run
# setup/bear_starcraft.sh afterwards.
#
#   cd <repo root> && sbatch setup/bear_env.sh
#
# Override the environment location with IC3_CONDA_ENV.

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbshort
#SBATCH --time=0-00:30:00
#SBATCH --job-name=ic3_env_setup
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

module purge; module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="${IC3_CONDA_ENV:-/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/ic3}"
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs"

mamba create --yes --prefix "${CONDA_ENV_PATH}" python=3.6.15
mamba activate "${CONDA_ENV_PATH}"
mamba install --yes pip setuptools wheel

pip install -r requirements.txt

# torch 0.4.0 (CPU) from the official archive, checked against the known SHA-256.
WHEEL=torch-0.4.0-cp36-cp36m-linux_x86_64.whl
WHEEL_SHA256=a561d6b74a394b8a81427a5ff6ed0e145c7cbe1ef61094f72cc73a965644b107
TMP=$(mktemp -d "/scratch/${USER}/torchwheel.XXXXXX")
curl -fsSL -o "${TMP}/${WHEEL}" "https://download.pytorch.org/whl/cpu/${WHEEL}"
echo "${WHEEL_SHA256}  ${TMP}/${WHEEL}" | sha256sum -c -
pip install "${TMP}/${WHEEL}"
rm -rf "${TMP}"

(cd ic3net-envs && python setup.py develop)

python -c "import torch, gym, ic3net_envs; print('torch', torch.__version__, '| gym', gym.__version__)"
echo "IC3Net environment set up in ${CONDA_ENV_PATH}"
