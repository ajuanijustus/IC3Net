#!/bin/bash

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbshort
#SBATCH --time 0-00:5:00  # days-hours:minutes:seconds

set -e

module purge; module load bluebear

module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)" 
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/${USER}_ic3_conda_env" 
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs" 

# Create environment with Python first
mamba create --yes --prefix "${CONDA_ENV_PATH}" python=3.6.13

# Activate environment
mamba activate "${CONDA_ENV_PATH}"

# Install conda packages
mamba install --yes pip setuptools wheel

mamba install --yes \
  pip=21.2.2 \
  setuptools=58.0.4 \
  wheel=0.37.1 \

# Install pip packages
pip install /rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/torch-0.4.0-cp36-cp36m-linux_x86_64.whl

pip install \
  charset-normalizer==2.0.12 \
  gym==0.9.6 \
  idna==3.10 \
  numpy==1.13.3 \
  pillow==6.2.0 \
  pyglet==2.0.10 \
  pyzmq==25.1.2 \
  requests==2.27.1 \
  six==1.17.0 \
  tornado==6.1 \
  urllib3==1.26.20 \
  visdom==0.1.4

# Install IC3Net environments
git clone https://github.com/IC3Net/IC3Net
cd IC3Net/ic3net-envs
python setup.py develop