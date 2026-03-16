#!/bin/bash

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbshort

#SBATCH --time 0-00:5:00  # days-hours:minutes:seconds
#SBATCH --nodes 1
#SBATCH --ntasks 16

set -e

module purge; module load bluebear

module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)" 
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/${USER}_ic3_conda_env" 
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs" 

# Activate the environment
mamba activate "${CONDA_ENV_PATH}"

# Run commands within the activate environment
python -c "print('hello world')"