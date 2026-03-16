#!/bin/bash

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault

#SBATCH --time 0-00:59:59  # days-hours:minutes:seconds
#SBATCH --nodes 1
#SBATCH --ntasks 16

#SBATCH --output=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.out
#SBATCH --error=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.err

set -e

module purge; module load bluebear

module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)" 
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/ic3" 
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs" 

# Activate the environment
mamba activate "${CONDA_ENV_PATH}"

# Test within the environment
python -c "print('hello world')"

# Define save directory
SAVE_DIR="/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/trained_models"
mkdir -p "${SAVE_DIR}"

# Optional: timestamped run name
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="${SAVE_DIR}/${RUN_NAME}"

echo "Saving to: ${SAVE_PATH}"

# Run training
python main.py \
  --env_name traffic_junction \
  --nagents 5 \
  --nprocesses 16 \
  --num_epochs 2000 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --dim 6 \
  --max_steps 20 \
  --ic3net \
  --vision 0 \
  --recurrent \
  --add_rate_min 0.1 \
  --add_rate_max 0.3 \
  --curr_start 250 \
  --curr_end 1250 \
  --difficulty easy \
  --save "${SAVE_PATH}" \
  --save_every 250