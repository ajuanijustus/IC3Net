#!/bin/bash

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault

#SBATCH --time=2-00:00:00  # days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1

#SBATCH --output=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.out
#SBATCH --error=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.err

set -e

module purge; module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)" 
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/ic3" 
mamba activate "${CONDA_ENV_PATH}"

TORCRAFT_DIR="/rds/projects/b/baberc-human-agent-teaming/Aju/TorchCraft"

export CXXFLAGS="-I${CONDA_PREFIX}/include $CXXFLAGS"
export CFLAGS="-I${CONDA_PREFIX}/include $CFLAGS"
export LDFLAGS="-L${CONDA_PREFIX}/lib $LDFLAGS"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" # macOS DYLD_ removed
export OPENBW_MPQ_PATH="/rds/projects/b/baberc-human-agent-teaming/Aju/sc_mpq"

# Define save directory
SAVE_DIR="/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/sc_trained_models"
mkdir -p "${SAVE_DIR}"

# Optional: timestamped run name
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="${SAVE_DIR}/${RUN_NAME}"

echo "Saving to: ${SAVE_PATH}"

python -u main.py \
  --env_name starcraft \
  --task_type explore \
  --nagents 10 \
  --num_epochs 1000 \
  --hid_size 128 \
  --lrate 0.002 \
  --max_steps 60 \
  --nprocesses 16 \
  --torchcraft_dir="$TORCRAFT_DIR" \
  --frame_skip 8\
   --nenemies 1 \
  --our_unit_type 34 \
  --enemy_unit_type 34 \
  --init_range_end 150 \
  --ic3net \
  --recurrent \
  --rnn_type LSTM \
  --detach_gap 10 \
  --stay_near_enemy \
  --explore_vision 10 \
  --step_size 16 \
  --save "${SAVE_PATH}" \
  --save_every 250