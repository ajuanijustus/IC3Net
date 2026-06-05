#!/bin/bash

#SBATCH --array=0-23

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault

#SBATCH --time=0-10:00:00  # days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1

#SBATCH --output=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%A_%a.out
#SBATCH --error=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%A_%a.err

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
# python -c "print('hello world')"

# Define save directory
SAVE_DIR="/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/trained_models_3"
mkdir -p "${SAVE_DIR}"

# Read raw config line
RAW_CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" bear_config.txt)

# 1. Extract metadata for the RUN_NAME
ENV=$(echo "$RAW_CONFIG" | grep -oP '(?<=--env_name )\S+')
MODEL=$(echo "$RAW_CONFIG" | grep -oP '(--ic3net|--commnet|--mean_ratio 0)' | head -n1)
DIFF=$(echo "$RAW_CONFIG" | grep -oP '(?<=--difficulty )\S+' || echo "na")

# Clean model name string
if [[ "$MODEL" == "--ic3net" ]]; then MODEL="ic3net"; fi
if [[ "$MODEL" == "--commnet" ]]; then MODEL="commnet"; fi
if [[ "$MODEL" == "--mean_ratio 0" ]]; then MODEL="iric"; fi
if [[ -z "$MODEL" ]]; then MODEL="ic"; fi

# 2. Clean the config string sent to Python (Strip out the difficulty flag if present)
CLEANED_CONFIG=$(echo "$RAW_CONFIG" | sed -E 's/--difficulty [^ ]+//g')

# 3. Create a clean base run name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${ENV}_${MODEL}_${DIFF}_job${SLURM_ARRAY_TASK_ID}_${TIMESTAMP}"
SAVE_PATH="${SAVE_DIR}/${RUN_NAME}"

echo "Run name: ${RUN_NAME}"
echo "Executing with Cleaned Config: ${CLEANED_CONFIG}"

# Custom logging setup
LOG_DIR="/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/stdout.log"
ERR_FILE="${LOG_DIR}/stderr.log"

exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$ERR_FILE" >&2)

# 4. Run training using the safe configuration parameters
python main.py ${CLEANED_CONFIG} \
  --save "${SAVE_PATH}" \
  --save_every 500

# 5. Hotfix for checkpoint tracking:
# Since main.py saves the final output without an epoch suffix, 
# create a symlink or copy it so our plotting regex detects it seamlessly as the 1500 landmark.
if [ -f "${SAVE_PATH}" ]; then
    cp "${SAVE_PATH}" "${SAVE_PATH}_1500"
    echo "Created explicit _1500 target copy for replication plotting."
fi