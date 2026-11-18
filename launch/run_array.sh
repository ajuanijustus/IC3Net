#!/bin/bash
# Run one line of a run table (configs/*.tsv) per SLURM array task.
#
#   sbatch --array=1-140 --time=3-00:00:00 launch/run_array.sh configs/corrected_sc.tsv   <out_dir>
#   sbatch --array=1-420 --time=4-00:00:00 launch/run_array.sh configs/corrected_tjpp.tsv <out_dir>
#
# Submit from the repo root (paths are relative to it). Array task k runs line k of the
# table, not counting '#' comment lines. Each run is saved as <out_dir>/<label>, with
# 500-epoch snapshots <out_dir>/<label>_500 etc. and its log in <out_dir>/logs/<label>.log.
#
# Safe to resubmit the same array: finished runs (marked <out_dir>/<label>.done) are
# skipped, and unfinished ones resume from their last checkpoint (--resume).
#
# Override paths with IC3_CONDA_ENV, TORCHCRAFT_DIR, OPENBW_MPQ_PATH if needed.
# DRY_RUN=1 prints the command instead of running it (no SLURM or conda needed).

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -o pipefail

CONFIG=${1:?usage: run_array.sh <config.tsv> <out_dir>}
OUT_DIR=${2:?usage: run_array.sh <config.tsv> <out_dir>}
TASK=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set (submit with sbatch --array)}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

LINE=$(grep -v '^#' "$CONFIG" | sed -n "${TASK}p")
if [[ -z "$LINE" ]]; then
  echo "No line $TASK in $CONFIG"; exit 1
fi
LABEL=$(printf '%s' "$LINE" | cut -f1)
ARGS=$(printf '%s' "$LINE" | cut -f2)
ENV_NAME=$(printf '%s' "$ARGS" | grep -oE -- '--env_name [^ ]+' | cut -d' ' -f2)

SAVE_PATH="${OUT_DIR}/${LABEL}"
LOG_FILE="${OUT_DIR}/logs/${LABEL}.log"

if [[ -f "${SAVE_PATH}.done" ]]; then
  echo "${LABEL}: already finished (${SAVE_PATH}.done exists), skipping."
  exit 0
fi

EXTRA=""
if [[ "$ENV_NAME" == "starcraft" ]]; then
  TORCHCRAFT_DIR=${TORCHCRAFT_DIR:-/rds/projects/b/baberc-human-agent-teaming/Aju/TorchCraft}
  EXTRA="--torchcraft_dir=${TORCHCRAFT_DIR}"
fi

CMD="python main.py ${ARGS} --save ${SAVE_PATH} --save_every 500 --resume ${EXTRA}"

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "label:  ${LABEL}"
  echo "env:    ${ENV_NAME}"
  echo "log:    ${LOG_FILE}"
  echo "cmd:    ${CMD}"
  exit 0
fi

# ---- environment
module purge; module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0
eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs"
mamba activate "${IC3_CONDA_ENV:-/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/ic3}"

if [[ "$ENV_NAME" == "starcraft" ]]; then
  export CXXFLAGS="-I${CONDA_PREFIX}/include ${CXXFLAGS:-}"
  export CFLAGS="-I${CONDA_PREFIX}/include ${CFLAGS:-}"
  export LDFLAGS="-L${CONDA_PREFIX}/lib ${LDFLAGS:-}"
  export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export OPENBW_MPQ_PATH=${OPENBW_MPQ_PATH:-/rds/projects/b/baberc-human-agent-teaming/Aju/sc_mpq}
fi

# ---- run
mkdir -p "${OUT_DIR}/logs"
export RUN_LABEL="${LABEL}"   # recorded in the checkpoint metadata

{
  echo "==== ${LABEL}  job ${SLURM_JOB_ID:-?} task ${TASK}  $(date '+%Y-%m-%d %H:%M:%S')  host $(hostname)"
  echo "git: $(git rev-parse HEAD 2>/dev/null)$( [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo ' (dirty)')"
  echo "cmd: ${CMD}"
} | tee -a "${LOG_FILE}"

${CMD} 2>&1 | tee -a "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}

if [[ $STATUS -eq 0 ]]; then
  touch "${SAVE_PATH}.done"
  echo "==== ${LABEL} finished" | tee -a "${LOG_FILE}"
else
  echo "==== ${LABEL} exited with status ${STATUS}; resubmit the array to resume it" | tee -a "${LOG_FILE}"
fi
exit $STATUS
