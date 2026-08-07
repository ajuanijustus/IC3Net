#!/bin/bash

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=torchcraft_setup

#SBATCH --output=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.out
#SBATCH --error=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.err

set -e

# 1. Load HPC Modules and Activate Mamba
module purge; module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)" 
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/ic3" 
mamba activate "${CONDA_ENV_PATH}"

# 2. Install all dependencies into Conda (Replaces 'brew install')
mamba install -c conda-forge zstd czmq zeromq sdl2 cmake pkg-config -y

# 3. Directory Setup
# Adjusted to your RDS project directory rather than $HOME
TORCRAFT_DIR="/rds/projects/b/baberc-human-agent-teaming/Aju/TorchCraft"
mkdir -p "$TORCRAFT_DIR"
cd "$TORCRAFT_DIR"

# Clone TorchCraft
git clone https://github.com/TorchCraft/TorchCraft .
git submodule update --init --recursive

# 4. Set Compiler/Build Flags (Replaces 'brew --prefix' with '$CONDA_PREFIX')
export CXXFLAGS="-I${CONDA_PREFIX}/include $CXXFLAGS"
export CFLAGS="-I${CONDA_PREFIX}/include $CFLAGS"
export LDFLAGS="-L${CONDA_PREFIX}/lib $LDFLAGS"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" # macOS DYLD_ removed

# 5. Build BWAPI & OpenBW
cd "$TORCRAFT_DIR"
git clone https://github.com/openbw/openbw
git clone https://github.com/openbw/bwapi

cd "$TORCRAFT_DIR/bwapi"
mkdir -p build && cd build
cmake .. \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENBW_DIR=../../openbw \
  -DOPENBW_ENABLE_UI=1 \
  -DCMAKE_INSTALL_PREFIX=../../bwapi_install

# Hardcoded to 8 cores to be safe on the login node
make install -j8

# 6. Build BWEnv
cd "$TORCRAFT_DIR/BWEnv"
mkdir -p build && cd build

# rm -rf * # uncomment if you need to clear cache later

cmake .. \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=relwithdebinfo \
  -DBWAPI_DIR="$TORCRAFT_DIR/bwapi_install" \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DCMAKE_INCLUDE_PATH="${CONDA_PREFIX}/include" \
  -DCMAKE_LIBRARY_PATH="${CONDA_PREFIX}/lib"

make -j8

# 7. Install TorchCraft Python Bindings
cd "$TORCRAFT_DIR"
pip install -e . --no-cache-dir

echo "TorchCraft build completed successfully!"