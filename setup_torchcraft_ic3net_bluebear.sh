#!/bin/bash
# =============================================================================
# TorchCraft + IC3Net Full Setup Script for BlueBEAR HPC (x86_64, EL8/IceLake)
# =============================================================================
# Run this as a SLURM job or interactively on a compute node.
# Prerequisites:
#   - A Blizzard Battle.net account with StarCraft: Brood War installed locally
#   - .mpq files (StarCraft.mpq, BroodWar.mpq, patch_rt.mpq) copied to $BASE
#     via: scp /path/to/*.mpq axa1943@bluebear.bham.ac.uk:$BASE/
#
# SLURM header (optional, save separately as job.sh and call this script):
#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbshort
#SBATCH --time=0-02:00:00
#SBATCH --constraint=icelake
#SBATCH --output=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.out
#SBATCH --error=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.err

set -e

echo "=== TorchCraft + IC3Net Setup Script ==="

# =============================================================================
# 0. Modules
# =============================================================================
module purge; module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0
module load CMake/3.24.3-GCCcore-12.2.0
module load ZeroMQ/4.3.4-GCCcore-12.2.0
module load zstd/1.5.2-GCCcore-12.2.0
module load binutils/2.39-GCCcore-12.2.0   # critical: provides newer 'as' assembler

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

# =============================================================================
# 1. Paths
# =============================================================================
BASE="/rds/projects/b/baberc-human-agent-teaming/Aju"
CONDA_ENV_PATH="$BASE/${USER}_conda_envs/ic3"
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs"

mkdir -p "$BASE"
cd "$BASE"

# =============================================================================
# 2. Conda environment
# =============================================================================
echo "--- Creating conda environment ---"
mamba create --yes --prefix "${CONDA_ENV_PATH}" python=3.6.13
mamba activate "${CONDA_ENV_PATH}"

mamba install --yes \
    pip=21.2.2 \
    setuptools=58.0.4 \
    wheel=0.37.1

# =============================================================================
# 3. Python packages
# =============================================================================
echo "--- Installing Python packages ---"
pip install "$BASE/IC3Net/torch-0.4.0-cp36-cp36m-linux_x86_64.whl"

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
    visdom==0.1.4 \
    pybind11

# =============================================================================
# 4. Clone repos
# =============================================================================
echo "--- Cloning repositories ---"
git clone https://github.com/TorchCraft/TorchCraft
cd TorchCraft
git fetch origin develop:develop
git checkout develop
git submodule update --init --recursive
cd ..

git clone https://github.com/openbw/openbw
git clone https://github.com/openbw/bwapi

# =============================================================================
# 5. Build BWAPI (must be done before TorchCraft)
# =============================================================================
echo "--- Building BWAPI ---"
cd "$BASE/bwapi"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENBW_DIR="$BASE/openbw" \
    -DCMAKE_INSTALL_PREFIX="$BASE/bwapi_install" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

make -j$(nproc)
make install
cd "$BASE"

# =============================================================================
# 6. Fix FlatBuffers stl_emulation.h (GCC 12 compatibility)
# =============================================================================
echo "--- Patching stl_emulation.h for GCC 12 ---"
sed -i '1s/^/#include <limits>\n/' \
    "$BASE/TorchCraft/BWEnv/fbs/stl_emulation.h"

# =============================================================================
# 7. Build TorchCraft BWEnv
# =============================================================================
echo "--- Building TorchCraft BWEnv ---"
cd "$BASE/TorchCraft/BWEnv"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBWAPI_DIR="$BASE/bwapi_install" \
    -DCMAKE_INSTALL_PREFIX="$BASE/TorchCraft/install"

make -j$(nproc)
cd "$BASE"

# =============================================================================
# 8. TorchCraft Python bindings
# =============================================================================
echo "--- Installing TorchCraft Python bindings ---"
cd "$BASE/TorchCraft"
pip install -e .

# =============================================================================
# 9. Copy .mpq files into TorchCraft directory
# =============================================================================
echo "--- Copying .mpq files ---"
# Make sure StarCraft.mpq, BroodWar.mpq, patch_rt.mpq are in $BASE first
cp "$BASE"/*.mpq "$BASE/TorchCraft/" 2>/dev/null || \
    echo "WARNING: No .mpq files found in $BASE — copy them manually before running!"

# =============================================================================
# 10. IC3Net environments
# =============================================================================
echo "--- Installing IC3Net environments ---"
cd "$BASE/IC3Net/ic3net-envs"
python setup.py develop

# =============================================================================
# 11. Verify
# =============================================================================
echo "--- Verifying installs ---"
python -c "import torchcraft; print('TorchCraft: OK')"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import gym; print('Gym: OK')"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run IC3Net on StarCraft, use two terminals / SLURM steps:"
echo "  Terminal 1 (server): $BASE/TorchCraft/BWEnv/build/BWEnvClient <config>"
echo "  Terminal 2 (client): python examples/attack_closest.py --server_ip 127.0.0.1 \\"
echo "      --torchcraft_dir=$BASE/TorchCraft ..."
