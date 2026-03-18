#!/bin/bash

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=torchcraft_setup

#SBATCH --output=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.out
#SBATCH --error=/rds/projects/b/baberc-human-agent-teaming/Aju/IC3Net/slurm_logs/%x_%j.err

set -e

echo "=== TorchCraft setup started ==="

# -----------------------------
# Modules
# -----------------------------
module purge; module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0
module load CMake
module load GCC

# -----------------------------
# Directories (use RDS, not HOME)
# -----------------------------
BASE_DIR="/rds/projects/b/baberc-human-agent-teaming/Aju"
BUILD_DIR="${BASE_DIR}/torchcraft_build"
INSTALL_DIR="${BASE_DIR}/apps"

mkdir -p "${BUILD_DIR}"
mkdir -p "${INSTALL_DIR}"

cd "${BUILD_DIR}"

# -----------------------------
# libsodium
# -----------------------------
echo "Installing libsodium..."
wget -nc https://download.libsodium.org/libsodium/releases/old/unsupported/libsodium-1.0.14.tar.gz
tar xf libsodium-1.0.14.tar.gz
cd libsodium-1.0.14

./configure --prefix="${INSTALL_DIR}"
make -j${SLURM_CPUS_PER_TASK}
make install

cd ..

# -----------------------------
# ZeroMQ
# -----------------------------
echo "Installing ZeroMQ..."
wget -nc https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz
tar xf zeromq-4.1.4.tar.gz
cd zeromq-4.1.4

PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
./configure --prefix="${INSTALL_DIR}"

make -j${SLURM_CPUS_PER_TASK}
make install

cd ..

# -----------------------------
# Environment variables
# -----------------------------
export CFLAGS="-I${INSTALL_DIR}/include"
export LDFLAGS="-L${INSTALL_DIR}/lib"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"

# -----------------------------
# zstd
# -----------------------------
echo "Installing zstd..."
git clone https://github.com/facebook/zstd || true
cd zstd
make -j${SLURM_CPUS_PER_TASK}
make PREFIX=${INSTALL_DIR} install
cd ..

# -----------------------------
# OpenBW + BWAPI
# -----------------------------
echo "Cloning OpenBW + BWAPI..."
git clone https://github.com/openbw/openbw || true
git clone https://github.com/openbw/bwapi || true

cd bwapi
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENBW_DIR=../../openbw \
  -DCMAKE_INSTALL_PREFIX=${BASE_DIR}/bwapi

make -j${SLURM_CPUS_PER_TASK}
make install

cd ../..

# -----------------------------
# TorchCraft
# -----------------------------
echo "Installing TorchCraft..."
git clone https://github.com/TorchCraft/TorchCraft || true
cd TorchCraft
git fetch origin develop:develop || true
git submodule update --init --recursive

cd BWEnv
mkdir -p build && cd build

CC=gcc CXX=g++ \
CXXFLAGS="-I${INSTALL_DIR}/include" \
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBWAPI_DIR=../../bwapi/

make -j${SLURM_CPUS_PER_TASK}

cd ../..

# -----------------------------
# Python install (into your env)
# -----------------------------
CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/ic3"

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
mamba activate "${CONDA_ENV_PATH}"

pip install pybind11

LDFLAGS="-L${INSTALL_DIR}/lib" \
CFLAGS="-I${INSTALL_DIR}/include" \
pip install -e .

# -----------------------------
# Persist environment
# -----------------------------
echo "Saving environment variables..."

cat <<EOF >> ${BASE_DIR}/torchcraft_env.sh
export CFLAGS="-I${INSTALL_DIR}/include"
export LDFLAGS="-L${INSTALL_DIR}/lib"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
EOF

echo "=== TorchCraft setup complete ==="
