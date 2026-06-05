#!/bin/bash
set -e

echo "=== TorchCraft Full Setup Script ==="
echo "Activating Conda environment 'ic3'..."
conda activate ic3

BASE_DIR="$HOME/Documents/ic3_torchcraft"
APPS_DIR="$BASE_DIR/apps"

mkdir -p "$BASE_DIR"
mkdir -p "$APPS_DIR"

cd "$BASE_DIR"

if ! command -v pkg-config &>/dev/null; then
    echo "Installing pkg-config via Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install pkg-config
fi

export CFLAGS="-I$APPS_DIR/include"
export LDFLAGS="-L$APPS_DIR/lib"
export LD_LIBRARY_PATH="$APPS_DIR/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$APPS_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"

echo "Downloading libsodium..."
curl -LO https://download.libsodium.org/libsodium/releases/old/unsupported/libsodium-1.0.14.tar.gz
tar xf libsodium-1.0.14.tar.gz
cd libsodium-1.0.14
./configure --prefix="$APPS_DIR"
make -j$(sysctl -n hw.ncpu)
make check
make install
cd ..

echo "Downloading zeromq..."
curl -LO https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz
tar xf zeromq-4.1.4.tar.gz
cd zeromq-4.1.4

PKG_CONFIG_PATH="$APPS_DIR/lib/pkgconfig" ./configure --prefix="$APPS_DIR"
make -j$(sysctl -n hw.ncpu)
make install
cd ..

echo "Cloning zstd..."
git clone https://github.com/facebook/zstd
cd zstd
make -j$(sysctl -n hw.ncpu) PREFIX="$APPS_DIR" install
cd ..

echo "Cloning OpenBW and BWAPI..."
git clone https://github.com/openbw/openbw
git clone https://github.com/openbw/bwapi

if ! brew list sdl2 &>/dev/null; then
    echo "Installing SDL2 via Homebrew..."
    if ! command -v brew &>/dev/null; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install sdl2
fi

export CFLAGS="-I$APPS_DIR/include -I$(brew --prefix sdl2)/include/SDL2"
export LDFLAGS="-L$APPS_DIR/lib -L$(brew --prefix sdl2)/lib"
export LD_LIBRARY_PATH="$APPS_DIR/lib:$(brew --prefix sdl2)/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$APPS_DIR/lib/pkgconfig:$(brew --prefix sdl2)/lib/pkgconfig:$PKG_CONFIG_PATH"
export DYLD_LIBRARY_PATH="$APPS_DIR/lib:$DYLD_LIBRARY_PATH"

echo "Installing BWAPI..."
cd bwapi
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENBW_DIR=../../openbw \
  -DOPENBW_ENABLE_UI=1 \
  -DCMAKE_INSTALL_PREFIX="$BASE_DIR/bwapi" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j$(sysctl -n hw.ncpu)
make install
cd ../..

echo "Downloading TorchCraft..."
git clone https://github.com/TorchCraft/TorchCraft
cd TorchCraft
git fetch origin develop:develop
git submodule update --init --recursive

cd BWEnv
mkdir -p build && cd build

CC=gcc CXX=g++ \
CXXFLAGS="-I$APPS_DIR/include -I$(brew --prefix sdl2)/include -I$(brew --prefix zeromq)/include" \
LDFLAGS="-L$APPS_DIR/lib -L$(brew --prefix sdl2)/lib -L$(brew --prefix zeromq)/lib" \
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBWAPI_DIR=../../bwapi/ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j

ZMQ_PREFIX=$(brew --prefix zeromq)
SDL_PREFIX=$(brew --prefix sdl2)
BWAPI_LIB=../../bwapi/lib

cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBWAPI_DIR=../../bwapi/ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_FLAGS="-arch x86_64 -I$ZMQ_PREFIX/include -I$SDL_PREFIX/include" \
  -DCMAKE_CXX_FLAGS="-arch x86_64 -I$ZMQ_PREFIX/include -I$SDL_PREFIX/include" \
  -DCMAKE_SHARED_LINKER_FLAGS="-arch x86_64 -L$ZMQ_PREFIX/lib -lzmq -L$SDL_PREFIX/lib -lSDL2 -L$BWAPI_LIB -lBWAPILIB"

cd ../..

echo "Installing Python bindings..."
pip install "pybind11<=2.6"

LDFLAGS="-L$APPS_DIR/lib -L$(brew --prefix sdl2)/lib" \
CFLAGS="-I$APPS_DIR/include -I$(brew --prefix sdl2)/include/SDL2" \
pip install -e .

# echo "Adding environment variables to ~/.zshrc..."
# echo 'export CFLAGS="-I'"$APPS_DIR"'/include -I'"$(brew --prefix sdl2)"'/include/SDL2"' >> ~/.zshrc
# echo 'export LDFLAGS="-L'"$APPS_DIR"'/lib -L'"$(brew --prefix sdl2)"'/lib"' >> ~/.zshrc
# echo 'export LD_LIBRARY_PATH="'"$APPS_DIR"'/lib:'"$(brew --prefix sdl2)"'/lib:$LD_LIBRARY_PATH"' >> ~/.zshrc
# echo 'export PKG_CONFIG_PATH="'"$APPS_DIR"'/lib/pkgconfig:'"$(brew --prefix sdl2)"'/lib/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.zshrc
# echo 'export DYLD_LIBRARY_PATH="'"$APPS_DIR"'/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc

# echo 'TorchCraft setup complete!'
# echo 'Copy your .mpq files into $BASE_DIR/TorchCraft to play.'