**Environment Setup (Apple Silicon / osx-64)**

Create an osx-64 architecture Conda environment to support legacy Python 3.6.15:

``` Bash
conda create -c conda-forge --platform osx-64 --name ic3 python=3.6.15
conda activate ic3

# Enforce osx-64 and conda-forge for future packages in this environment
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda env config vars set CONDA_SUBDIR=osx-64
```

**Clone & Install IC3Net**
``` Bash
git clone https://github.com/IC3Net/IC3Net

cd IC3Net
pip install -r requirements.txt

cd ic3net-envs
python setup.py develop
```

**TorchCraft & StarCraft Dependencies (Optional)**

To run experiments on StarCraft, first setup the following up-to-date to install torchcraft and `gym-starcraft` dependencies and then install the `gym-starcraft` package included in this repository.

``` Bash
# Directory Setup & Dependency Installation
TORCRAFT_DIR="$HOME/Documents/Tech/TorchCraft"

mkdir -p TORCRAFT_DIR
cd TORCRAFT_DIR
git clone https://github.com/TorchCraft/TorchCraft .
git submodule update --init --recursive

brew install zstd czmq zeromq sdl2 cmake pkg-config

export CXXFLAGS="-I$(brew --prefix)/include $CXXFLAGS"
export CFLAGS="-I$(brew --prefix)/include $CFLAGS"
export LDFLAGS="-L$(brew --prefix)/lib $LDFLAGS"
export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="$(brew --prefix):$CMAKE_PREFIX_PATH"

# Build BWAPI & OpenBW
cd "$TARGET_DIR"
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
make install -j$(sysctl -n hw.ncpu)

# Build BWEnv
cd "$TORCRAFT_DIR/BWEnv"
mkdir -p build && cd build
rm -rf * # clear cache to ensure clean build

cmake .. \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=relwithdebinfo \
  -DBWAPI_DIR="$TORCRAFT_DIR/bwapi_install" \
  -DCMAKE_PREFIX_PATH="$(brew --prefix)" \
  -DCMAKE_INCLUDE_PATH="$(brew --prefix)/include" \
  -DCMAKE_LIBRARY_PATH="$(brew --prefix)/lib"

make -j$(sysctl -n hw.ncpu)

# Install TorchCraft Python Bindings
conda activate ic3

cd "$TORCRAFT_DIR/py"
pip install -e .
```

Update `./gym-starcraft/gym_starcraft/envs/config.yml`.

I used wine to install the SC BW 1.16.1 windows installer I found online, and extracted the MPQs from the local files generated during the installation. And moved them to a new location and then: `export OPENBW_MPQ_PATH="/Users/ajuanijustus/Documents/Tech/sc_mpq"`


BrooDat.mpq
BroodWar.mpq
patch_rt.mpq
StarCraft.mpq
StarDat.mp

Also had to add this to `main.py`:
```python
import ctypes, sys, os

# Force load ZeroMQ with global symbol visibility for dynamic extensions
try:
    # Try Conda environment libzmq first
    ctypes.CDLL(os.path.join(sys.prefix, "lib", "libzmq.dylib"), mode=ctypes.RTLD_GLOBAL)
except OSError:
    try:
        # Fallback to Homebrew libzmq
        ctypes.CDLL("/opt/homebrew/lib/libzmq.dylib", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        ctypes.CDLL("libzmq.dylib", mode=ctypes.RTLD_GLOBAL)
```

And modified the config path in `gym-starcraft/gym_starcraft/envs/starcraft_base_env.py`:
```python
# self.config_path = config_path # original
self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml') # aju
```

```Bash
cd gym_starcraft
python setup.py develop
cd ..
```

Now you're ready to go back to the README.md and try running the training scripts.