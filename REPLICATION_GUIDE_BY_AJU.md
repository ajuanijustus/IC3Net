# IC3NET REPLICATION GUIDE (MacOS)

## Environment Setup (Apple Silicon / osx-64)

Rosetta 2 is required for running Intel x86 binaries on M1. If you haven't installed Rosetta 2 yet, open your Terminal and run:

```Bash
softwareupdate --install-rosetta --agree-to-license
```

Create an osx-64 architecture Conda environment to support legacy Python 3.6.15:

``` Bash
conda create -c conda-forge --platform osx-64 --name ic3 python=3.6.15
conda activate ic3

# Enforce osx-64 and conda-forge for future packages in this environment
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda env config vars set CONDA_SUBDIR=osx-64
```

## Clone & Install IC3Net
``` Bash
git clone https://github.com/IC3Net/IC3Net

cd IC3Net
pip install -r requirements.txt

cd ic3net-envs
python setup.py develop
```

## TorchCraft & StarCraft Dependencies

To run experiments on StarCraft, first setup the following up-to-date to install torchcraft and `gym-starcraft` dependencies and then install the `gym-starcraft` package included in this repository.

``` Bash
# Directory Setup & Dependency Installation
TORCRAFT_DIR="$HOME/Documents/TorchCraft"

mkdir -p "$TORCRAFT_DIR"
cd "$TORCRAFT_DIR"
git clone https://github.com/TorchCraft/TorchCraft .
git submodule update --init --recursive

brew install zstd czmq zeromq sdl2 cmake pkg-config

export CXXFLAGS="-I$(brew --prefix)/include $CXXFLAGS"
export CFLAGS="-I$(brew --prefix)/include $CFLAGS"
export LDFLAGS="-L$(brew --prefix)/lib $LDFLAGS"
export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="$(brew --prefix):$CMAKE_PREFIX_PATH"

# Build BWAPI & OpenBW
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
make install -j$(sysctl -n hw.ncpu)

# Build BWEnv
cd "$TORCRAFT_DIR/BWEnv"
mkdir -p build && cd build
# rm -rf * # clear cache to ensure clean build

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
conda install -c conda-forge zeromq pkg-config

export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/opt/homebrew/lib:$LD_LIBRARY_PATH"

cd "$TORCRAFT_DIR"
pip install -e . --no-cache-dir
```

### Configure StarCraft MPQ Assets

1. Obtain the StarCraft 1.16.1 Windows installer online, install it using wine (or a Windows environment) and extract the required MPQ files.
2. Place the MPQs into a dedicated directory (e.g., `$HOME/Documents/Tech/sc_mpq`), ensuring the following files are present:
   - `BrooDat.mpq`
   - `BroodWar.mpq`
   - `patch_rt.mpq`
   - `StarCraft.mpq`
   - `StarDat.mpq`
3. Export the path variable in your shell configuration or current terminal session:
```Bash
export OPENBW_MPQ_PATH="$HOME/Documents/Tech/sc_mpq"
```
4. Update `./gym-starcraft/gym_starcraft/envs/config.yml` to point to your StarCraft installation assets.

### Finalize `gym-starcraft` Installation
```Bash
cd gym-starcraft
python setup.py develop
cd ..
```

## Notes
**Patch notes to original IC3Net get it working:**
1. Fix ZeroMQ Dynamic Loading (`main.py`): To prevent symbol visibility errors with czmq / libzmq extensions on modern macOS architectures, added the following snippet to `main.py`:
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

2. Fix Configuration Path Resolution (`starcraft_base_env.py`)
Update `./gym-starcraft/gym_starcraft/envs/starcraft_base_env.py` so the config path resolves dynamically regardless of your working directory execution context:
```python
# self.config_path = config_path # original
self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
```

**To test the setup with nprocesses 1:**
```Bash
TORCRAFT_DIR="$HOME/Documents/TorchCraft"

python -u main.py --env_name starcraft --task_type explore --nagents 10 --num_epochs 1000 --hid_size 128 --lrate 0.002 --max_steps 60 --nprocesses 1 --torchcraft_dir="$TORCRAFT_DIR" --frame_skip 8 --nenemies 1 --our_unit_type 34 --enemy_unit_type 34 --init_range_end 150 --ic3net --recurrent --rnn_type LSTM --detach_gap 10 --stay_near_enemy --explore_vision 10 --step_size 16

Phase 2: 
single trainer runs:
python main.py --env_name predator_prey --nagents 3 --nprocesses 1 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --ic3net --vision 0 --recurrent --phase 2 --save "trained_models/phase2" --save_every 500

python main.py --env_name predator_prey --nagents 3 --nprocesses 1 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --ic3net --vision 0 --recurrent --phase 3 --save "trained_models/phase3" --save_every 500

python main.py --env_name predator_prey --nagents 3 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --ic3net --vision 0 --recurrent --phase 4 --save "trained_models/phase4" --save_every 500

--env_name traffic_junction --nagents 5 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 6 --max_steps 20 --ic3net --vision 0 --recurrent --add_rate_min 0.1 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --difficulty easy
```

