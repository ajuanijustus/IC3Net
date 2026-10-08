import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION, REGEX & LEGENDS
# ==========================================
MODELS_DIRS = ["trained_models_aug_all_1", "trained_models_aug_all_2", "trained_models_aug_all_3", \
               "trained_models_aug_all_temp_1", "trained_models_aug_all_temp_2", "trained_models_aug_all_temp_3", \
               "trained_models_aug_all_sc_1", "trained_models_aug_all_sc_2", "trained_models_aug_all_sc_3"] 
SAVE_DIR = "paper_diagnostics_results"

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

KEY_MAPPING = {
    'success': 'success',
    'reward': 'reward',
    'steps': 'steps_taken'  
}

LEGEND_MAPPING = {
    'commnet': 'CommNet',
    'iric': 'IRIC',
    'ic': 'IC',
    'ic3net': 'IC3Net (Comm+Env)',
    'phase1': 'IC3Net (Comm+Env)',
    'phase2': 'Comm/Env (comm gating + shared encoder)',
    'phase3': 'Comm/Env (comm matrix + shared encoder)',
    'phase4': 'Comm/Env (comm matrix + split encoder)',
    'phase5': 'Comm/Env (fully decoupled)'
}

ALGO_COLORS = {
    'commnet': '#5785c1',      
    'ic3net': '#dc8d6d',       
    'iric': '#78b38a',         
    'ic': '#ba71af',           
    'phase1': '#dc8d6d',       
    'phase2': '#e6b800',       
    'phase3': '#df5454',       
    'phase4': '#a366ff',
    'phase5': "#00670c"      
}

# Unified Regex capturing either difficulty or map in group 3 (?P<param>)
model_pattern = re.compile(
    r"^(?P<env>predator_prey|traffic_junction|starcraft)_"  # Environment
    r"(?P<algo>ic3net|commnet|iric|ic)_"                    # Model/Algorithm
    r"(?P<param>.+?)_"                                       # Captures difficulty or map dynamically
    r"(?:phase(?P<phase>\d+)_)?"                             # Optional phase group
    r"(?P<job_id>job\d+_\d+_\d+)"                          # Job ID
    r"(?:_(?P<epoch>\d+))?"                                  # Captures trailing epoch digits if present
    r"(?:\..*)?$"                                            # Optional file extensions
)

# ==========================================
# 2. MULTI-DIR PARSE & BASE-ONLY FILTER
# ==========================================
raw_dataset = {}

for models_dir in MODELS_DIRS:
    models_dir = os.path.join('trained_models', models_dir)
    if not os.path.exists(models_dir):
        print(f"Warning: Directory '{models_dir}' not found. Skipping.")
        continue

    print(f"Scanning directory: {models_dir}")
    for item in os.listdir(models_dir):

        if item.endswith('_env.pt') or item.endswith('_comm.pt'):
            continue # Skip the standalone binaries – the training script is saving three things for phase 5: my_model.pt (monolithic with logs), my_model.pt_env.pt(weights only), and my_model.pt_comm.pt (weights only).

        match = model_pattern.match(item)
        if match:
            meta = match.groupdict()
            env, algo, param = meta["env"], meta["algo"], meta["param"]
            phase, job_id, epoch_str = (
                meta["phase"],
                meta["job_id"],
                meta["epoch"],
            )

            # --- ONLY USE BASE CHECKPOINT ---
            if epoch_str is not None:
                continue

            full_path = os.path.join(models_dir, item)

            if phase:
                algo = f"phase{phase}"

            env_dict = raw_dataset.setdefault(env, {})
            param_dict = env_dict.setdefault(param, {})
            algo_dict = param_dict.setdefault(algo, {})

            # Add the base checkpoint path using job_id as the seed key
            algo_dict[job_id] = full_path

# Flatten into the final structure for the plotting loop
dataset = {}
total_checkpoints = 0
for env, param_dict in raw_dataset.items():
    dataset[env] = {}
    for param_val, algo_dict in param_dict.items():
        dataset[env][param_val] = {}
        for algo, jobs in algo_dict.items():
            paths = list(jobs.values())
            dataset[env][param_val][algo] = paths
            total_checkpoints += len(paths)

print(
    f"\nFound a total of {total_checkpoints} unique base job runs across all directories."
)

# ==========================================
# 3. ENHANCED LOG PARSER
# ==========================================
def extract_metric(log_data, metric_key):
    actual_key = KEY_MAPPING[metric_key]
    raw_field = log_data.get(actual_key, None)
    
    if raw_field is None:
        return None

    if hasattr(raw_field, 'data'):
        raw_values = raw_field.data
    elif isinstance(raw_field, tuple) and len(raw_field) == 4 and isinstance(raw_field[0], list):
        raw_values = raw_field[0]
    else:
        raw_values = raw_field
        
    if not isinstance(raw_values, list):
        return None

    cleaned_values = []
    for item in raw_values:
        if hasattr(item, 'numpy') or isinstance(item, (list, tuple, np.ndarray)):
            arr = item.numpy() if hasattr(item, 'numpy') else np.array(item)
            if arr.size == 0:
                cleaned_values.append(0.0)
            else:
                cleaned_values.append(float(np.mean(arr)))
        elif isinstance(item, (bool, int, float, np.number)):
            cleaned_values.append(float(item))
        else:
            try:
                cleaned_values.append(float(item))
            except:
                cleaned_values.append(0.0)
                
    return cleaned_values

# ==========================================
# 4. CHECKPOINT PARAMETER COUNTING
# ==========================================
import os
import torch
import torch.nn as nn
from collections.abc import Mapping

def load_checkpoint(path):
    """
    Load a PyTorch checkpoint on CPU.

    First tries a full checkpoint load, which is needed when the file
    contains an nn.Module object. Falls back to weights_only=True for
    state-dict-only checkpoints.
    """
    try:
        # PyTorch >= 2.6: explicitly allow loading pickled model objects.
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
        return ckpt

    except Exception as e_full:
        print(f"[INFO] Full checkpoint load failed for {path}")
        print(f"       {type(e_full).__name__}: {e_full}")
        print("       Trying weights_only=True...")

        try:
            ckpt = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )
            return ckpt
        except Exception as e_weights:
            raise RuntimeError(
                f"Could not load checkpoint:\n{path}\n\n"
                f"Full load error:\n{e_full}\n\n"
                f"Weights-only error:\n{e_weights}"
            )


def find_module(obj, visited=None):
    """
    Recursively search a checkpoint for an nn.Module.

    This handles common checkpoint structures such as:
        checkpoint["model"]
        checkpoint["policy"]
        checkpoint["actor"]
        checkpoint["network"]
        ...
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    if isinstance(obj, nn.Module):
        return obj

    if isinstance(obj, Mapping):
        # Search likely locations first.
        priority_keys = [
            "model",
            "policy",
            "actor",
            "network",
            "net",
            "agent",
            "module",
            "policy_model",
        ]

        for key in priority_keys:
            if key in obj:
                found = find_module(obj[key], visited)
                if found is not None:
                    return found

        # Then recursively search everything else.
        for value in obj.values():
            found = find_module(value, visited)
            if found is not None:
                return found

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_module(value, visited)
            if found is not None:
                return found

    return None


def find_state_dict(obj, visited=None):
    """
    Recursively find a state_dict in a checkpoint.

    Returns None if no tensor dictionary can be found.
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    if isinstance(obj, Mapping):

        # Direct state_dict
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj

        # Common checkpoint keys
        priority_keys = [
            "state_dict",
            "model_state_dict",
            "policy_state_dict",
            "actor_state_dict",
            "network_state_dict",
            "weights",
            "model_weights",
        ]

        for key in priority_keys:
            if key in obj and isinstance(obj[key], Mapping):
                candidate = find_state_dict(obj[key], visited)
                if candidate is not None:
                    return candidate

        # Recursive search
        for value in obj.values():
            candidate = find_state_dict(value, visited)
            if candidate is not None:
                return candidate

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            candidate = find_state_dict(value, visited)
            if candidate is not None:
                return candidate

    return None


def count_module_parameters(model):
    """
    Exact parameter count from an nn.Module.

    parameters() deduplicates shared Parameter objects, which is important
    when an architecture contains shared encoders.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params


def count_state_dict_parameters(state_dict):
    """
    Count tensor elements in a state_dict.

    NOTE:
    A raw state_dict can contain buffers as well as Parameters
    (e.g. BatchNorm running statistics). Therefore this is a fallback
    estimate of stored tensor size, not necessarily an exact trainable
    parameter count.
    """
    return sum(
        tensor.numel()
        for tensor in state_dict.values()
        if torch.is_tensor(tensor)
    )


def count_checkpoint_parameters(path):
    """
    Return parameter information from a checkpoint.

    Returns
    -------
    dict with:
        total_params
        trainable_params
        method
        module
        state_dict
    """
    ckpt = load_checkpoint(path)

    # --------------------------------------------------
    # Preferred: actual nn.Module
    # --------------------------------------------------
    model = find_module(ckpt)

    if model is not None:
        total, trainable = count_module_parameters(model)

        return {
            "total_params": total,
            "trainable_params": trainable,
            "method": "nn.Module",
            "module": model,
            "state_dict": None,
        }

    # --------------------------------------------------
    # Fallback: state_dict
    # --------------------------------------------------
    state_dict = find_state_dict(ckpt)

    if state_dict is not None:
        total = count_state_dict_parameters(state_dict)

        return {
            "total_params": total,
            "trainable_params": None,
            "method": "state_dict",
            "module": None,
            "state_dict": state_dict,
        }

    raise ValueError(
        f"Could not find either an nn.Module or state_dict in:\n{path}\n"
        f"Checkpoint type: {type(ckpt)}"
    )


def format_number(n):
    """Human-readable parameter count."""
    if n is None:
        return "N/A"

    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f} M"

    if n >= 1_000:
        return f"{n / 1_000:.3f} K"

    return str(n)


# ==========================================
# 5. COUNT ALL CHECKPOINTS
# ==========================================

parameter_records = []

for env, param_dict in dataset.items():

    for scenario, algo_dict in param_dict.items():

        for algo, paths in algo_dict.items():

            for checkpoint_path in paths:

                try:
                    info = count_checkpoint_parameters(checkpoint_path)

                    record = {
                        "env": env,
                        "scenario": scenario,
                        "architecture": algo,
                        "checkpoint": checkpoint_path,
                        "total_params": info["total_params"],
                        "trainable_params": info["trainable_params"],
                        "method": info["method"],
                    }

                    parameter_records.append(record)

                    print(
                        f"{env:18s} | "
                        f"{scenario:25s} | "
                        f"{algo:8s} | "
                        f"{format_number(info['total_params']):>10s} | "
                        f"{format_number(info['trainable_params']):>10s}"
                    )

                except Exception as e:
                    print(
                        f"[ERROR] Could not count parameters for:\n"
                        f"        {checkpoint_path}\n"
                        f"        {type(e).__name__}: {e}"
                    )


# ==========================================
# 6. SAVE RESULTS
# ==========================================
import pandas as pd

parameter_df = pd.DataFrame(parameter_records)

parameter_csv = os.path.join(
    SAVE_DIR,
    "parameter_counts_all_checkpoints.csv"
)

parameter_df.to_csv(parameter_csv, index=False)

print("\nParameter-count table saved to:")
print(parameter_csv)

# ==========================================
# 7. COUNT MODULAR ENV / COMM COMPONENTS
# ==========================================

modular_records = []

for env, param_dict in dataset.items():

    for scenario, algo_dict in param_dict.items():

        # Only phase5 is fully modular in your naming scheme.
        if "phase5" not in algo_dict:
            continue

        for base_checkpoint in algo_dict["phase5"]:

            env_checkpoint = base_checkpoint + "_env.pt"
            comm_checkpoint = base_checkpoint + "_comm.pt"

            env_params = None
            comm_params = None

            # -----------------------------
            # Action / environment policy
            # -----------------------------
            if os.path.exists(env_checkpoint):
                try:
                    env_info = count_checkpoint_parameters(env_checkpoint)
                    env_params = env_info["total_params"]
                except Exception as e:
                    print(f"[ERROR] {env_checkpoint}: {e}")
            else:
                print(f"[WARNING] Missing: {env_checkpoint}")

            # -----------------------------
            # Communication policy
            # -----------------------------
            if os.path.exists(comm_checkpoint):
                try:
                    comm_info = count_checkpoint_parameters(comm_checkpoint)
                    comm_params = comm_info["total_params"]
                except Exception as e:
                    print(f"[ERROR] {comm_checkpoint}: {e}")
            else:
                print(f"[WARNING] Missing: {comm_checkpoint}")

            if env_params is not None and comm_params is not None:

                total_modular = env_params + comm_params

                modular_records.append({
                    "env": env,
                    "scenario": scenario,
                    "architecture": "phase5",
                    "checkpoint": base_checkpoint,
                    "env_policy_params": env_params,
                    "comm_policy_params": comm_params,
                    "total_modular_params": total_modular,
                })

modular_df = pd.DataFrame(modular_records)

modular_csv = os.path.join(
    SAVE_DIR,
    "phase5_modular_parameter_counts.csv"
)

modular_df.to_csv(modular_csv, index=False)

print("\nPhase-5 modular parameter counts:")
print(modular_df.to_string(index=False))

print("\nSaved to:")
print(modular_csv)

# ==========================================
# 8. PAPER SUMMARY
# ==========================================

if not parameter_df.empty:

    summary = (
        parameter_df
        .groupby(["env", "scenario", "architecture"], as_index=False)
        .agg(
            mean_params=("total_params", "mean"),
            std_params=("total_params", "std"),
            min_params=("total_params", "min"),
            max_params=("total_params", "max"),
            n_checkpoints=("total_params", "count"),
        )
    )

    summary["mean_params_M"] = summary["mean_params"] / 1e6
    summary["std_params_M"] = summary["std_params"].fillna(0) / 1e6

    print("\n==========================================")
    print("PARAMETER SUMMARY")
    print("==========================================")

    print(
        summary[
            [
                "env",
                "scenario",
                "architecture",
                "mean_params",
                "std_params",
                "n_checkpoints",
            ]
        ].to_string(index=False)
    )

    summary.to_csv(
        os.path.join(SAVE_DIR, "parameter_summary.csv"),
        index=False
    )

# ==========================================
# 9. PHASE4 vs PHASE5 PARAMETER DIFFERENCE
# ==========================================

compare = parameter_df[
    parameter_df["architecture"].isin(["phase4", "phase5"])
].copy()

pivot = (
    compare
    .groupby(["env", "scenario", "architecture"])["total_params"]
    .mean()
    .unstack("architecture")
)

if "phase4" in pivot.columns and "phase5" in pivot.columns:

    pivot["additional_params"] = (
        pivot["phase5"] - pivot["phase4"]
    )

    # Added parameters as a fraction of the modular model.
    pivot["added_pct_of_phase5"] = (
        100.0
        * pivot["additional_params"]
        / pivot["phase5"]
    )

    # Equivalent percentage increase relative to phase4.
    pivot["relative_increase_pct"] = (
        100.0
        * pivot["additional_params"]
        / pivot["phase4"]
    )

    print("\n==========================================")
    print("PHASE4 -> PHASE5")
    print("==========================================")

    print(
        pivot[
            [
                "phase4",
                "phase5",
                "additional_params",
                "added_pct_of_phase5",
                "relative_increase_pct",
            ]
        ].to_string()
    )

    print("\nOverall mean across scenarios:")
    print(
        f"Additional parameters: "
        f"{pivot['additional_params'].mean():,.0f}"
    )

    print(
        f"Added parameters as % of phase5: "
        f"{pivot['added_pct_of_phase5'].mean():.2f}%"
    )

    print(
        f"Relative increase over phase4: "
        f"{pivot['relative_increase_pct'].mean():.2f}%"
    )

def print_parameter_breakdown(checkpoint_path):
    info = count_checkpoint_parameters(checkpoint_path)

    if info["module"] is None:
        print(
            "This checkpoint is state-dict-only, so an exact "
            "Parameter-vs-buffer/layer breakdown is unavailable."
        )
        return

    model = info["module"]

    rows = []

    for name, param in model.named_parameters():
        rows.append({
            "name": name,
            "shape": tuple(param.shape),
            "params": param.numel(),
            "trainable": param.requires_grad,
        })

    df = pd.DataFrame(rows)

    print("\nLayer-by-layer parameter count:")
    print(df.to_string(index=False))

    print("\nTotal:")
    print(f"  Parameters: {df['params'].sum():,}")
    print(
        f"  Trainable:  "
        f"{df.loc[df['trainable'], 'params'].sum():,}"
    )

    return df

example_checkpoint = dataset["traffic_junction"]["easy"]["phase5"][0]

layer_df = print_parameter_breakdown(example_checkpoint)