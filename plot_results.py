import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION, REGEX & LEGENDS
# ==========================================
MODELS_DIRS = ["trained_models_aug_abl_1", "trained_models_aug_abl_2", "trained_models_aug_abl_3"] 
SAVE_DIR = "plots_aug"

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
    'phase4': '#a366ff'        
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
    if not os.path.exists(models_dir):
        print(f"Warning: Directory '{models_dir}' not found. Skipping.")
        continue

    print(f"Scanning directory: {models_dir}")
    for item in os.listdir(models_dir):
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
# 4. PLOTTING LOOP WITH SHADED ERROR AREAS
# ==========================================
for env, param_dict in dataset.items():
    for param_val, algo_dict in param_dict.items():
        print(f"Generating aggregated plots for {env} ({param_val})...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['success', 'reward', 'steps']
        titles = ['Success Rate (%)', 'Total Reward', 'Steps Taken']
        
        data_plotted = False 
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Sort algos based on their key position in LEGEND_MAPPING
            ordered_algos = [a for a in LEGEND_MAPPING if a in algo_dict] + \
                            [a for a in algo_dict if a not in LEGEND_MAPPING]

            for algo in ordered_algos:
                paths = algo_dict[algo]
                all_runs_data = []
                
                for path in paths:
                    try:
                        checkpoint = torch.load(path, map_location='cpu')
                        if 'log' in checkpoint:
                            data_vector = extract_metric(checkpoint['log'], metric)
                            if data_vector is not None and len(data_vector) > 0:
                                all_runs_data.append(data_vector)
                    except Exception as e:
                        print(f"  Warning: Could not read {path}. Skipping. Error: {e}")
                
                if not all_runs_data:
                    continue
                
                data_plotted = True
                
                # Find the absolute longest run across all seeds so NO data is cropped
                max_len = max(len(r) for r in all_runs_data)
                
                # Pad shorter runs with np.nan to safely align them in a 2D array
                padded_runs = np.full((len(all_runs_data), max_len), np.nan, dtype=np.float32)
                for i, r in enumerate(all_runs_data):
                    padded_runs[i, :len(r)] = r
                
                # Scale the raw data FIRST
                if metric == 'success':
                    padded_runs *= 100
                
                # Compute base cross-seed means & standard deviation
                mean_line = np.nanmean(padded_runs, axis=0)
                std_line = np.nanstd(padded_runs, axis=0, ddof=1)
                
                min_values = mean_line - std_line
                max_values = mean_line + std_line
                
                epochs = np.arange(1, max_len + 1)
                color = ALGO_COLORS.get(algo, '#7f7f7f')
                
                display_label = LEGEND_MAPPING.get(algo, algo.upper())
                
                ax.plot(epochs, mean_line, label=display_label, color=color, linewidth=1.5)
                ax.fill_between(epochs, min_values, max_values, color=color, alpha=0.2)

            ax.set_title(titles[idx], fontsize=14, fontweight='bold')
            ax.set_xlabel('Epochs', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='best', frameon=True)
                
        if data_plotted:
            env_title = env.replace('_', ' ').title()
            
            # Format subtitle depending on whether it's Starcraft (Map) or other environments (Difficulty)
            if env == 'starcraft':
                param_title = f"Map: {param_val}"
            else:
                param_title = param_val.upper() if param_val == 'na' else param_val.title()

            plt.suptitle(f"Aggregated Paper Results: {env_title} ({param_title})", fontsize=16, fontweight='bold', y=1.02)
            
            save_filename = f"{env}_{param_val}_metrics.png"
            save_path = os.path.join(SAVE_DIR, save_filename)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            print(f"  Saved aggregated plot to: {save_path}")
        
        plt.close(fig)

print("\nReplication tracking successfully finished!")