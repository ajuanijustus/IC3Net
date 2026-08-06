import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION, REGEX & LEGENDS
# ==========================================
# Now accepts a list of directories to aggregate together
MODELS_DIRS = ["trained_models_abl_3", "trained_models_abl_4", "trained_models_abl_5"] 
SAVE_DIR = "plots"

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

KEY_MAPPING = {
    'success': 'success',
    'reward': 'reward',
    'steps': 'steps_taken'  
}

LEGEND_MAPPING = {
    'commnet': 'CommNet',
    'ic3net': 'IC3Net',
    'iric': 'IRIC',
    'ic': 'IC',
    'phase1': 'IC3Net',
    'phase2': 'Phase 2 (shared encoder)',
    'phase3': 'Phase 3 (comm matrix))',
    'phase4': 'Phase 4 (comm matrix + split encoder)'
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

# Unified Regex for grouping jobs across variations
model_pattern = re.compile(
    r"^(?P<env>predator_prey|traffic_junction)_"
    r"(?P<algo>ic3net|commnet|iric|ic)_"
    r"(?P<diff>na|easy|medium|hard)_"
    r"(?:phase(?P<phase>\d+)_)?"      
    r"(?P<job_id>job\d+_\d+_\d+)"     
    r"(?:_(?P<epoch>\d+))?"           # Captures trailing epoch digits if present
    r"(?:\..*)?$"                     
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
            env, algo, diff = meta['env'], meta['algo'], meta['diff']
            phase, job_id, epoch_str = meta['phase'], meta['job_id'], meta['epoch']
            
            # --- ONLY USE BASE CHECKPOINT ---
            # If an epoch suffix exists (_500, _1000, etc.), completely skip it
            if epoch_str is not None:
                continue
                
            full_path = os.path.join(models_dir, item)
            
            if phase:
                algo = f"phase{phase}"
                
            env_dict = raw_dataset.setdefault(env, {})
            diff_dict = env_dict.setdefault(diff, {})
            algo_dict = diff_dict.setdefault(algo, {})
            
            # Add the base checkpoint path using job_id as the seed key
            algo_dict[job_id] = full_path

# Flatten into the final structure for the plotting loop
dataset = {}
total_checkpoints = 0
for env, diff_dict in raw_dataset.items():
    dataset[env] = {}
    for diff, algo_dict in diff_dict.items():
        dataset[env][diff] = {}
        for algo, jobs in algo_dict.items():
            paths = list(jobs.values())
            dataset[env][diff][algo] = paths
            total_checkpoints += len(paths)

print(f"\nFound a total of {total_checkpoints} unique base job runs across all directories.")

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
for env, diff_dict in dataset.items():
    for diff, algo_dict in diff_dict.items():
        print(f"Generating aggregated plots for {env} ({diff})...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['success', 'reward', 'steps']
        titles = ['Success Rate (%)', 'Total Reward', 'Steps Taken']
        
        data_plotted = False 
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for algo, paths in algo_dict.items():
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
                
                # REPLICATION QUIRK 1: Hard crop timeline at 1000 epochs max
                min_len = min(min(len(r) for r in all_runs_data), 1000)
                truncated_runs = np.array([r[:min_len] for r in all_runs_data], dtype=np.float32)
                
                # Compute base cross-seed means across all aggregated base runs
                mean_line = np.mean(truncated_runs, axis=0)
                
                # REPLICATION QUIRK 2: Scale Success rate by 100 to show %
                if metric == 'success':
                    mean_line *= 100
                
                min_values = []
                max_values = []
                
                # Compute paper-compliant deviation envelope bounds epoch-by-epoch
                for epoch_idx in range(min_len):
                    val_at_epoch = truncated_runs[:, epoch_idx]
                    
                    # REPLICATION QUIRK 3: Cross-seed Variance instead of Standard Deviation
                    variance = float(np.var(val_at_epoch))
                    
                    if metric == 'success':
                        variance *= 100
                    
                    # REPLICATION QUIRK 4: Hard ceiling cap on variance bounds at 20
                    variance = variance if variance < 20 else 20
                    
                    max_values.append(mean_line[epoch_idx] + variance)
                    min_values.append(mean_line[epoch_idx] - variance)
                
                epochs = np.arange(1, min_len + 1)
                color = ALGO_COLORS.get(algo, '#7f7f7f')
                
                # Fetch custom legend layout name or fall back to capitalized key
                display_label = LEGEND_MAPPING.get(algo, algo.upper())
                
                # Plot mean trajectories and their shaded variance areas
                ax.plot(epochs, mean_line, label=display_label, color=color, linewidth=1.5)
                ax.fill_between(epochs, min_values, max_values, color=color, alpha=0.2)
            
            ax.set_title(titles[idx], fontsize=14, fontweight='bold')
            ax.set_xlabel('Epochs', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if idx == 0:
                ax.legend(loc='best', frameon=True)
        
        if data_plotted:
            env_title = env.replace('_', ' ').title()
            diff_title = diff.upper() if diff == 'na' else diff.title()
            plt.suptitle(f"Aggregated Paper Results: {env_title} ({diff_title})", fontsize=16, fontweight='bold', y=1.02)
            
            save_filename = f"{env}_{diff}_metrics_jul6.png"
            save_path = os.path.join(SAVE_DIR, save_filename)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            print(f"  Saved aggregated plot to: {save_path}")
        
        plt.close(fig)

print("\nReplication tracking successfully finished!")