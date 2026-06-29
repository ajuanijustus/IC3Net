import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
MODELS_DIR = "trained_models_3" 
SAVE_DIR = "plots"

# Added distinct colors for each phase line
ALGO_COLORS = {
    'commnet': '#5785c1',      # Blue
    'ic3net': '#dc8d6d',       # Peach/Orange
    'iric': '#78b38a',         # Green
    'ic': '#ba71af',           # Purple
    'phase2': '#e6b800',       # Gold/Yellow
    'phase3': '#df5454',       # Coral Red
    'phase4': '#a366ff'        # Lavender
}

# Unified Regex: Parses standard baselines AND safely handles optional phase injection
checkpoint_1500_pattern = re.compile(
    r"^(?P<env>predator_prey|traffic_junction)_"
    r"(?P<algo>ic3net|commnet|iric|ic)_"
    r"(?P<diff>na|easy|medium|hard)_"
    r"(?:phase(?P<phase>\d+)_)?"  # Optional phase-capture group
    r"job\d+_\d+_\d+_1500$" 
)

dataset = {}
for item in os.listdir(MODELS_DIR):
    match = checkpoint_1500_pattern.match(item)
    if match:
        meta = match.groupdict()
        env, algo, diff, phase = meta['env'], meta['algo'], meta['diff'], meta['phase']
        full_path = os.path.join(MODELS_DIR, item)
        
        # TREAT PHASES AS DIFFERENT ALGORITHMS
        # If a phase exists, overwrite the algorithm name so it plots as its own line
        if phase:
            algo = f"phase{phase}"
        
        dataset.setdefault(env, {}).setdefault(diff, {}).setdefault(algo, []).append(full_path)

# ==================================================================================
# MANUAL PARSE AND GROUP MODEL PATHS
# dataset = {}

# manual_env  = "phase2_predator_prey"    # Options: 'predator_prey' or 'traffic_junction'
# manual_algo = "wip"                     # Options: 'ic3net', 'commnet', 'iric', 'ic'
# manual_diff = "easy"                    # Options: 'na', 'easy', 'medium', 'hard'
# manual_path = "trained_models/phase2_1000" 
# # --------------------------------------------------

# # Reconstruct the exact dictionary hierarchy the downstream loop expects
# dataset = {
#     manual_env: {
#         manual_diff: {
#             manual_algo: [manual_path]
#         }
#     }
# }
# ==================================================================================

print(f"Found {sum(len(paths) for env in dataset.values() for diff in env.values() for paths in diff.values())} checkpoints at epoch 1500.")

# ==========================================
# 2. ENHANCED LOG PARSER (Unpacks Custom LogField Objects)
# ==========================================
def extract_metric(log_data, metric_key):
    """
    Extracts the full history sequence hidden inside the original repo's 
    custom LogField objects, flattening nested tracking lists safely.
    """
    actual_key = KEY_MAPPING[metric_key]
    raw_field = log_data.get(actual_key, None)
    
    if raw_field is None:
        return None

    # --- UNPACK LOGFIELD QUIRK ---
    # Case A: Object unpickled with its named class property intact
    if hasattr(raw_field, 'data'):
        raw_values = raw_field.data
    # Case B: Object unpickled as a raw fallback tuple (data_list, plot_flag, x_axis, divide_by)
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
# 3. PLOTTING LOOP WITH EXACT PAPER REPLICATION MATH
# ==========================================
for env, diff_dict in dataset.items():
    for diff, algo_dict in diff_dict.items():
        print(f"Generating replicated plots for {env} ({diff})...")
        
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
                        checkpoint = torch.load(path)
                        if 'log' in checkpoint:
                            data_vector = extract_metric(checkpoint['log'], metric)
                            if data_vector is not None and len(data_vector) > 0:
                                all_runs_data.append(data_vector)
                    except Exception as e:
                        print(f"  Warning: Could not read {path}. Skipping. Error: {e}")
                
                if not all_runs_data:
                    continue
                
                data_plotted = True
                
                # --- REPLICATION QUIRK 1: Hard crop timeline at 1000 epochs max ---
                min_len = min(min(len(r) for r in all_runs_data), 1000)
                truncated_runs = np.array([r[:min_len] for r in all_runs_data], dtype=np.float32)
                
                # Compute base cross-seed means
                mean_line = np.mean(truncated_runs, axis=0)
                
                # --- REPLICATION QUIRK 2: Scale Success rate by 100 to show % ---
                if metric == 'success':
                    mean_line *= 100
                
                min_values = []
                max_values = []
                
                # Compute paper-compliant deviation envelop bounds epoch-by-epoch
                for epoch_idx in range(min_len):
                    val_at_epoch = truncated_runs[:, epoch_idx]
                    
                    # --- REPLICATION QUIRK 3: Cross-seed Variance instead of Standard Deviation ---
                    variance = float(np.var(val_at_epoch))
                    
                    if metric == 'success':
                        variance *= 100
                    
                    # --- REPLICATION QUIRK 4: Hard ceiling cap on variance bounds at 20 ---
                    variance = variance if variance < 20 else 20
                    
                    max_values.append(mean_line[epoch_idx] + variance)
                    min_values.append(mean_line[epoch_idx] - variance)
                
                epochs = np.arange(1, min_len + 1)
                color = ALGO_COLORS.get(algo, '#7f7f7f')
                
                # Plot trajectories matching original presentation formats
                ax.plot(epochs, mean_line, label=algo.upper(), color=color, linewidth=1.5)
                ax.fill_between(epochs, min_values, max_values, color=color, alpha=0.2)
            
            ax.set_title(titles[idx], fontsize=14, fontweight='bold')
            ax.set_xlabel('Epochs', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if idx == 0:
                ax.legend(loc='best', frameon=True)
        
        if data_plotted:
            env_title = env.replace('_', ' ').title()
            diff_title = diff.upper() if diff == 'na' else diff.title()
            plt.suptitle(f"Replicated Paper Results: {env_title} ({diff_title})", fontsize=16, fontweight='bold', y=1.02)
            
            save_filename = f"{env}_{diff}_replicated_metrics.png"
            save_path = os.path.join(SAVE_DIR, save_filename)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            print(f"  Saved replicated plot to: {save_path}")
        
        plt.close(fig)

print("\nReplication tracking successfully finished!")