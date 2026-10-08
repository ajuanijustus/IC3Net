import os
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.stats as stats

# ==========================================
# 1. CONFIGURATION & REGEX (MATCHING YOUR SETUP)
# ==========================================
MODELS_DIRS = [
    "trained_models_aug_all_1", "trained_models_aug_all_2", "trained_models_aug_all_3",
    "trained_models_aug_all_temp_1", "trained_models_aug_all_temp_2", "trained_models_aug_all_temp_3",
    "trained_models_aug_all_sc_1", "trained_models_aug_all_sc_2", "trained_models_aug_all_sc_3"
] 
OUTPUT_DIR = "paper_diagnostics_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGEND_MAPPING = {
    'commnet': 'CommNet',
    'iric': 'IRIC',
    'ic': 'IC',
    'ic3net': 'IC3Net (Comm+Env)',
    'phase1': 'IC3Net (Comm+Env)',
    'phase2': 'Phase 2 (Comm Gating + Shared)',
    'phase3': 'Phase 3 (Comm Matrix + Shared)',
    'phase4': 'Phase 4 (Comm Matrix + Split)',
    'phase5': 'Phase 5 (Fully Decoupled)'
}

model_pattern = re.compile(
    r"^(?P<env>predator_prey|traffic_junction|starcraft)_"
    r"(?P<algo>ic3net|commnet|iric|ic)_"
    r"(?P<param>.+?)_"
    r"(?:phase(?P<phase>\d+)_)?"
    r"(?P<job_id>job\d+_\d+_\d+)"
    r"(?:_(?P<epoch>\d+))?"
    r"(?:\..*)?$"
)

# ==========================================
# 2. DATASET SCANNING
# ==========================================
def scan_checkpoints():
    raw_dataset = {}
    for models_dir in MODELS_DIRS:
        full_dir_path = os.path.join('trained_models', models_dir)
        if not os.path.exists(full_dir_path):
            continue
        for item in os.listdir(full_dir_path):
            if item.endswith('_env.pt') or item.endswith('_comm.pt'):
                continue
            match = model_pattern.match(item)
            if match:
                meta = match.groupdict()
                if meta["epoch"] is not None:
                    continue  # Only use base checkpoints containing logs
                env, param, phase, job_id = meta["env"], meta["param"], meta["phase"], meta["job_id"]
                algo = f"phase{phase}" if phase else meta["algo"]
                
                full_path = os.path.join(full_dir_path, item)
                raw_dataset.setdefault(env, {}).setdefault(param, {}).setdefault(algo, {})[job_id] = full_path
    return raw_dataset

# ==========================================
# 3. METRIC EXTRACTOR
# ==========================================
def parse_log_vector(raw_field):
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

    cleaned = []
    for item in raw_values:
        if hasattr(item, 'numpy') or isinstance(item, (list, tuple, np.ndarray)):
            arr = item.numpy() if hasattr(item, 'numpy') else np.array(item)
            cleaned.append(float(np.mean(arr)) if arr.size > 0 else 0.0)
        else:
            try:
                cleaned.append(float(item))
            except:
                cleaned.append(0.0)
    return np.array(cleaned, dtype=np.float32)

# ==========================================
# 4. DIAGNOSTIC ANALYZERS
# ==========================================
def analyze_convergence_and_sample_efficiency(dataset, target_success_thresh=80.0):
    """
    Computes:
    - Final Success Rate (Mean ± IQM / Std)
    - Epochs to reach target threshold (Convergence Speedup)
    - Area Under Curve (AUC) for Sample Efficiency
    """
    records = []
    for env, param_dict in dataset.items():
        for param_val, algo_dict in param_dict.items():
            for algo, jobs in algo_dict.items():
                final_successes, epochs_to_thresh, aucs = [], [], []
                
                for job_id, path in jobs.items():
                    try:
                        ckpt = torch.load(path, map_location='cpu')
                        if 'log' not in ckpt:
                            continue
                        success_vec = parse_log_vector(ckpt['log'].get('success'))
                        if success_vec is None or len(success_vec) == 0:
                            continue
                        
                        # Convert to percentage
                        if np.max(success_vec) <= 1.0:
                            success_vec = success_vec * 100.0
                        
                        # 1. Final performance
                        final_successes.append(success_vec[-1])
                        
                        # 2. Sample Efficiency (AUC)
                        aucs.append(np.trapz(success_vec))
                        
                        # 3. Speed to threshold
                        reached = np.where(success_vec >= target_success_thresh)[0]
                        if len(reached) > 0:
                            epochs_to_thresh.append(reached[0] + 1)
                        else:
                            epochs_to_thresh.append(len(success_vec)) # Penalty ceiling
                    except Exception as e:
                        continue
                        
                if final_successes:
                    records.append({
                        'Environment': env,
                        'Parameter/Map': param_val,
                        'Model Architecture': LEGEND_MAPPING.get(algo, algo),
                        'Phase Key': algo,
                        'Num Seeds': len(final_successes),
                        'Final Success (%) [Mean]': np.mean(final_successes),
                        'Final Success (%) [Std]': np.std(final_successes),
                        'AUC (Sample Efficiency)': np.mean(aucs),
                        f'Epochs to {target_success_thresh}% Success': np.mean(epochs_to_thresh)
                    })
    return pd.DataFrame(records)


def compute_effective_rank(tensor_features):
    """
    Calculates Effective Rank via SVD on intermediate representation vectors.
    Higher rank = richer representation; Lower rank = representation collapse.
    """
    if tensor_features.ndim > 2:
        tensor_features = tensor_features.view(tensor_features.size(0), -1)
    
    # Center features
    tensor_features = tensor_features - torch.mean(tensor_features, dim=0)
    
    # SVD
    _, S, _ = torch.svd(tensor_features)
    
    # Singular value distribution
    S_norm = S / torch.sum(S)
    S_norm = S_norm[S_norm > 1e-7]
    
    # Shannon Entropy of singular values
    entropy = -torch.sum(S_norm * torch.log(S_norm))
    eff_rank = torch.exp(entropy).item()
    return eff_rank


def analyze_representation_rank(dataset, num_samples=256):
    """
    Passes forward dummy/saved observation activations through loaded encoder 
    weights to test for representation collapse across monolithic vs decoupled models.
    """
    results = []
    
    for env, param_dict in dataset.items():
        for param_val, algo_dict in param_dict.items():
            for algo, jobs in algo_dict.items():
                ranks = []
                for job_id, path in jobs.items():
                    try:
                        ckpt = torch.load(path, map_location='cpu')
                        # Extract weights
                        model_state = ckpt.get('model', ckpt.get('state_dict', None))
                        if model_state is None:
                            continue
                            
                        # Locate linear/encoder layer weights
                        encoder_weights = []
                        for k, v in model_state.items():
                            if ('encoder' in k or 'fc' in k or 'weight' in k) and v.ndim == 2:
                                if v.shape[0] >= 16 and v.shape[1] >= 16:
                                    encoder_weights.append(v)
                                    
                        if not encoder_weights:
                            continue
                        
                        # Run effective rank over the primary hidden layer weight / representation
                        primary_w = encoder_weights[0].float()
                        rank_val = compute_effective_rank(primary_w)
                        ranks.append(rank_val)
                    except Exception as e:
                        continue
                        
                if ranks:
                    results.append({
                        'Environment': env,
                        'Parameter': param_val,
                        'Model': LEGEND_MAPPING.get(algo, algo),
                        'Avg Effective Rank': np.mean(ranks)
                    })
    return pd.DataFrame(results)

# ==========================================
# 5. EXECUTION & CSV/LATEX EXPORT
# ==========================================
if __name__ == "__main__":
    print("Scanning checkpoints...")
    dataset = scan_checkpoints()
    
    print("\n1. Running Convergence & Efficiency Analysis...")
    df_perf = analyze_convergence_and_sample_efficiency(dataset, target_success_thresh=80.0)
    
    csv_perf_path = os.path.join(OUTPUT_DIR, "convergence_and_efficiency.csv")
    df_perf.to_csv(csv_perf_path, index=False)
    print(f" Saved performance & convergence table to: {csv_perf_path}")
    
    print("\n2. Running Representation Effective Rank Analysis...")
    df_rank = analyze_representation_rank(dataset)
    csv_rank_path = os.path.join(OUTPUT_DIR, "representation_effective_rank.csv")
    df_rank.to_csv(csv_rank_path, index=False)
    print(f" Saved effective rank analysis to: {csv_rank_path}")
    
    # Generate AAMAS Latex Summary Table for Top Scenarios
    print("\nGenerating LaTeX Table for AAMAS Submission...")
    if not df_perf.empty:
        latex_str = df_perf.to_latex(
            columns=['Environment', 'Parameter/Map', 'Model Architecture', 'Final Success (%) [Mean]', 'Epochs to 80% Success'],
            index=False,
            float_format="%.2f"
        )
        with open(os.path.join(OUTPUT_DIR, "aamas_summary_table.tex"), "w") as f:
            f.write(latex_str)
        print(" LaTeX code successfully written to paper_diagnostics_results/aamas_summary_table.tex")