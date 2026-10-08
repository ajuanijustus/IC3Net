import os
import re
import csv
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================
# 1. CONFIGURATION
# ============================================================
TARGET_FOLDERS = [
    ".",
    "trained_models_aug_all_1",
    "trained_models_aug_all_2",
    "trained_models_aug_all_3",
    "trained_models_aug_all_temp_1",
    "trained_models_aug_all_temp_2",
    "trained_models_aug_all_temp_3",
    "trained_models_aug_all_sc_1",
    "trained_models_aug_all_sc_2",
    "trained_models_aug_all_sc_3",
    "trained_models",
]

OUTPUT_DIR = "paper_diagnostics_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SUCCESS_THRESH = 80.0

LEGEND_MAPPING = {
    "commnet": "CommNet",
    "iric": "IRIC",
    "ic": "IC",
    "ic3net": "IC3Net (Comm+Env)",
    "phase1": "IC3Net (Comm+Env)",
    "phase2": "Phase 2 (Comm Gating + Shared)",
    "phase3": "Phase 3 (Comm Matrix + Shared)",
    "phase4": "Phase 4 (Comm Matrix + Split)",
    "phase5": "Phase 5 (Fully Decoupled)",
}

# Keep this strict: we do not want files from unrelated experiments silently included.
MODEL_PATTERN = re.compile(
    r"^(?P<env>predator_prey|traffic_junction|starcraft)_"
    r"(?P<algo>ic3net|commnet|iric|ic)_"
    r"(?P<param>.+?)_"
    r"(?:phase(?P<phase>\d+)_)?"
    r"(?P<job_id>job\d+_\d+_\d+)"
    r"(?:_(?P<epoch>\d+))?"
    r"(?:_(?P<split>comm|env))?"
    r"(?:\.pt|\.pth)?$"
)


# ============================================================
# 2. CHECKPOINT IO
# ============================================================
def load_checkpoint(path: str) -> Any:
    """Load a local checkpoint with an explicit, compatible torch.load policy."""
    try:
        # These are local experiment checkpoints. Explicitly setting weights_only=False
        # preserves compatibility with checkpoints containing optimizer/log objects.
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose weights_only.
        return torch.load(path, map_location="cpu")


def _prefix_state_dict(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Prefix keys so env/comm branches cannot overwrite identical parameter names."""
    return {f"{prefix}.{k}": v for k, v in state_dict.items()}


def extract_policy_net(ckpt: Any) -> Optional[Dict[str, torch.Tensor]]:
    """Extract a state dictionary from the checkpoint's known container conventions."""
    if not isinstance(ckpt, dict):
        return None

    if isinstance(ckpt.get("policy_net"), dict):
        return ckpt["policy_net"]

    for key in ("model", "state_dict", "agent", "net"):
        if isinstance(ckpt.get(key), dict):
            return ckpt[key]

    # Some checkpoints are themselves flat state dicts.
    if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    return None


def load_job_state_dict(paths: Dict[str, str]) -> Tuple[Optional[Dict[str, torch.Tensor]], List[str]]:
    """
    Load the model state for one job.

    Important Phase-5 fix:
    - If a single main checkpoint exists, use it.
    - Otherwise, if env+comm split checkpoints exist, load BOTH and prefix their keys.
    - Never silently analyze only the env half of a split model.
    """
    warnings: List[str] = []

    main_path = paths.get("main")
    if main_path:
        ckpt = load_checkpoint(main_path)
        state = extract_policy_net(ckpt)
        if state is None:
            warnings.append(f"No policy state_dict found in main checkpoint: {main_path}")
        return state, warnings

    env_path = paths.get("env")
    comm_path = paths.get("comm")
    if env_path and comm_path:
        env_ckpt = load_checkpoint(env_path)
        comm_ckpt = load_checkpoint(comm_path)
        env_state = extract_policy_net(env_ckpt) or {}
        comm_state = extract_policy_net(comm_ckpt) or {}

        if not env_state and not comm_state:
            warnings.append(
                f"Neither env nor comm state_dict could be extracted: {env_path} ; {comm_path}"
            )
            return None, warnings

        # Prefixes prevent same-named parameters from one branch overwriting the other.
        merged = {}
        merged.update(_prefix_state_dict(env_state, "env"))
        merged.update(_prefix_state_dict(comm_state, "comm"))
        return merged, warnings

    fallback_path = env_path or comm_path
    if fallback_path:
        ckpt = load_checkpoint(fallback_path)
        state = extract_policy_net(ckpt)
        if state is None:
            warnings.append(f"No policy state_dict found: {fallback_path}")
        return state, warnings

    warnings.append("No checkpoint path found for job")
    return None, warnings


# ============================================================
# 3. DATASET SCANNER
# ============================================================
def scan_checkpoints() -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, str]]]]], List[str]]:
    """Index checkpoints and explicitly report duplicate logical jobs."""
    raw_dataset: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, str]]]]] = {}
    scanned_dirs = set()
    warnings: List[str] = []

    search_dirs: List[str] = []
    for folder in TARGET_FOLDERS:
        for candidate in (folder, os.path.join("trained_models", folder)):
            if not os.path.isdir(candidate):
                continue
            abs_candidate = os.path.abspath(candidate)
            if abs_candidate not in scanned_dirs:
                scanned_dirs.add(abs_candidate)
                search_dirs.append(abs_candidate)

    total_files = 0
    for directory in search_dirs:
        try:
            items = os.listdir(directory)
        except OSError as exc:
            warnings.append(f"Could not scan directory {directory}: {exc}")
            continue

        for item in items:
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                continue

            match = MODEL_PATTERN.match(item)
            if not match:
                continue

            meta = match.groupdict()
            # Epoch-specific files are not final job checkpoints under the original convention.
            if meta["epoch"] is not None:
                continue

            env = meta["env"]
            param = meta["param"]
            phase = meta["phase"]
            job_id = meta["job_id"]
            split = meta["split"]
            algo = f"phase{phase}" if phase else meta["algo"]

            entry = (
                raw_dataset
                .setdefault(env, {})
                .setdefault(param, {})
                .setdefault(algo, {})
                .setdefault(job_id, {})
            )

            slot = split if split in ("env", "comm") else "main"
            previous = entry.get(slot)
            if previous and os.path.abspath(previous) != os.path.abspath(full_path):
                warnings.append(
                    f"Duplicate checkpoint for {env}/{param}/{algo}/{job_id}/{slot}: "
                    f"keeping {previous}, ignoring {full_path}"
                )
                continue

            entry[slot] = full_path
            total_files += 1

    num_jobs = sum(
        len(jobs)
        for param_dict in raw_dataset.values()
        for algo_dict in param_dict.values()
        for jobs in algo_dict.values()
    )
    print(
        f"Indexed {total_files} relevant checkpoint files across "
        f"{len(raw_dataset)} environment domains and {num_jobs} logical jobs."
    )
    return raw_dataset, warnings


# ============================================================
# 4. LOG PARSING
# ============================================================
def parse_log_vector(raw_field: Any) -> Optional[np.ndarray]:
    """Convert common checkpoint-log representations into a 1-D float array."""
    if raw_field is None:
        return None

    if isinstance(raw_field, torch.Tensor):
        raw_values = raw_field.detach().cpu().numpy()
    elif hasattr(raw_field, "data") and not isinstance(raw_field, np.ndarray):
        raw_values = raw_field.data
    elif isinstance(raw_field, tuple) and raw_field and isinstance(raw_field[0], list):
        raw_values = raw_field[0]
    else:
        raw_values = raw_field

    if isinstance(raw_values, pd.Series):
        raw_values = raw_values.tolist()

    if not isinstance(raw_values, (list, tuple, np.ndarray)):
        return None

    cleaned: List[float] = []
    for item in raw_values:
        if isinstance(item, torch.Tensor):
            arr = item.detach().cpu().numpy()
            cleaned.append(float(np.mean(arr)) if arr.size else 0.0)
        elif isinstance(item, (list, tuple, np.ndarray)):
            arr = np.asarray(item)
            cleaned.append(float(np.mean(arr)) if arr.size else 0.0)
        else:
            try:
                cleaned.append(float(item))
            except (TypeError, ValueError):
                cleaned.append(np.nan)

    arr = np.asarray(cleaned, dtype=np.float64)
    return arr if arr.size else None


def normalize_success_percent(success_vec: np.ndarray) -> np.ndarray:
    """Convert fraction logs [0,1] to percent while preserving percent logs."""
    vec = np.asarray(success_vec, dtype=np.float64)
    finite = vec[np.isfinite(vec)]
    if finite.size and np.nanmax(finite) <= 1.0:
        vec = vec * 100.0
    return vec


def extract_x_axis(log_data: Dict[str, Any], expected_len: int) -> Tuple[np.ndarray, str]:
    """Use logged steps/episodes when available; otherwise use the log index."""
    candidate_keys = (
        "steps", "step", "timesteps", "timestep", "episodes", "episode",
        "samples", "sample", "env_steps", "env_step"
    )
    for key in candidate_keys:
        vec = parse_log_vector(log_data.get(key))
        if vec is not None and len(vec) == expected_len:
            if np.all(np.isfinite(vec)) and np.all(np.diff(vec) >= 0):
                return vec, key
    return np.arange(expected_len, dtype=np.float64), "log_index"


def summarize_success(log_data: Dict[str, Any], target_success_thresh: float) -> Dict[str, Any]:
    """Compute outcome metrics without pretending unreached thresholds were reached."""
    success_vec = parse_log_vector(log_data.get("success"))
    if success_vec is None or len(success_vec) == 0:
        return {}

    success_vec = normalize_success_percent(success_vec)
    finite_mask = np.isfinite(success_vec)
    if not finite_mask.any():
        return {}

    # Keep the final finite observation rather than indexing a NaN tail.
    final_success = float(success_vec[np.where(finite_mask)[0][-1]])
    x, x_name = extract_x_axis(log_data, len(success_vec))

    valid = finite_mask & np.isfinite(x)
    xv = x[valid]
    yv = success_vec[valid]

    if len(xv) >= 2:
        try:
            auc = float(np.trapezoid(yv, xv))
        except AttributeError:
            auc = float(np.trapz(yv, xv))
        duration = float(xv[-1] - xv[0])
        normalized_auc = auc / duration if duration > 0 else np.nan
    else:
        auc = np.nan
        normalized_auc = np.nan

    reached = np.where(success_vec >= target_success_thresh)[0]
    if len(reached):
        threshold_step = int(x[reached[0]]) if np.isfinite(x[reached[0]]) else np.nan
        threshold_index = int(reached[0] + 1)
        reached_flag = 1
    else:
        threshold_step = np.nan
        threshold_index = np.nan
        reached_flag = 0

    return {
        "final_success": final_success,
        "success_auc": auc,
        "normalized_success_auc": normalized_auc,
        "x_axis_name": x_name,
        "threshold_x": threshold_step,
        "threshold_index": threshold_index,
        "threshold_reached": reached_flag,
        "num_log_points": int(len(success_vec)),
    }


# ============================================================
# 5. STATE-DICT STRUCTURAL METRICS
# ============================================================
def floating_weight_tensors(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    weights = {}
    for name, value in state_dict.items():
        if (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and ("weight" in name.lower())
            and value.ndim >= 2
        ):
            weights[name] = value.detach().cpu().float()

    # Explicit fallback for unusual naming, but still only matrices/tensors that can
    # meaningfully participate in SVD.
    if not weights:
        for name, value in state_dict.items():
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and value.ndim >= 2
            ):
                weights[name] = value.detach().cpu().float()
    return weights


def effective_rank(weight: torch.Tensor) -> float:
    """Effective rank = exp(entropy of normalized singular values)."""
    matrix = weight.reshape(weight.shape[0], -1)
    if min(matrix.shape) < 2:
        return np.nan

    try:
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered)
        total = singular_values.sum()
        if not torch.isfinite(total) or total <= 0:
            return np.nan
        p = singular_values / total
        p = p[p > 1e-12]
        entropy = -(p * torch.log(p)).sum()
        return float(torch.exp(entropy).item())
    except Exception:
        return np.nan


def classify_layer(name: str) -> Optional[str]:
    """Conservative layer classification: no arbitrary 'first half/second half' fallback."""
    lower = name.lower()
    encoder_keywords = ("encoder", "affine1", "env_encoder", "comm_encoder", "fc1", "embed")
    head_keywords = ("head", "heads", "value_head", "comm_f_module", "env_f_module", "lstm")

    is_head = any(k in lower for k in head_keywords)
    is_encoder = any(k in lower for k in encoder_keywords)

    if is_encoder and not is_head:
        return "encoder"
    if is_head:
        return "head"
    return None


def compute_structural_metrics(state_dict: Optional[Dict[str, torch.Tensor]]) -> Tuple[float, float, str]:
    """Compute architecture diagnostics without fabricated layer assignments."""
    if not state_dict:
        return np.nan, np.nan, "no_state_dict"

    weights = floating_weight_tensors(state_dict)
    if not weights:
        return np.nan, np.nan, "no_float_weight_tensors"

    ranks = [effective_rank(w) for w in weights.values()]
    ranks = [r for r in ranks if np.isfinite(r)]
    mean_rank = float(np.mean(ranks)) if ranks else np.nan

    enc_norms = []
    head_norms = []
    for name, weight in weights.items():
        group = classify_layer(name)
        if group == "encoder":
            enc_norms.append(float(torch.linalg.vector_norm(weight).item()))
        elif group == "head":
            head_norms.append(float(torch.linalg.vector_norm(weight).item()))

    if enc_norms and head_norms:
        ratio = float(np.mean(enc_norms) / (np.mean(head_norms) + 1e-12))
        ratio_status = "ok"
    else:
        ratio = np.nan
        ratio_status = "unclassified_encoder_or_head"

    return mean_rank, ratio, ratio_status


# ============================================================
# 6. GRADIENT DIAGNOSTIC
# ============================================================
def extract_stored_gradient_vectors(ckpt: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Use actual stored gradients only when the checkpoint explicitly contains them.

    Supported shape conventions:
      {"gradients": {"action": ..., "comm": ...}}
      {"gradients": {"policy": ..., "communication": ...}}
      {"grads": {"action": ..., "comm": ...}}

    We do NOT fabricate a 'training gradient' from final weights.
    """
    if not isinstance(ckpt, dict):
        return None

    container = ckpt.get("gradients")
    if container is None:
        container = ckpt.get("grads")
    if not isinstance(container, dict):
        return None

    action = None
    comm = None
    for key in ("action", "policy", "policy_gradient", "actor"):
        if key in container:
            action = container[key]
            break
    for key in ("comm", "communication", "communication_gradient", "gating"):
        if key in container:
            comm = container[key]
            break

    def flatten(value: Any) -> Optional[np.ndarray]:
        if isinstance(value, dict):
            parts = []
            for v in value.values():
                arr = flatten(v)
                if arr is not None:
                    parts.append(arr)
            return np.concatenate(parts) if parts else None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().reshape(-1).numpy()
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float64)
            return arr.reshape(-1)
        return None

    a = flatten(action)
    c = flatten(comm)
    if a is None or c is None or len(a) != len(c) or len(a) == 0:
        return None
    return a, c


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / denom)


def compute_gradient_metric(ckpt: Any) -> Tuple[float, str]:
    """Return real stored-gradient cosine only; otherwise return NaN with a reason."""
    vectors = extract_stored_gradient_vectors(ckpt)
    if vectors is None:
        return np.nan, "no_stored_action_comm_gradients"
    return cosine_similarity(vectors[0], vectors[1]), "stored_action_comm_gradients"


# ============================================================
# 7. DIAGNOSTIC ANALYZER
# ============================================================
def analyze_all_metrics(
    dataset: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, str]]]]],
    target_success_thresh: float = TARGET_SUCCESS_THRESH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for env, param_dict in dataset.items():
        for param_val, algo_dict in param_dict.items():
            for algo, jobs in algo_dict.items():
                final_successes: List[float] = []
                aucs: List[float] = []
                normalized_aucs: List[float] = []
                threshold_xs: List[float] = []
                threshold_reached = 0
                valid_log_seed_count = 0
                log_axis_names: List[str] = []

                gradient_cosines: List[float] = []
                gradient_statuses: List[str] = []
                ranks: List[float] = []
                ratios: List[float] = []
                ratio_statuses: List[str] = []
                valid_state_seed_count = 0

                for job_id, paths in sorted(jobs.items()):
                    try:
                        # Load checkpoint(s) exactly once for each logical job when possible.
                        ckpt_for_logs = None
                        if paths.get("main"):
                            ckpt_for_logs = load_checkpoint(paths["main"])
                        elif paths.get("env"):
                            ckpt_for_logs = load_checkpoint(paths["env"])
                        elif paths.get("comm"):
                            ckpt_for_logs = load_checkpoint(paths["comm"])

                        # ---------- Logs ----------
                        if isinstance(ckpt_for_logs, dict) and isinstance(ckpt_for_logs.get("log"), dict):
                            summary = summarize_success(
                                ckpt_for_logs["log"],
                                target_success_thresh,
                            )
                            if summary:
                                valid_log_seed_count += 1
                                final_successes.append(summary["final_success"])
                                if np.isfinite(summary["success_auc"]):
                                    aucs.append(summary["success_auc"])
                                if np.isfinite(summary["normalized_success_auc"]):
                                    normalized_aucs.append(summary["normalized_success_auc"])
                                if summary["threshold_reached"]:
                                    threshold_reached += 1
                                    threshold_xs.append(float(summary["threshold_x"]))
                                log_axis_names.append(summary["x_axis_name"])
                        else:
                            errors.append({
                                "Environment": env,
                                "Parameter/Map": param_val,
                                "Model Architecture": LEGEND_MAPPING.get(algo, algo),
                                "Phase Key": algo,
                                "Job ID": job_id,
                                "Stage": "logs",
                                "Error": "No log dictionary found in primary checkpoint",
                            })

                        # ---------- State / structure ----------
                        state_dict, load_warnings = load_job_state_dict(paths)
                        for warning in load_warnings:
                            errors.append({
                                "Environment": env,
                                "Parameter/Map": param_val,
                                "Model Architecture": LEGEND_MAPPING.get(algo, algo),
                                "Phase Key": algo,
                                "Job ID": job_id,
                                "Stage": "checkpoint",
                                "Error": warning,
                            })

                        if state_dict:
                            valid_state_seed_count += 1
                            rank_val, ratio_val, ratio_status = compute_structural_metrics(state_dict)
                            if np.isfinite(rank_val):
                                ranks.append(rank_val)
                            if np.isfinite(ratio_val):
                                ratios.append(ratio_val)
                            ratio_statuses.append(ratio_status)

                            grad_ckpt = ckpt_for_logs
                            if paths.get("main") is None and paths.get("comm") and paths.get("env"):
                                # Split Phase 5: there is no single primary checkpoint that
                                # contains both branches' possible gradient logs.
                                # We intentionally do not synthesize a gradient from weights.
                                grad_ckpt = None
                            grad_val, grad_status = compute_gradient_metric(grad_ckpt)
                            gradient_statuses.append(grad_status)
                            if np.isfinite(grad_val):
                                gradient_cosines.append(grad_val)

                    except Exception as exc:
                        errors.append({
                            "Environment": env,
                            "Parameter/Map": param_val,
                            "Model Architecture": LEGEND_MAPPING.get(algo, algo),
                            "Phase Key": algo,
                            "Job ID": job_id,
                            "Stage": "job",
                            "Error": f"{type(exc).__name__}: {exc}",
                        })

                def mean_or_nan(values: List[float]) -> float:
                    return float(np.mean(values)) if values else np.nan

                def std_or_nan(values: List[float]) -> float:
                    return float(np.std(values, ddof=1)) if len(values) >= 2 else (0.0 if len(values) == 1 else np.nan)

                # Only report a threshold time when at least one seed actually reached it.
                # The explicit reach rate prevents the old 'final epoch = reached threshold' bug.
                threshold_rate = (
                    threshold_reached / valid_log_seed_count if valid_log_seed_count else np.nan
                )

                if valid_log_seed_count or valid_state_seed_count:
                    records.append({
                        "Environment": env,
                        "Parameter/Map": param_val,
                        "Model Architecture": LEGEND_MAPPING.get(algo, algo),
                        "Phase Key": algo,
                        "Num Jobs Found": len(jobs),
                        "Valid Log Seeds": valid_log_seed_count,
                        "Valid State Seeds": valid_state_seed_count,
                        "Final Success (%)": mean_or_nan(final_successes),
                        "Final Success (Std)": std_or_nan(final_successes),
                        "Success AUC (x-axis units)": mean_or_nan(aucs),
                        "Normalized Success AUC (%)": mean_or_nan(normalized_aucs),
                        f"Mean x to {target_success_thresh}% (reached seeds only)": mean_or_nan(threshold_xs),
                        f"Reached {target_success_thresh}% Rate": threshold_rate,
                        "Gradient Cosine cos(theta) [stored grads only]": mean_or_nan(gradient_cosines),
                        "Effective Rank (mean layer rank)": mean_or_nan(ranks),
                        "Encoder/Head Ratio (classified layers only)": mean_or_nan(ratios),
                        "AUC x-axis sources": ";".join(sorted(set(log_axis_names))),
                        "Gradient metric status": ";".join(sorted(set(gradient_statuses))),
                        "Ratio metric status": ";".join(sorted(set(ratio_statuses))),
                    })

    df = pd.DataFrame(records)
    errors_df = pd.DataFrame(errors)
    return df, errors_df


# ============================================================
# 8. EXPORT
# ============================================================
def export_results(df: pd.DataFrame, errors_df: pd.DataFrame) -> None:
    csv_path = os.path.join(OUTPUT_DIR, "marl_comprehensive_diagnostics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved diagnostic CSV: {csv_path}")

    errors_path = os.path.join(OUTPUT_DIR, "marl_diagnostic_errors.csv")
    errors_df.to_csv(errors_path, index=False)
    print(f"Saved diagnostic/error log: {errors_path}")

    if not df.empty:
        preferred_cols = [
            "Environment",
            "Parameter/Map",
            "Model Architecture",
            "Final Success (%)",
            "Final Success (Std)",
            "Normalized Success AUC (%)",
            f"Mean x to {TARGET_SUCCESS_THRESH}% (reached seeds only)",
            f"Reached {TARGET_SUCCESS_THRESH}% Rate",
            "Gradient Cosine cos(theta) [stored grads only]",
            "Effective Rank (mean layer rank)",
            "Encoder/Head Ratio (classified layers only)",
        ]
        available_cols = [c for c in preferred_cols if c in df.columns]
        latex_str = df.to_latex(
            columns=available_cols,
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "",
        )
        tex_path = os.path.join(OUTPUT_DIR, "aamas_full_results_table.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"Saved LaTeX table: {tex_path}")


# ============================================================
# 9. EXECUTION
# ============================================================
if __name__ == "__main__":
    print("Scanning directories for MARL checkpoints...")
    dataset, scan_warnings = scan_checkpoints()

    if scan_warnings:
        print(f"\n[WARN] {len(scan_warnings)} scanner warnings.")
        for warning in scan_warnings[:20]:
            print(f"  - {warning}")
        if len(scan_warnings) > 20:
            print(f"  ... and {len(scan_warnings) - 20} more (see your console/log capture).")

    if not dataset:
        print("\n[ERROR] No checkpoint files found.")
    else:
        print("\nProcessing corrected diagnostics...")
        df, errors_df = analyze_all_metrics(
            dataset,
            target_success_thresh=TARGET_SUCCESS_THRESH,
        )
        export_results(df, errors_df)

        print("\nDone.")
        print("Important: gradient cosine is reported ONLY when actual stored action/communication gradients exist.")
        print("Threshold times exclude seeds that never reached the target; use the reach-rate column alongside them.")
