import torch
import os
import pandas as pd


# ============================================================
# LOAD STATE DICT
# ============================================================

def load_state_dict_only(path):
    """
    Load a checkpoint containing custom classes such as utils.LogField.

    These checkpoints are assumed to come from a trusted source.
    """

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------
    # Case 1: the checkpoint itself is a state_dict
    # --------------------------------------------------
    if isinstance(ckpt, dict):
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

        # --------------------------------------------------
        # Case 2: state_dict stored under a common key
        # --------------------------------------------------
        for key in [
            "state_dict",
            "model_state_dict",
            "policy_state_dict",
            "actor_state_dict",
            "network_state_dict",
            "weights",
            "model_weights",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):

                candidate = ckpt[key]

                if candidate and all(
                    torch.is_tensor(v) for v in candidate.values()
                ):
                    return candidate

    # --------------------------------------------------
    # Case 3: recursively search nested dictionaries
    # --------------------------------------------------
    def recursive_find_state_dict(obj):

        if isinstance(obj, dict):

            # Direct state dict
            if obj and all(torch.is_tensor(v) for v in obj.values()):
                return obj

            for value in obj.values():
                result = recursive_find_state_dict(value)

                if result is not None:
                    return result

        elif isinstance(obj, (list, tuple)):

            for value in obj:
                result = recursive_find_state_dict(value)

                if result is not None:
                    return result

        return None

    state_dict = recursive_find_state_dict(ckpt)

    if state_dict is not None:
        return state_dict

    raise ValueError(
        f"Could not find a state_dict in checkpoint:\n{path}\n"
        f"Checkpoint type: {type(ckpt)}"
    )


# ============================================================
# CLASSIFY TENSORS
# ============================================================

# Common PyTorch buffer names.
# These are NOT counted as learnable parameters.
KNOWN_BUFFER_SUFFIXES = {
    "running_mean",
    "running_var",
    "num_batches_tracked",
}


def is_likely_parameter(name, tensor):
    """
    State dicts do not contain an explicit Parameter/buffer flag.

    For standard neural networks, tensors associated with weights,
    biases, embeddings, recurrent matrices, etc. are parameters.

    Known BatchNorm-style buffers are excluded.
    """

    last_component = name.split(".")[-1]

    if last_component in KNOWN_BUFFER_SUFFIXES:
        return False

    return True


# ============================================================
# DETAILED BREAKDOWN
# ============================================================

def analyse_state_dict(path):
    state_dict = load_state_dict_only(path)

    rows = []

    for name, tensor in state_dict.items():

        if not torch.is_tensor(tensor):
            continue

        n = tensor.numel()
        likely_param = is_likely_parameter(name, tensor)

        rows.append({
            "name": name,
            "shape": tuple(tensor.shape),
            "numel": n,
            "likely_parameter": likely_param,
            "dtype": str(tensor.dtype),
        })

    df = pd.DataFrame(rows)

    parameter_count = df.loc[
        df["likely_parameter"], "numel"
    ].sum()

    buffer_count = df.loc[
        ~df["likely_parameter"], "numel"
    ].sum()

    print(f"\nCheckpoint:")
    print(path)

    print(f"\nTotal tensors:        {len(df):,}")
    print(f"Parameter tensors:    {df['likely_parameter'].sum():,}")
    print(f"Likely parameters:    {parameter_count:,}")
    print(f"Known buffers:        {buffer_count:,}")
    print(f"Total stored values:  {df['numel'].sum():,}")

    print("\nLayer/tensor breakdown:")
    print(
        df[
            [
                "name",
                "shape",
                "numel",
                "likely_parameter"
            ]
        ].to_string(index=False)
    )

    return df

def module_prefix(name):
    """
    Convert, for example:

        encoder.fc1.weight
        encoder.fc1.bias

    into:

        encoder.fc1
    """

    parts = name.split(".")

    if parts[-1] in {"weight", "bias"}:
        return ".".join(parts[:-1])

    return ".".join(parts[:-1])


def compare_parameter_counts(ic3net_df, phase5_df):

    # Only likely parameters
    ic = ic3net_df[
        ic3net_df["likely_parameter"]
    ].copy()

    p5 = phase5_df[
        phase5_df["likely_parameter"]
    ].copy()

    ic["module"] = ic["name"].apply(module_prefix)
    p5["module"] = p5["name"].apply(module_prefix)

    ic_counts = (
        ic.groupby("module")["numel"]
        .sum()
        .rename("ic3net_params")
    )

    p5_counts = (
        p5.groupby("module")["numel"]
        .sum()
        .rename("phase5_params")
    )

    comparison = pd.concat(
        [ic_counts, p5_counts],
        axis=1
    ).fillna(0)

    comparison["difference"] = (
        comparison["phase5_params"]
        - comparison["ic3net_params"]
    )

    comparison["pct_change"] = 0.0

    mask = comparison["ic3net_params"] > 0

    comparison.loc[mask, "pct_change"] = (
        100
        * comparison.loc[mask, "difference"]
        / comparison.loc[mask, "ic3net_params"]
    )

    comparison = comparison.sort_values(
        "difference",
        ascending=False
    )

    print("\n================================================")
    print("IC3Net vs Phase 5")
    print("================================================")

    print(comparison.to_string())

    print("\nTOTAL")

    ic_total = ic["numel"].sum()
    p5_total = p5["numel"].sum()

    difference = p5_total - ic_total

    print(f"IC3Net:       {ic_total:,}")
    print(f"Phase 5:      {p5_total:,}")
    print(f"Difference:   {difference:,}")
    print(
        f"Increase:     {100 * difference / ic_total:.2f}%"
    )
    print(
        f"Phase 5 / IC3Net: {p5_total / ic_total:.3f}x"
    )

    return comparison


ic3net_path = "trained_models/trained_models_aug_all_1/predator_prey_ic3net_easy_phase1_job12_20260813_193208"
phase5_path = "trained_models/trained_models_aug_all_temp_1/predator_prey_ic3net_easy_phase5_job0_20260815_212142"

ic3net_df = analyse_state_dict(ic3net_path)
phase5_df = analyse_state_dict(phase5_path)

comparison_df = compare_parameter_counts(
    ic3net_df,
    phase5_df
)

comparison_df.to_csv(
    "paper_diagnostics_results/ic3net_vs_phase5_parameter_breakdown.csv"
)

def compare_tensors(ic3net_df, phase5_df):

    ic = ic3net_df.set_index("name")
    p5 = phase5_df.set_index("name")

    all_names = sorted(
        set(ic.index) | set(p5.index)
    )

    rows = []

    for name in all_names:

        ic_params = (
            ic.loc[name, "numel"]
            if name in ic.index
            else 0
        )

        p5_params = (
            p5.loc[name, "numel"]
            if name in p5.index
            else 0
        )

        rows.append({
            "tensor": name,
            "ic3net": ic_params,
            "phase5": p5_params,
            "difference": p5_params - ic_params,
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "difference",
        ascending=False
    )

    return result

tensor_comparison = compare_tensors(
    ic3net_df,
    phase5_df
)

print(tensor_comparison.to_string(index=False))