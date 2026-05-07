from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
VERSION1_DIR = ROOT / "version_1"
DEFAULT_DATASET = VERSION1_DIR / "heat_exchanger_dataset.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def evenly_distributed_counts(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def select_diverse_rows(group: pd.DataFrame, n_rows: int, flow_column: str, random_state: int) -> pd.DataFrame:
    if n_rows <= 0:
        return group.iloc[0:0].copy()
    if n_rows >= len(group):
        return group.copy()

    shuffled = group.sample(frac=1.0, random_state=random_state).sort_values(flow_column)
    positions = np.linspace(0, len(shuffled) - 1, num=n_rows)
    indices = np.unique(np.round(positions).astype(int))

    while len(indices) < n_rows:
        remaining = [i for i in range(len(shuffled)) if i not in set(indices)]
        indices = np.sort(np.append(indices, remaining[0]))

    return shuffled.iloc[indices[:n_rows]].copy()


def select_representative_rows(
    group: pd.DataFrame,
    n_rows: int,
    flow_column: str,
    random_state: int,
    group_index: int,
    total_groups: int,
) -> pd.DataFrame:
    if n_rows <= 0:
        return group.iloc[0:0].copy()
    if n_rows >= len(group):
        return group.copy()

    sorted_group = group.sort_values(flow_column).reset_index(drop=True)

    # For single-row selection, vary the chosen flow position across temperature groups
    # so a tiny subset does not collapse to the same flow value everywhere.
    if n_rows == 1:
        # Use a coprime step so neighboring temperature groups do not always
        # align with neighboring flow values in the final subset.
        step = 37
        row_index = int((group_index * step + random_state) % len(sorted_group))
        return sorted_group.iloc[[row_index]].copy()

    return select_diverse_rows(
        group=group,
        n_rows=n_rows,
        flow_column=flow_column,
        random_state=random_state,
    )


def build_low_data_subset(
    df: pd.DataFrame,
    target_size: int,
    temp_column: str,
    flow_column: str,
    temp_bins: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    unique_temperatures = np.sort(working[temp_column].unique())
    use_exact_temperature_groups = len(unique_temperatures) <= target_size

    if use_exact_temperature_groups:
        working["_temp_group"] = working[temp_column]
        temp_groups = []
        for temp_value in unique_temperatures:
            group = working.loc[working["_temp_group"] == temp_value].copy()
            if group.empty:
                continue
            temp_groups.append((float(temp_value), group))
        grouping_mode = "exact_temperature"
    else:
        if target_size < temp_bins:
            raise ValueError(
                f"target_size={target_size} is too small for temp_bins={temp_bins}. "
                "Use at least one sample per temperature bin."
            )
        temp_edges = np.linspace(working[temp_column].min(), working[temp_column].max(), temp_bins + 1)
        working["_temp_group"] = pd.cut(
            working[temp_column],
            bins=temp_edges,
            include_lowest=True,
            labels=False,
        )
        temp_groups = []
        for temp_bin in range(temp_bins):
            group = working.loc[working["_temp_group"] == temp_bin].copy()
            if group.empty:
                continue
            temp_groups.append((int(temp_bin), group))
        grouping_mode = "temperature_bin"

    counts = evenly_distributed_counts(target_size, len(temp_groups))

    selected_parts = []
    summary_rows = []
    total_groups = len(temp_groups)
    for group_index, ((temp_group, group), take_count) in enumerate(zip(temp_groups, counts)):
        selected = select_representative_rows(
            group=group,
            n_rows=min(take_count, len(group)),
            flow_column=flow_column,
            random_state=random_state + len(summary_rows),
            group_index=group_index,
            total_groups=total_groups,
        )
        selected_parts.append(selected)
        summary_rows.append(
            {
                "grouping_mode": grouping_mode,
                "temp_group": temp_group,
                "bin_min_temp_k": float(group[temp_column].min()),
                "bin_max_temp_k": float(group[temp_column].max()),
                "available_rows": int(len(group)),
                "selected_rows": int(len(selected)),
                "selected_min_flow": float(selected[flow_column].min()),
                "selected_max_flow": float(selected[flow_column].max()),
            }
        )

    subset = pd.concat(selected_parts, axis=0).drop(columns=["_temp_group"]).sort_index()
    summary = pd.DataFrame(summary_rows)
    return subset, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create low-data subsets that preserve hot-inlet temperature coverage."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the full dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store sampled datasets and summaries.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 125],
        help="Target subset sizes to generate.",
    )
    parser.add_argument(
        "--temp-column",
        default="hot_inlet_temperature_k",
        help="Column used to preserve temperature-range diversity.",
    )
    parser.add_argument(
        "--flow-column",
        default="cold_inlet_mass_flow_kg_s",
        help="Column used for within-temperature-bin diversity.",
    )
    parser.add_argument(
        "--temp-bins",
        type=int,
        default=10,
        help="Number of operating-temperature bins to preserve.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible tie-breaking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for size in args.sizes:
        subset, summary = build_low_data_subset(
            df=df,
            target_size=size,
            temp_column=args.temp_column,
            flow_column=args.flow_column,
            temp_bins=args.temp_bins,
            random_state=args.random_state,
        )

        subset_path = args.output_dir / f"low_data_{size}.csv"
        summary_path = args.output_dir / f"low_data_{size}_summary.csv"
        subset.to_csv(subset_path, index=False)
        summary.to_csv(summary_path, index=False)

        manifest_rows.append(
            {
                "subset_size": int(size),
                "saved_rows": int(len(subset)),
                "temp_min_k": float(subset[args.temp_column].min()),
                "temp_max_k": float(subset[args.temp_column].max()),
                "unique_temp_values": int(subset[args.temp_column].nunique()),
                "flow_min": float(subset[args.flow_column].min()),
                "flow_max": float(subset[args.flow_column].max()),
                "subset_file": str(subset_path.name),
                "summary_file": str(summary_path.name),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "sampling_manifest.csv", index=False)

    print("Low-data subsets created successfully.")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
