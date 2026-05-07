#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "version_1" / "heat_exchanger_dataset.csv"
BACKUP = SRC.with_suffix(f".csv.bak_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")

print(f"Loading {SRC}")
df = pd.read_csv(SRC)

# Assumed constants from DEFAULT_CONFIG
COLD_INLET_TEMPERATURE_K = 293.15
CP_COLD_KJ_KG_K = 4.18

# Ensure numeric
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception:
        pass

# Backup original
print(f"Backing up original to {BACKUP}")
SRC.rename(BACKUP)

# Compute corrected cold outlet temps where heat load & cold flow available
mask = (
    df['hx_1_heat_load_kw'].notna() &
    df['cold_inlet_mass_flow_kg_s'].notna() &
    (df['cold_inlet_mass_flow_kg_s'] != 0)
)

print(f"Rows with valid heat load and cold flow: {mask.sum()}")

# Compute cold outlet
df.loc[mask, 'cold_outlet_temperature_k'] = (
    COLD_INLET_TEMPERATURE_K + df.loc[mask, 'hx_1_heat_load_kw']
    / (df.loc[mask, 'cold_inlet_mass_flow_kg_s'] * CP_COLD_KJ_KG_K)
)

# Noisy variant
mask_noisy = (
    df['hx_1_heat_load_kw_noisy'].notna() &
    df['cold_inlet_mass_flow_kg_s_noisy'].notna() &
    (df['cold_inlet_mass_flow_kg_s_noisy'] != 0)
)
print(f"Rows with valid noisy heat load and noisy cold flow: {mask_noisy.sum()}")

df.loc[mask_noisy, 'cold_outlet_temperature_k_noisy'] = (
    COLD_INLET_TEMPERATURE_K + df.loc[mask_noisy, 'hx_1_heat_load_kw_noisy']
    / (df.loc[mask_noisy, 'cold_inlet_mass_flow_kg_s_noisy'] * CP_COLD_KJ_KG_K)
)

# Fill any remaining NaNs in cold_outlet with previous value or cold inlet
filled = 0
for col in ['cold_outlet_temperature_k', 'cold_outlet_temperature_k_noisy']:
    n_before = df[col].isna().sum()
    df[col] = df[col].fillna(COLD_INLET_TEMPERATURE_K)
    n_after = df[col].isna().sum()
    filled += (n_before - n_after)

print(f"Filled {filled} missing cold outlet values with default inlet temp {COLD_INLET_TEMPERATURE_K}")

# Save corrected file
print(f"Saving corrected dataset to {SRC}")
df.to_csv(SRC, index=False)
print("Done.")
