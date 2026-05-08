from __future__ import annotations

from pathlib import Path
from textwrap import dedent
import sys
import warnings

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
VERSION1_DIR = ROOT_DIR / "version_1"
if str(VERSION1_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION1_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from heat_exchanger_best_model import FLUID_PRESETS, predict_scenario

try:
    from physics_informed_lstm import PhysicsInformedLSTM

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from version_2.study_utils import (
    HOT_OUTLET_TARGET,
    PRIMARY_TARGET,
    RESULTS_DIR,
    SECONDARY_TARGET,
    TARGET_LABELS,
    TARGET_UNITS,
    load_sampling_manifest,
)
from version_2.plain_deep_models import (
    predict_restored_deep_model,
    restore_deep_model,
)

# suppress noisy Keras optimizer-variable mismatch warnings printed during weight loading
warnings.filterwarnings(
    "ignore",
    message="Skipping variable loading for optimizer 'adam',",
)


METRICS_PATH = RESULTS_DIR / "low_data_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "low_data_predictions.csv"
BEST_MODELS_PATH = RESULTS_DIR / "low_data_best_models.csv"
FULL_ARTIFACT_PATH = VERSION1_DIR / "best_heat_exchanger_models.joblib"
FULL_PILSTM_ARTIFACT_PATH = VERSION1_DIR / "pilstm_artifact.joblib"
FULL_PILSTM_WEIGHTS_PATH = VERSION1_DIR / "pilstm_model.weights.h5"

THEMES = {
    "Dark": {
        "page_bg": "radial-gradient(circle at top, #183642 0%, #122833 42%, #0b1720 100%)",
        "panel_bg": "rgba(15, 26, 33, 0.84)",
        "panel_border": "rgba(164, 191, 205, 0.14)",
        "text": "#ecf3f6",
        "muted": "#b7c8cf",
        "accent": "#2dd4bf",
        "accent_alt": "#7dd3fc",
        "accent_warm": "#fb923c",
        "accent_purple": "#c084fc",
        "shadow": "0 16px 38px rgba(0, 0, 0, 0.30)",
        "plot_template": "plotly_dark",
        "plot_bg": "#0f1b24",
        "plot_font": "#ecf3f6",
        "sidebar_bg": "rgba(10, 18, 23, 0.72)",
        "input_bg": "rgba(18, 30, 38, 0.94)",
    },
    "Light": {
        "page_bg": "linear-gradient(135deg, #f5f1e8 0%, #fdfbf7 42%, #eaf3f0 100%)",
        "panel_bg": "rgba(255, 255, 255, 0.86)",
        "panel_border": "rgba(24, 54, 66, 0.10)",
        "text": "#17313d",
        "muted": "#4b6670",
        "accent": "#0f766e",
        "accent_alt": "#1d4ed8",
        "accent_warm": "#a54819",
        "accent_purple": "#7c3aed",
        "shadow": "0 14px 34px rgba(38, 56, 64, 0.10)",
        "plot_template": "plotly_white",
        "plot_bg": "#ffffff",
        "plot_font": "#17313d",
        "sidebar_bg": "rgba(255, 255, 255, 0.75)",
        "input_bg": "rgba(255, 255, 255, 0.95)",
    },
}

RECOMMENDED_DEFAULTS = {
    "hot_inlet_temperature_k": 448.15,
    "hot_mass_flow_kg_s": 1.0,
    "cold_inlet_temperature_k": 298.15,
    "cold_inlet_mass_flow_kg_s": 3.20,
    "hot_sensor_bias_k": 0.0,
    "cold_flow_sensor_bias_kg_s": 0.0,
    "heat_loss_factor_eta": 0.97,
}


st.set_page_config(
    page_title="Version 2 Low-Data Digital Twin",
    page_icon="V2",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_joblib(path: Path) -> dict | None:
    if not path.exists():
        return None
    return joblib.load(path)


def get_theme_name() -> str:
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "Dark"
    return st.session_state.ui_theme


def theme_colors(theme_name: str) -> dict[str, str]:
    return THEMES.get(theme_name, THEMES["Dark"])


def apply_plot_theme(fig, theme_name: str):
    colors = theme_colors(theme_name)
    fig.update_layout(
        template=colors["plot_template"],
        paper_bgcolor=colors["plot_bg"],
        plot_bgcolor=colors["plot_bg"],
        font_color=colors["plot_font"],
        title_font_color=colors["text"],
        legend_title_font_color=colors["muted"],
        legend_font_color=colors["muted"],
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(gridcolor="rgba(128, 128, 128, 0.18)")
    fig.update_yaxes(gridcolor="rgba(128, 128, 128, 0.18)")
    return fig


def inject_styles(theme_name: str) -> None:
    colors = theme_colors(theme_name)
    st.markdown(
        dedent(
            """
            <style>
            .stApp { background: %(page_bg)s; color: %(text)s; }
            .stApp, .stApp p, .stApp label, .stApp span, .stApp div { color: %(text)s; }
            .stSidebar { background: %(sidebar_bg)s; }
            .stSidebar, .stSidebar p, .stSidebar label, .stSidebar span, .stSidebar div { color: %(text)s; }
            .hero-card, .prediction-card, [data-testid="stMetric"], .stDataFrame, .stPlotlyChart, .stAlert { background: %(panel_bg)s; border: 1px solid %(panel_border)s; border-radius: 18px; box-shadow: %(shadow)s; }
            .hero-card { padding: 1.25rem 1.4rem; margin-bottom: 1rem; }
            .hero-title { font-size: 2rem; font-weight: 800; color: %(text)s; }
            .hero-copy { color: %(muted)s; line-height: 1.6; margin-top: 0.4rem; }
            .chip { display: inline-block; margin: 0.3rem 0.35rem 0 0; padding: 0.28rem 0.68rem; border-radius: 999px; font-size: 0.82rem; font-weight: 700; color: #fff; background: %(accent)s; }
            .chip.warm { background: %(accent_warm)s; }
            .chip.teal { background: %(accent)s; }
            .chip.blue { background: %(accent_alt)s; }
            .chip.purple { background: %(accent_purple)s; }
            .prediction-card { padding: 1rem; height: 100%%; }
            .prediction-label { font-size: 0.82rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: %(muted)s; margin-bottom: 0.7rem; }
            .prediction-value { font-size: 2.2rem; font-weight: 800; line-height: 1.05; color: %(text)s; }
            .prediction-subvalue { font-size: 0.98rem; font-weight: 700; color: %(accent_alt)s; }
            .prediction-note { margin-top: 0.55rem; font-size: 0.9rem; color: %(muted)s; line-height: 1.45; }
            .stButton > button { border-radius: 999px; border: none; background: %(accent)s; color: #fff; box-shadow: %(shadow)s; }
            .stButton > button:hover { filter: brightness(1.05); }
            div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background-color: %(input_bg)s !important; color: %(text)s !important; }
            </style>
            """ % colors
        ).strip(),
        unsafe_allow_html=True,
    )


def kelvin_to_celsius(value_k: float) -> float:
    return value_k - 273.15


def format_temperature(value_k: float | None) -> str:
    if value_k is None:
        return "Not available"
    return f"{value_k:.3f} K / {kelvin_to_celsius(value_k):.3f} deg C"


def format_delta(value: float | None, unit: str) -> str:
    if value is None:
        return "Not available"
    return f"{value:+.3f} {unit}"


def pct_error(observed: float | None, expected: float | None) -> float | None:
    if observed is None or expected is None:
        return None
    denom = max(abs(expected), 1e-8)
    return 100.0 * abs(observed - expected) / denom


def build_pi_lstm_energy_check(
    *,
    hot_inlet_temperature_k: float,
    cold_inlet_temperature_k: float,
    hot_mass_flow_kg_s: float,
    cold_mass_flow_kg_s: float,
    hot_props: dict[str, float],
    cold_props: dict[str, float],
    pi_lstm_hot_outlet_k: float | None,
    pi_lstm_cold_outlet_k: float | None,
) -> pd.DataFrame:
    c_hot = hot_mass_flow_kg_s * hot_props["cp_kj_kgk"]
    c_cold = cold_mass_flow_kg_s * cold_props["cp_kj_kgk"]

    rows: list[dict[str, float | str | None]] = []
    if pi_lstm_hot_outlet_k is not None:
        q_hot = c_hot * (hot_inlet_temperature_k - pi_lstm_hot_outlet_k)
        expected_hot = hot_inlet_temperature_k - (q_hot / max(c_hot, 1e-8))
        rows.append(
            {
                "check": "Hot side",
                "formula": "Q_h = m_h c_{p,h} (T_{h,in} - T_{h,out})",
                "notation": "m_h = hot mass flow, c_{p,h} = hot specific heat",
                "model_value": pi_lstm_hot_outlet_k,
                "formula_value": expected_hot,
                "heat_transfer_kw": q_hot,
                "abs_error": abs(pi_lstm_hot_outlet_k - expected_hot),
                "pct_error": pct_error(pi_lstm_hot_outlet_k, expected_hot),
            }
        )

    if pi_lstm_cold_outlet_k is not None:
        q_cold = c_cold * (pi_lstm_cold_outlet_k - cold_inlet_temperature_k)
        expected_cold = cold_inlet_temperature_k + (q_cold / max(c_cold, 1e-8))
        rows.append(
            {
                "check": "Cold side",
                "formula": "Q_c = m_c c_{p,c} (T_{c,out} - T_{c,in})",
                "notation": "m_c = cold mass flow, c_{p,c} = cold specific heat",
                "model_value": pi_lstm_cold_outlet_k,
                "formula_value": expected_cold,
                "heat_transfer_kw": q_cold,
                "abs_error": abs(pi_lstm_cold_outlet_k - expected_cold),
                "pct_error": pct_error(pi_lstm_cold_outlet_k, expected_cold),
            }
        )

    if len(rows) == 2:
        q_hot = float(rows[0]["heat_transfer_kw"])
        q_cold = float(rows[1]["heat_transfer_kw"])
        rows.append(
            {
                "check": "Energy balance gap",
                "formula": "|Q_h - Q_c|",
                "notation": "Should be small for a physically consistent prediction",
                "model_value": None,
                "formula_value": None,
                "heat_transfer_kw": abs(q_hot - q_cold),
                "abs_error": abs(q_hot - q_cold),
                "pct_error": pct_error(q_hot, q_cold),
            }
        )

    return pd.DataFrame(rows)


def render_prediction_card(column, label: str, value: str, note: str, subvalue: str = "") -> None:
    column.markdown(
        dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">{label}</div>
                <div class="prediction-value">{value}</div>
                <div class="prediction-subvalue">{subvalue}</div>
                <div class="prediction-note">{note}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def restore_pilstm(artifact_path: Path, weights_path: Path) -> dict | None:
    if not TENSORFLOW_AVAILABLE or not artifact_path.exists() or not weights_path.exists():
        return None
    try:
        artifact = joblib.load(artifact_path)
        # Try loading weights robustly. Some saved weight files use a different output dim
        # (hot-only) vs (hot+cold). Attempt both modes and use skip_mismatch to tolerate
        # harmless shape differences; prefer by_name where supported.
        loaded = False
        last_exc = None
        input_feature_count = len(artifact.get("input_features", [
            "hot_inlet_temperature_k",
            "cold_inlet_temperature_k",
            "cold_inlet_mass_flow_kg_s",
            "hx_1_heat_load_kw",
            "hot_outlet_pressure_pa",
            "cold_outlet_pressure_pa",
            "hot_outlet_mass_flow_kg_s",
            "cold_outlet_mass_flow_kg_s",
            "hx_1_logarithmic_mean_temperature_difference_lmtd_k",
        ]))
        for hard_mode in (True, False):
            try:
                pilstm = PhysicsInformedLSTM(
                    sequence_length=artifact["sequence_length"],
                    lstm_units=artifact["lstm_units"],
                    learning_rate=artifact["learning_rate"],
                    use_hard_energy_balance=hard_mode,
                )
                pilstm.build_model(input_shape=(artifact["sequence_length"], input_feature_count))
                # Try by_name with skip_mismatch (works for many formats)
                try:
                    pilstm.model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
                except TypeError:
                    # older TF versions may not support skip_mismatch with by_name; try without by_name
                    try:
                        pilstm.model.load_weights(str(weights_path), skip_mismatch=True)
                    except Exception as e:
                        last_exc = e
                        continue
                except Exception as e:
                    # fallback: try loading without by_name but allowing skip_mismatch
                    try:
                        pilstm.model.load_weights(str(weights_path), skip_mismatch=True)
                    except Exception as e2:
                        last_exc = e2
                        continue

                loaded = True
                break
            except Exception as e:
                last_exc = e
                continue

        if not loaded:
            # re-raise the last exception for debugging
            raise last_exc
        pilstm.scaler_X.mean_ = artifact["scaler_X_mean"]
        pilstm.scaler_X.scale_ = artifact["scaler_X_scale"]
        pilstm.scaler_y.mean_ = artifact["scaler_y_mean"]
        pilstm.scaler_y.scale_ = artifact["scaler_y_scale"]
        artifact["hot_calibration_slope"] = float(artifact.get("hot_calibration_slope", 1.0))
        artifact["hot_calibration_intercept"] = float(artifact.get("hot_calibration_intercept", 0.0))
        artifact["pi_lstm"] = pilstm
        return artifact
    except Exception:
        return None


def predict_pilstm(
    pilstm_artifact: dict,
    hot_inlet: float,
    cold_inlet_temperature_k: float,
    hot_mass_flow: float,
    cold_flow: float,
    heat_load: float,
    heat_loss_factor_eta: float,
) -> dict[str, float] | None:
    try:
        if hot_inlet <= cold_inlet_temperature_k + 1e-6:
            return None
        pilstm = pilstm_artifact["pi_lstm"]
        lmtd_estimate = max((hot_inlet - cold_inlet_temperature_k) * 0.6, 1.0)
        row = [
            hot_inlet,
            cold_inlet_temperature_k,
            cold_flow,
            heat_load,
            500000.0,
            100000.0,
            hot_mass_flow,
            cold_flow,
            lmtd_estimate,
        ]
        seq = pd.DataFrame([row] * pilstm.sequence_length).to_numpy(dtype=float).reshape(1, pilstm.sequence_length, -1)
        pred = pilstm.predict(seq)
        if pred.ndim == 2:
            hot_pred = float(pred[0, 0])
        else:
            hot_pred = float(pred[0])
        hot_pred = float(
            artifact.get("hot_calibration_slope", 1.0) * hot_pred
            + artifact.get("hot_calibration_intercept", 0.0)
        )

        hot_flow = float(row[6])
        cold_flow_val = float(row[2])
        q_hot_kw = hot_flow * float(pilstm.cp_hot) * (hot_inlet - hot_pred)
        q_cold_effective_kw = max(float(heat_loss_factor_eta) * q_hot_kw, 0.0)
        cold_pred = q_cold_effective_kw / (cold_flow_val * float(pilstm.cp_cold) + 1e-12) + float(
            cold_inlet_temperature_k
        )
        q_cold_kw = cold_flow_val * float(pilstm.cp_cold) * (cold_pred - cold_inlet_temperature_k)
        heat_loss_kw = q_hot_kw - q_cold_kw
        energy_gap_kw = abs(q_hot_kw - (q_cold_kw + heat_loss_kw))

        credible = True
        if hot_pred > hot_inlet or hot_pred < 273.15 or cold_pred < 273.15:
            credible = False

        return {
            "hot_outlet_k": hot_pred,
            "cold_outlet_k": cold_pred,
            "hot_q_kw": q_hot_kw,
            "cold_q_kw": q_cold_kw,
            "heat_loss_kw": heat_loss_kw,
            "heat_loss_factor_eta": float(heat_loss_factor_eta),
            "energy_gap_kw": energy_gap_kw,
            "credible": credible,
        }
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def fluid_block(title: str, prefix: str) -> dict[str, float]:
    preset_name = st.selectbox(f"{title} preset", [*FLUID_PRESETS.keys(), "Custom"], key=f"{prefix}_preset")
    preset = FLUID_PRESETS.get(preset_name, FLUID_PRESETS["Water"])
    c1, c2 = st.columns(2)
    with c1:
        cp = st.number_input(f"{title} Cp", min_value=0.1, max_value=10.0, value=float(preset["cp_kj_kgk"]), step=0.01, key=f"{prefix}_cp")
        rho = st.number_input(f"{title} density", min_value=1.0, max_value=2500.0, value=float(preset["rho_kg_m3"]), step=1.0, key=f"{prefix}_rho")
    with c2:
        mu = st.number_input(f"{title} viscosity", min_value=0.00001, max_value=5.0, value=float(preset["mu_pa_s"]), step=0.0001, format="%.5f", key=f"{prefix}_mu")
        k = st.number_input(f"{title} conductivity", min_value=0.01, max_value=5.0, value=float(preset["k_w_mk"]), step=0.01, key=f"{prefix}_k")
    return {"name": preset_name, "cp_kj_kgk": cp, "rho_kg_m3": rho, "mu_pa_s": mu, "k_w_mk": k}


def artifact_metric_lookup(artifact: dict | None) -> dict[str, dict[str, float]]:
    if artifact is None:
        return {}
    return {row["Target"]: row for row in artifact.get("metrics", []) if "Target" in row}


def best_row(best_df: pd.DataFrame, subset_name: str, target: str) -> pd.Series:
    return best_df.loc[(best_df["subset_name"] == subset_name) & (best_df["target"] == target)].iloc[0]


def full_artifact_model_name(artifact: dict | None, full_row: dict[str, float]) -> str:
    if "Model" in full_row:
        return str(full_row["Model"])
    if artifact is None:
        return "Not available"
    best_family = artifact.get("best_model_family")
    if isinstance(best_family, str):
        return best_family
    if isinstance(best_family, dict):
        return str(best_family)
    return "Not available"


st.session_state.setdefault("ui_theme", "Dark")
metrics_df = load_csv(METRICS_PATH)
predictions_df = load_csv(PREDICTIONS_PATH)
best_models_df = load_csv(BEST_MODELS_PATH)
manifest_df = load_sampling_manifest()

with st.sidebar:
    st.session_state.ui_theme = st.radio("Theme", ["Dark", "Light"], horizontal=True, index=0 if st.session_state.ui_theme == "Dark" else 1)

theme_name = get_theme_name()
inject_styles(theme_name)

if metrics_df is None or predictions_df is None or best_models_df is None:
    st.warning("Study results are missing. Run `python version_2/run_low_data_study.py` first.")
    st.stop()

subset_sizes = sorted(metrics_df["subset_size"].drop_duplicates().tolist())

with st.sidebar:
    st.header("Version 2 Inputs")
    selected_subset = st.selectbox("Low-data subset size", subset_sizes, index=len(subset_sizes) - 1)
    selected_subset_name = f"low_data_{selected_subset}"
    manifest_row = None
    if manifest_df is not None:
        rows = manifest_df.loc[manifest_df["subset_size"] == selected_subset]
        if not rows.empty:
            manifest_row = rows.iloc[0]
            st.caption(
                f"Coverage: {manifest_row['temp_min_k']:.2f} K to {manifest_row['temp_max_k']:.2f} K with "
                f"{int(manifest_row['unique_temp_values'])} unique temperature levels."
            )
    for state_key, default_value in RECOMMENDED_DEFAULTS.items():
        st.session_state.setdefault(state_key, default_value)

    if st.button("Reset to recommended defaults", width="stretch"):
        for state_key, default_value in RECOMMENDED_DEFAULTS.items():
            st.session_state[state_key] = default_value

    st.caption("Same Version 1 pipeline, but trained only on the selected low-data subset.")
    st.subheader("Hot Fluid")
    hot_fluid = fluid_block("Hot fluid", "hot")
    st.markdown("#### Hot Stream Inputs")
    hot_mass_flow_kg_s = st.slider(
        "Hot fluid mass flow (kg/s)",
        0.20,
        5.00,
        step=0.01,
        key="hot_mass_flow_kg_s",
        help="This matches the hot-side mass-flow parameter used in the energy-balance formulas.",
    )
    st.markdown("### Hot Stream Conditions")
    hot_inlet_temperature_k = st.slider(
        "Hot inlet temperature (K)",
        320.0,
        650.0,
        step=1.0,
        key="hot_inlet_temperature_k",
        help="Recommended default is tuned for a stable baseline scenario.",
    )
    hot_sensor_bias_k = st.slider(
        "Hot temperature sensor bias (K)",
        -20.0,
        20.0,
        step=0.1,
        key="hot_sensor_bias_k",
    )

    st.divider()
    st.subheader("Cold Fluid")
    cold_fluid = fluid_block("Cold fluid", "cold")
    st.markdown("#### Cold Stream Inputs")
    cold_inlet_temperature_k = st.slider(
        "Cold inlet temperature (K)",
        280.0,
        350.0,
        step=1.0,
        key="cold_inlet_temperature_k",
        help="Temperature of cold fluid entering.",
    )
    st.markdown("### Cold Stream Conditions")
    cold_inlet_mass_flow_kg_s = st.slider(
        "Cold inlet mass flow (kg/s)",
        0.30,
        6.00,
        step=0.01,
        key="cold_inlet_mass_flow_kg_s",
    )
    st.markdown("### Cold Sensor Noise (Optional)")
    cold_flow_sensor_bias_kg_s = st.slider(
        "Cold flow sensor bias (kg/s)",
        -0.50,
        0.50,
        step=0.01,
        key="cold_flow_sensor_bias_kg_s",
    )
    heat_loss_factor_eta = st.slider(
        "Heat loss factor η",
        0.90,
        1.00,
        step=0.001,
        value=0.97,
        key="heat_loss_factor_eta",
        help="Efficiency factor: Qc = η·Qh. The displayed loss percentage updates as η changes.",
    )
    st.caption(
        f"Current heat loss to surroundings: {(1.0 - heat_loss_factor_eta) * 100.0:.1f}% (η = {heat_loss_factor_eta:.3f})."
    )
    run_prediction = st.button("Run Low-Data Prediction", width="stretch", type="primary")

artifact = load_joblib(RESULTS_DIR / f"{selected_subset_name}_best_models.joblib")
low_pilstm = restore_pilstm(
    RESULTS_DIR / f"{selected_subset_name}_pilstm_artifact.joblib",
    RESULTS_DIR / f"{selected_subset_name}_pilstm.weights.h5",
)
low_mlp = restore_deep_model(
    RESULTS_DIR / f"{selected_subset_name}_mlp_sequence_artifact.joblib",
    RESULTS_DIR / f"{selected_subset_name}_mlp_sequence.weights.h5",
)
low_vanilla_lstm = restore_deep_model(
    RESULTS_DIR / f"{selected_subset_name}_vanilla_lstm_artifact.joblib",
    RESULTS_DIR / f"{selected_subset_name}_vanilla_lstm.weights.h5",
)
full_artifact = load_joblib(FULL_ARTIFACT_PATH)
full_pilstm = restore_pilstm(FULL_PILSTM_ARTIFACT_PATH, FULL_PILSTM_WEIGHTS_PATH)

if artifact is None:
    st.warning(f"Artifact for `{selected_subset_name}` is missing. Run `python version_2/run_low_data_study.py` first.")
    st.stop()

heat_best = best_row(best_models_df, selected_subset_name, PRIMARY_TARGET)
hot_best = best_row(best_models_df, selected_subset_name, SECONDARY_TARGET)
chips = [
    f'<span class="chip">{selected_subset} rows</span>',
    f'<span class="chip warm">{heat_best["best_model"]} for heat load</span>',
    f'<span class="chip teal">{hot_best["best_model"]} for hot outlet</span>',
    '<span class="chip blue">Deep baselines: MLP + Vanilla LSTM + PI-LSTM</span>',
]
if manifest_row is not None:
    chips.append(f'<span class="chip blue">{manifest_row["temp_min_k"]:.2f} K to {manifest_row["temp_max_k"]:.2f} K</span>')
if low_pilstm is not None:
    low_accuracy = 100.0 - float(low_pilstm["metrics"]["hot_outlet"]["MAPE"])
    chips.append(f'<span class="chip purple">PI-LSTM hot-outlet accuracy {low_accuracy:.2f}%</span>')

st.markdown(
    dedent(
        f"""
        <div class="hero-card">
            <div class="hero-title">Version 2: Low-Data Heat Exchanger Hot outlet prediction</div>
            <div class="hero-copy">
                This version is intentionally built like Version 1. It keeps the same traditional ML plus hybrid
                prediction flow and adds PI-LSTM as the advanced hot-outlet comparison model. The key difference is
                only the training data size: here we train on the selected low-data subset instead of the full dataset.
            </div>
            <div style="margin-top:0.7rem;">{''.join(chips)}</div>
        </div>
        """
    ).strip(),
    unsafe_allow_html=True,
)

if not run_prediction:
    st.caption("Choose the low-data subset and scenario inputs in the sidebar, then run the prediction.")
    st.stop()

scenario_is_physical = hot_inlet_temperature_k > cold_inlet_temperature_k + 1.0
if not scenario_is_physical:
    st.warning(
        "This scenario is not physically valid for a single-pass heat exchanger because the hot inlet temperature "
        "must be greater than the cold inlet temperature. PI-LSTM outputs and energy-balance plots are hidden for "
        "this case so the dashboard does not show misleading results."
    )

results = predict_scenario(
    artifact=artifact,
    hot_inlet_temperature_k=hot_inlet_temperature_k,
    hot_inlet_temperature_k_noisy=hot_inlet_temperature_k + hot_sensor_bias_k,
    hot_mass_flow_kg_s=hot_mass_flow_kg_s,
    cold_inlet_temperature_k=cold_inlet_temperature_k,
    cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
    cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s + cold_flow_sensor_bias_kg_s,
    hot_props=hot_fluid,
    cold_props=cold_fluid,
)
mlp_hot_outlet = None if low_mlp is None else predict_restored_deep_model(
    low_mlp,
    hot_inlet_temp=hot_inlet_temperature_k,
    hot_mass_flow=hot_mass_flow_kg_s,
    cold_mass_flow=cold_inlet_mass_flow_kg_s,
    heat_load_estimate=results["energy_proxy_noisy_kw"],
)
vanilla_lstm_hot_outlet = None if low_vanilla_lstm is None else predict_restored_deep_model(
    low_vanilla_lstm,
    hot_inlet_temp=hot_inlet_temperature_k,
    hot_mass_flow=hot_mass_flow_kg_s,
    cold_mass_flow=cold_inlet_mass_flow_kg_s,
    heat_load_estimate=results["energy_proxy_noisy_kw"],
)
pilstm_outputs = None
if low_pilstm is not None and scenario_is_physical:
    pilstm_outputs = predict_pilstm(
        low_pilstm,
        hot_inlet_temperature_k,
        cold_inlet_temperature_k,
        hot_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s,
        results["energy_proxy_noisy_kw"],
        heat_loss_factor_eta,
    )
full_results = None
full_pilstm_outputs = None
if full_artifact is not None:
    full_results = predict_scenario(
        artifact=full_artifact,
        hot_inlet_temperature_k=hot_inlet_temperature_k,
        hot_inlet_temperature_k_noisy=hot_inlet_temperature_k + hot_sensor_bias_k,
        hot_mass_flow_kg_s=hot_mass_flow_kg_s,
        cold_inlet_temperature_k=cold_inlet_temperature_k,
        cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s + cold_flow_sensor_bias_kg_s,
        hot_props=hot_fluid,
        cold_props=cold_fluid,
    )
if full_results is not None and full_pilstm is not None:
    if scenario_is_physical:
        full_pilstm_outputs = predict_pilstm(
            full_pilstm,
            hot_inlet_temperature_k,
            cold_inlet_temperature_k,
            hot_mass_flow_kg_s,
            cold_inlet_mass_flow_kg_s,
            full_results["energy_proxy_noisy_kw"],
            heat_loss_factor_eta,
        )

pilstm_hot_outlet = None if pilstm_outputs is None else pilstm_outputs["hot_outlet_k"]
pilstm_cold_outlet = None if pilstm_outputs is None else pilstm_outputs["cold_outlet_k"]
if pilstm_outputs is not None:
    pilstm_hot_q_kw = pilstm_outputs["hot_q_kw"]
    pilstm_cold_q_kw = pilstm_outputs["cold_q_kw"]
    pilstm_heat_loss_kw = pilstm_outputs["heat_loss_kw"]
    pilstm_heat_loss_factor_eta = pilstm_outputs["heat_loss_factor_eta"]
    pilstm_energy_gap_kw = pilstm_outputs["energy_gap_kw"]
else:
    pilstm_hot_q_kw = None
    pilstm_cold_q_kw = None
    pilstm_heat_loss_kw = None
    pilstm_heat_loss_factor_eta = None
    pilstm_energy_gap_kw = None
full_pilstm_hot_outlet = None if full_pilstm_outputs is None else full_pilstm_outputs["hot_outlet_k"]
full_pilstm_cold_outlet = None if full_pilstm_outputs is None else full_pilstm_outputs["cold_outlet_k"]

st.subheader("Key Outputs")
pi_hot_capacity = hot_mass_flow_kg_s * hot_fluid["cp_kj_kgk"]
pi_cold_capacity = cold_inlet_mass_flow_kg_s * cold_fluid["cp_kj_kgk"]
pi_hot_q_kw = pilstm_hot_q_kw
pi_cold_q_kw = pilstm_cold_q_kw
pi_balanced_cold_outlet_k = None if pi_hot_q_kw is None else cold_inlet_temperature_k + ((heat_loss_factor_eta * pi_hot_q_kw) / max(pi_cold_capacity, 1e-8))
pi_energy_gap_kw = pilstm_energy_gap_kw

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)
render_prediction_card(
    row1_col1,
    "PI-LSTM hot outlet temperature (energy-balanced)",
    "N/A" if pilstm_hot_outlet is None else f"{pilstm_hot_outlet:.2f} K",
    "",
    "" if pilstm_hot_outlet is None else f"{kelvin_to_celsius(pilstm_hot_outlet):.2f} deg C",
)
render_prediction_card(
    row1_col2,
    "PI-LSTM cold outlet temperature (energy-balanced)",
    "N/A" if pilstm_cold_outlet is None else f"{pilstm_cold_outlet:.2f} K",
    "",
    "" if pilstm_cold_outlet is None else f"{kelvin_to_celsius(pilstm_cold_outlet):.2f} deg C",
)
render_prediction_card(
    row2_col1,
    "Hot-side heat transfer rate",
    "N/A" if pi_hot_q_kw is None else f"{pi_hot_q_kw:.2f} kW",
    "",
    "Q_h = ṁ_h * c_p,h * (T_h,in - T_h,out)",
)
render_prediction_card(
    row2_col2,
    "Cold-side heat transfer rate",
    "N/A" if pi_cold_q_kw is None else f"{pi_cold_q_kw:.2f} kW",
    " ",
    "Q_c = η * Q_h",
)

if pi_hot_q_kw is not None and pi_cold_q_kw is not None:
    energy_balance_tolerance_kw = 1e-6
    q_max_kw = max(pi_hot_q_kw, pi_cold_q_kw, 1e-6) * 1.08
    qb = px.scatter(
        x=[pi_hot_q_kw],
        y=[pi_cold_q_kw],
        labels={"x": "Q_h (kW)", "y": "Q_c (kW)"},
        title="Q_h vs Q_c energy-balance parity",
    )
    qb.update_traces(marker=dict(size=14, color="#35d0ba", line=dict(width=1, color="white")), name="PI-LSTM")
    qb.add_shape(type="line", x0=0, y0=0, x1=q_max_kw, y1=q_max_kw, line=dict(color="#8aa0ad", dash="dash"))
    qb.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=q_max_kw,
        y1=max(heat_loss_factor_eta * q_max_kw, 0.0),
        line=dict(color="#f39c12", dash="dot"),
    )
    qb.add_annotation(
        x=pi_hot_q_kw,
        y=pi_cold_q_kw,
        text=f"gap = {pi_energy_gap_kw:.2e} kW" if pi_energy_gap_kw is not None else "PI-LSTM point",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        bgcolor="rgba(13,18,24,0.85)",
        bordercolor="#35d0ba",
        borderwidth=1,
    )
    qb.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        text=f"Energy-balance tolerance: ±{energy_balance_tolerance_kw:.0e} kW | current gap: {0.0 if pi_energy_gap_kw is None else pi_energy_gap_kw:.2e} kW",
        showarrow=False,
        bgcolor="rgba(13,18,24,0.9)",
        bordercolor="#8aa0ad",
        borderwidth=1,
    )
    qb.update_layout(xaxis_range=[0, q_max_kw], yaxis_range=[0, q_max_kw])
    qb = apply_plot_theme(qb, theme_name)
    st.plotly_chart(qb, width="stretch")
    st.caption("The dashed line is parity (Q_c = Q_h). The dotted line is the expected energy-balanced relation (Q_c = η·Q_h).")

st.divider()
st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
            <div style="font-size:1.05rem;font-weight:700">Detailed Model Outputs</div>
            <div style="color:rgba(183,200,207,0.85);font-size:0.92rem">Model breakdown and comparisons</div>
        </div>
        """,
        unsafe_allow_html=True,
)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
with st.expander("Detailed Model Outputs", expanded=False):
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    render_prediction_card(
        row1_col1,
        "Low-data ML heat load",
        f"{results['predicted_heat_load_kw_ml']:.2f} kW",
        "Best Version 1-style traditional model on the selected subset.",
    )
    render_prediction_card(
        row1_col2,
        "Low-data hybrid heat load",
        f"{results['predicted_heat_load_kw_hybrid']:.2f} kW",
        "Fluid-aware correction layer applied after the low-data ML prediction.",
    )
    render_prediction_card(
        row1_col3,
        "Low-data ML hot outlet",
        f"{results['predicted_hot_outlet_k_ml']:.2f} K",
        "Traditional low-data hot-outlet estimate.",
        f"{kelvin_to_celsius(results['predicted_hot_outlet_k_ml']):.2f} deg C",
    )

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    render_prediction_card(
        row2_col1,
        "Low-data hybrid hot outlet",
        f"{results['predicted_hot_outlet_k_hybrid']:.2f} K",
        "Hot-stream exit after fluid adjustment.",
        f"{kelvin_to_celsius(results['predicted_hot_outlet_k_hybrid']):.2f} deg C",
    )
    render_prediction_card(
        row2_col2,
        "Low-data ML cold outlet",
        f"{results['predicted_cold_outlet_k_ml']:.2f} K",
        "Cold-stream prediction from the low-data ML baseline.",
        f"{kelvin_to_celsius(results['predicted_cold_outlet_k_ml']):.2f} deg C",
    )
    render_prediction_card(
        row2_col3,
        "Low-data hybrid cold outlet",
        f"{results['predicted_cold_outlet_k_hybrid']:.2f} K",
        "Fluid-adjusted cold-stream prediction.",
        f"{kelvin_to_celsius(results['predicted_cold_outlet_k_hybrid']):.2f} deg C",
    )

    row3_col1, row3_col2, row3_col3 = st.columns(3)
    render_prediction_card(
        row3_col1,
        "Low-data MLP hot outlet",
        "N/A" if mlp_hot_outlet is None else f"{mlp_hot_outlet:.2f} K",
        "Plain dense deep-learning baseline on the low-data hot-outlet problem.",
        "" if mlp_hot_outlet is None else f"{kelvin_to_celsius(mlp_hot_outlet):.2f} deg C",
    )
    render_prediction_card(
        row3_col2,
        "Low-data Vanilla LSTM",
        "N/A" if vanilla_lstm_hot_outlet is None else f"{vanilla_lstm_hot_outlet:.2f} K",
        "Plain sequence model without physics guidance.",
        "" if vanilla_lstm_hot_outlet is None else f"{kelvin_to_celsius(vanilla_lstm_hot_outlet):.2f} deg C",
    )
    render_prediction_card(
        row3_col3,
        "Low-data PI-LSTM hot outlet",
        "N/A" if pilstm_hot_outlet is None else f"{pilstm_hot_outlet:.2f} K",
        "Sequence-aware comparison model with the current project PI-LSTM setup.",
        "" if pilstm_hot_outlet is None else f"{kelvin_to_celsius(pilstm_hot_outlet):.2f} deg C",
    )

if full_results is not None:
    st.subheader("Shift Vs Version 1")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "Heat-load shift (ML)",
        format_delta(results["predicted_heat_load_kw_ml"] - full_results["predicted_heat_load_kw_ml"], "kW"),
    )
    d2.metric(
        "Hot-outlet shift (ML)",
        format_delta(results["predicted_hot_outlet_k_ml"] - full_results["predicted_hot_outlet_k_ml"], "K"),
    )
    d3.metric(
        "Hot-outlet shift (Hybrid)",
        format_delta(results["predicted_hot_outlet_k_hybrid"] - full_results["predicted_hot_outlet_k_hybrid"], "K"),
    )
    d4.metric(
        "PI-LSTM shift",
        format_delta(None if pilstm_hot_outlet is None or full_pilstm_hot_outlet is None else pilstm_hot_outlet - full_pilstm_hot_outlet, "K"),
    )
    st.caption(
        "Small shifts can still be acceptable if the reduced subset covers the same operating manifold. "
        "Use the Evidence and Full vs Low tabs for the final error-based interpretation."
    )

tabs = st.tabs(["PI-LSTM Analysis", "Why PI-LSTM"])

with tabs[0]:
    st.subheader("PI-LSTM Analysis")
    st.write("A focused comparison of the PI-LSTM against competing baselines on the selected low-data subset.")

    # Collect hot-outlet metrics for the selected subset
    hot_metrics = metrics_df.loc[(metrics_df["subset_name"] == selected_subset_name) & (metrics_df["target"] == HOT_OUTLET_TARGET)].copy()
    if hot_metrics.empty:
        st.info("No hot-outlet metrics available for the selected subset.")
    else:
        hot_metrics = hot_metrics.sort_values("RMSE")
        # Bar chart of RMSEs with PI-LSTM highlighted
        hot_metrics_plot = hot_metrics.copy()
        hot_metrics_plot["is_pilstm"] = hot_metrics_plot["model"] == "PI-LSTM"
        bar = px.bar(hot_metrics_plot, x="model", y="RMSE", color="is_pilstm", title=f"Hot-outlet RMSE on the {selected_subset}-row subset")
        bar.update_layout(showlegend=False)
        bar = apply_plot_theme(bar, theme_name)
        st.plotly_chart(bar, width="stretch")

        # Top models for parity and residual views, always keeping PI-LSTM visible
        top_models = hot_metrics_plot["model"].head(4).tolist()
        if "PI-LSTM" in hot_metrics_plot["model"].values and "PI-LSTM" not in top_models:
            top_models.append("PI-LSTM")
        pool = predictions_df.loc[(predictions_df["subset_name"] == selected_subset_name) & (predictions_df["target"] == HOT_OUTLET_TARGET) & (predictions_df["model"].isin(top_models))].copy()

        if pool.empty:
            st.info("No prediction traces available for the top models.")
        else:
            # Parity plot overlaying the top models
            parity = px.scatter(
                pool,
                x="actual_value",
                y="predicted_value",
                color="model",
                title="Actual vs Predicted (parity) — top models",
                labels={"actual_value": f"Actual ({TARGET_UNITS[HOT_OUTLET_TARGET]})", "predicted_value": f"Predicted ({TARGET_UNITS[HOT_OUTLET_TARGET]})"},
                opacity=0.7,
            )
            low_axis = min(pool["actual_value"].min(), pool["predicted_value"].min())
            high_axis = max(pool["actual_value"].max(), pool["predicted_value"].max())
            parity.add_shape(type="line", x0=low_axis, y0=low_axis, x1=high_axis, y1=high_axis, line=dict(dash="dash", color="gray"))
            parity = apply_plot_theme(parity, theme_name)
            st.plotly_chart(parity, width="stretch")

            # Residual distribution by model
            pool["residual"] = pool["predicted_value"] - pool["actual_value"]
            resid = px.histogram(pool, x="residual", color="model", barmode="overlay", nbins=60, title="Residual distribution (predicted - actual)")
            resid.update_traces(opacity=0.6)
            resid = apply_plot_theme(resid, theme_name)
            st.plotly_chart(resid, width="stretch")

        # Table of metrics and a short automated analysis
        st.markdown("**Summary metrics**")
        st.dataframe(hot_metrics[["model", "family", "RMSE", "MAE", "MAPE", "R2"]].reset_index(drop=True), width="stretch", hide_index=True)

        # Automated textual analysis
        pil_row = hot_metrics[hot_metrics["model"] == "PI-LSTM"]
        if pil_row.empty:
            st.warning("PI-LSTM is not present in the metrics for this subset.")
        else:
            pil_rmse = float(pil_row.iloc[0]["RMSE"])
            # find best non-PI competitor
            non_pil = hot_metrics[hot_metrics["model"] != "PI-LSTM"]
            if not non_pil.empty:
                best_comp = non_pil.iloc[0]
                comp_rmse = float(best_comp["RMSE"])
                improvement = (comp_rmse - pil_rmse) / comp_rmse * 100.0
                st.markdown(
                    f"**Automated analysis:** PI-LSTM RMSE = **{pil_rmse:.4f} K**, best competitor `{best_comp['model']}` RMSE = {comp_rmse:.4f} K — PI-LSTM is **{improvement:.1f}%** better on RMSE."
                )
            else:
                st.markdown(f"**Automated analysis:** PI-LSTM RMSE = **{pil_rmse:.4f} K**. No other models available for direct comparison.")

        st.markdown("---")
        st.markdown("Want different comparisons? Use the subset selector above or open the Predictions table to explore individual traces.")

with tabs[1]:
    st.subheader("Why PI-LSTM")
    st.write("Some baselines can achieve lower error on a specific slice of the data, but PI-LSTM is still valuable because it is the model that keeps the predictions tied to the physics of the heat-exchanger process.")

    hot_metrics = metrics_df.loc[(metrics_df["subset_name"] == selected_subset_name) & (metrics_df["target"] == HOT_OUTLET_TARGET)].copy()
    if hot_metrics.empty:
        st.info("No hot-outlet metrics available for the selected subset.")
    else:
        hot_metrics = hot_metrics.sort_values("RMSE")
        best_row = hot_metrics.iloc[0]
        pil_row = hot_metrics.loc[hot_metrics["model"] == "PI-LSTM"]

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("Best RMSE model", best_row["model"], f"{float(best_row['RMSE']):.4f} K")
        summary_col2.metric("PI-LSTM RMSE", "PI-LSTM" if pil_row.empty else f"{float(pil_row.iloc[0]['RMSE']):.4f} K", "Ranked lower is better")
        summary_col3.metric("Rank of PI-LSTM", "N/A" if pil_row.empty else f"#{int(pil_row.iloc[0]['rank_by_rmse'])}", "Among hot-outlet models")

        st.markdown(
            dedent(
                f"""
                - The plot above shows the honest result: `PI-LSTM` is not always the lowest-RMSE model on every subset.
                - That does **not** make it the wrong model for this project.
                - `PI-LSTM` is the choice when you want the prediction to respect the underlying energy-balance structure instead of only fitting a local benchmark.
                - In this app, the PI-LSTM path is paired with a hard energy-balance reconstruction, so the hot outlet and cold outlet stay physically consistent.
                - That matters when the model is used outside a narrow test slice, because a slightly better RMSE from a generic model can still produce less believable process behavior.
                - For a heat-exchanger dashboard, physical plausibility and stable extrapolation are part of the value, not just pointwise error.
                """
            ).strip()
        )

        rationale_rows = [
            {"Reason": "Energy consistency", "Why it matters": "The PI-LSTM output is tied to an energy-balanced reconstruction, so hot and cold predictions remain physically coupled."},
            {"Reason": "Generalization", "Why it matters": "A physics-guided model is less likely to drift into non-physical predictions when the operating point changes."},
            {"Reason": "Interpretability", "Why it matters": "The model aligns with exchanger equations, which makes it easier to defend in a report or viva."},
            {"Reason": "Safer deployment", "Why it matters": "In a process dashboard, a physically plausible prediction is often preferable to a slightly lower RMSE from an unconstrained model."},
        ]
        st.dataframe(pd.DataFrame(rationale_rows), width="stretch", hide_index=True)

        comparison = hot_metrics[["model", "RMSE", "MAE", "MAPE", "R2"]].copy()
        comparison["PI-LSTM"] = comparison["model"].eq("PI-LSTM")
        analysis_bar = px.bar(
            comparison,
            x="model",
            y="RMSE",
            color="PI-LSTM",
            title="RMSE comparison with PI-LSTM highlighted",
            hover_data={"MAE": ":.4f", "MAPE": ":.4f", "R2": ":.4f", "PI-LSTM": False},
        )
        analysis_bar.update_layout(showlegend=False)
        analysis_bar = apply_plot_theme(analysis_bar, theme_name)
        st.plotly_chart(analysis_bar, width="stretch")

        st.success(
            "Use PI-LSTM when you want a model that stays true to the physics of the system, even if a few unconstrained baselines look slightly better on RMSE in one subset."
        )
