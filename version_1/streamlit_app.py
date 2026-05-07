from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from heat_exchanger_best_model import FLUID_PRESETS, predict_scenario

try:
    import tensorflow as tf
    from tensorflow import keras
    from physics_informed_lstm import PhysicsInformedLSTM
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


st.set_page_config(
    page_title="Heat Exchanger Digital Twin",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = APP_DIR / "best_heat_exchanger_models.joblib"
PILSTM_ARTIFACT_PATH = APP_DIR / "pilstm_artifact.joblib"


def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Artifact not found at {ARTIFACT_PATH}. "
            "Run train_best_heat_exchanger_artifact.py first."
        )
    return joblib.load(ARTIFACT_PATH)


def load_pilstm_artifact() -> dict | None:
    if not TENSORFLOW_AVAILABLE:
        return None
    if not PILSTM_ARTIFACT_PATH.exists():
        return None
    try:
        artifact = joblib.load(PILSTM_ARTIFACT_PATH)
        
        # Reconstruct the PI-LSTM model
        pi_lstm = PhysicsInformedLSTM(
            sequence_length=artifact['sequence_length'],
            lstm_units=artifact['lstm_units'],
            learning_rate=artifact['learning_rate']
        )
        
        # Rebuild the model architecture
        pi_lstm.build_model(input_shape=(artifact['sequence_length'], 8))
        
        # Load weights
        weights_path = APP_DIR / "pilstm_model.weights.h5"
        if weights_path.exists():
            pi_lstm.model.load_weights(str(weights_path))
        
        # Restore scalers
        pi_lstm.scaler_X.mean_ = artifact['scaler_X_mean']
        pi_lstm.scaler_X.scale_ = artifact['scaler_X_scale']
        pi_lstm.scaler_y.mean_ = artifact['scaler_y_mean']
        pi_lstm.scaler_y.scale_ = artifact['scaler_y_scale']
        
        # Add the reconstructed model to artifact
        artifact['pi_lstm'] = pi_lstm
        
        return artifact
    except Exception:
        return None


def predict_pilstm(
    pilstm_artifact: dict,
    hot_inlet_temp: float,
    cold_inlet_temp: float,
    cold_mass_flow: float,
    heat_load_estimate: float,
) -> float | None:
    """
    Predict hot outlet temperature using PI-LSTM model
    
    NOTE: Only predicts hot outlet temperature because the dataset has
    incorrect cold outlet temperature data (constant at 2011.51 K).
    
    Args:
        pilstm_artifact: Loaded PI-LSTM artifact
        hot_inlet_temp: Hot inlet temperature (K)
        cold_inlet_temp: Cold inlet temperature (K)
        cold_mass_flow: Cold inlet mass flow (kg/s)
        heat_load_estimate: Estimated heat load (kW)
    
    Returns:
        Hot outlet temperature in Kelvin, or None if prediction fails
    """
    if pilstm_artifact is None:
        return None
    
    try:
        pi_lstm = pilstm_artifact['pi_lstm']
        
        # Calculate realistic LMTD estimate
        lmtd_estimate = (hot_inlet_temp - cold_inlet_temp) * 0.6
        
        # Use dataset statistics for constant values
        hot_outlet_pressure = 500000.0
        cold_outlet_pressure = 100000.0
        hot_mass_flow = 1.0
        
        # Create input row
        input_row = np.array([[
            hot_inlet_temp,
            cold_mass_flow,
            heat_load_estimate,
            hot_outlet_pressure,
            cold_outlet_pressure,
            hot_mass_flow,
            cold_mass_flow,
            lmtd_estimate
        ]])
        
        # Create sequence
        input_sequence = np.repeat(input_row, pi_lstm.sequence_length, axis=0)
        input_sequence = input_sequence.reshape(1, pi_lstm.sequence_length, -1)
        
        # Predict
        predictions = pi_lstm.predict(input_sequence)
        hot_outlet = float(predictions[0, 0])
        
        # Sanity check: hot outlet should be between cold inlet and hot inlet
        if hot_outlet > hot_inlet_temp or hot_outlet < cold_inlet_temp:
            return None
        
        return hot_outlet
    except Exception:
        return None


def inject_styles() -> None:
    st.markdown(
        dedent(
            """
        <style>
        :root {
            --ink: #183642;
            --muted: #2f4a53;
            --warm: #c2410c;
            --gold: #d97706;
            --teal: #0f766e;
            --blue: #1d4ed8;
            --card: rgba(255, 255, 255, 0.82);
            --soft: #f7f2ea;
            --panel: #fffdf9;
            --border: rgba(24, 54, 66, 0.12);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(239, 177, 77, 0.22), transparent 26%),
                radial-gradient(circle at top right, rgba(29, 120, 116, 0.18), transparent 24%),
                linear-gradient(135deg, #f4efe6 0%, #f6f1e8 35%, #eaf2ef 100%);
            color: var(--ink);
        }
        .main .block-container {
            padding-top: 4.5rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        .stApp, .stApp p, .stApp span, .stApp div, .stApp li, .stApp label, .stApp small {
            color: var(--ink);
        }
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stCaption, .stAlert, .stInfo {
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(248, 244, 237, 0.98), rgba(239, 246, 243, 0.98)) !important;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] > div:first-child {
            background: transparent !important;
        }
        .stSidebar, .stSidebar * {
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown div,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption {
            color: var(--ink) !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            color: var(--ink) !important;
            font-weight: 700 !important;
            background: rgba(255,255,255,0.45) !important;
            border-radius: 12px !important;
            margin-right: 0.35rem !important;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #0f172a !important;
            background: rgba(255,255,255,0.92) !important;
        }
        .stSelectbox label, .stNumberInput label, .stSlider label, .stRadio label, .stTextInput label, .stTextArea label {
            color: var(--ink) !important;
            font-weight: 700 !important;
        }
        .stSelectbox div[data-baseweb="select"] *, .stNumberInput input, .stTextInput input, .stTextArea textarea {
            color: #111827 !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stNumberInput > div > div,
        .stTextInput > div > div,
        .stTextArea textarea {
            background: rgba(255,255,255,0.96) !important;
            border: 1px solid rgba(24, 54, 66, 0.16) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }
        div[data-baseweb="select"] svg,
        div[data-baseweb="input"] svg {
            fill: var(--ink) !important;
        }
        .stSlider [data-baseweb="slider"] {
            padding-top: 0.35rem;
        }
        .stSlider [role="slider"] {
            background: var(--ink) !important;
            border-color: var(--ink) !important;
        }
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {
            background: rgba(24,54,66,0.18) !important;
        }
        .stButton button {
            background: linear-gradient(135deg, #173f4f, #25596f) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 24px rgba(23, 63, 79, 0.18);
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #123544, #1c4f63) !important;
        }
        .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"], .stSlider span {
            color: var(--ink) !important;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: var(--ink) !important;
        }
        [data-testid="stDataFrame"], [data-testid="stDataFrame"] * {
            color: #102a43 !important;
        }
        [data-testid="stTable"], [data-testid="stTable"] * {
            color: #102a43 !important;
        }
        .hero-card, .info-card, .stat-card {
            background: var(--card);
            border: 1px solid rgba(34, 56, 67, 0.08);
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 18px 40px rgba(38, 56, 64, 0.08);
            backdrop-filter: blur(8px);
        }
        .hero-card {
            background:
                linear-gradient(120deg, rgba(255,255,255,0.88), rgba(255,247,235,0.9)),
                var(--card);
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: var(--ink);
            margin-bottom: 0.4rem;
        }
        .hero-subtitle {
            color: #1f3f49;
            font-size: 1rem;
            line-height: 1.6;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--ink);
            margin-top: 0.4rem;
            margin-bottom: 0.6rem;
        }
        .mini-title {
            font-size: 0.88rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }
        .big-number {
            font-size: 1.9rem;
            font-weight: 800;
            color: var(--ink);
            line-height: 1.15;
        }
        .body-copy {
            color: var(--muted);
            line-height: 1.55;
            font-size: 0.95rem;
            font-weight: 500;
        }
        .note-chip {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            background: #173f4f;
            color: #f8fbfc !important;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .warm-chip { background: #a54819; }
        .teal-chip { background: #0f766e; }
        .blue-chip { background: #1d4ed8; }
        .purple-chip { background: #7c3aed; }
        .explain-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .explain-card {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(24, 54, 66, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(38, 56, 64, 0.06);
        }
        .flow-strip {
            display: grid;
            grid-template-columns: 1fr auto 1fr auto 1fr;
            gap: 0.8rem;
            align-items: center;
            margin-top: 0.8rem;
        }
        .flow-node {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(24, 54, 66, 0.10);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            text-align: center;
        }
        .flow-arrow {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--muted);
            text-align: center;
        }
        @media (max-width: 900px) {
            .explain-grid {
                grid-template-columns: 1fr;
            }
            .flow-strip {
                grid-template-columns: 1fr;
            }
            .flow-arrow {
                transform: rotate(90deg);
            }
        }
        [data-testid="stToolbar"],
        header[data-testid="stHeader"] {
            background: rgba(248, 244, 237, 0.96) !important;
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(10px);
        }
        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin-top: 0.4rem;
            margin-bottom: 1rem;
        }
        .prediction-card {
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(24, 54, 66, 0.10);
            border-radius: 20px;
            padding: 1rem 1rem 1.1rem 1rem;
            box-shadow: 0 16px 34px rgba(38, 56, 64, 0.08);
        }
        .prediction-label {
            font-size: 0.84rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #35515b !important;
            margin-bottom: 0.8rem;
        }
        .prediction-value {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }
        .prediction-subvalue {
            font-size: 1rem;
            font-weight: 700;
            color: #29424c !important;
        }
        .prediction-note {
            margin-top: 0.55rem;
            font-size: 0.92rem;
            color: #45606a !important;
            line-height: 1.45;
        }
        .warm-value { color: #b54708 !important; }
        .gold-value { color: #d97706 !important; }
        .teal-value { color: #0f766e !important; }
        .blue-value { color: #1d4ed8 !important; }
        .purple-value { color: #7c3aed !important; }
        @media (max-width: 1100px) {
            .prediction-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 720px) {
            .prediction-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def fluid_input_block(title: str, prefix: str, presets: dict[str, dict[str, float]]) -> dict[str, float]:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    preset_name = st.selectbox(
        f"{title} preset",
        options=[*presets.keys(), "Custom"],
        key=f"{prefix}_preset",
    )
    preset = presets.get(preset_name, presets["Water"]).copy()

    col1, col2 = st.columns(2)
    with col1:
        cp = st.number_input(
            f"{title} specific heat Cp (kJ/kg-K)",
            min_value=0.1,
            max_value=10.0,
            value=float(preset["cp_kj_kgk"]),
            step=0.01,
            key=f"{prefix}_cp",
        )
        rho = st.number_input(
            f"{title} density rho (kg/m^3)",
            min_value=1.0,
            max_value=2500.0,
            value=float(preset["rho_kg_m3"]),
            step=1.0,
            key=f"{prefix}_rho",
        )
    with col2:
        mu = st.number_input(
            f"{title} viscosity mu (Pa.s)",
            min_value=0.00001,
            max_value=5.0,
            value=float(preset["mu_pa_s"]),
            step=0.0001,
            format="%.5f",
            key=f"{prefix}_mu",
        )
        conductivity = st.number_input(
            f"{title} conductivity k (W/m-K)",
            min_value=0.01,
            max_value=5.0,
            value=float(preset["k_w_mk"]),
            step=0.01,
            key=f"{prefix}_k",
        )

    return {
        "name": preset_name,
        "cp_kj_kgk": cp,
        "rho_kg_m3": rho,
        "mu_pa_s": mu,
        "k_w_mk": conductivity,
    }


def kelvin_to_celsius(value_k: float) -> float:
    return value_k - 273.15


def format_temperature(value_k: float) -> str:
    return f"{value_k:.3f} K / {kelvin_to_celsius(value_k):.3f} deg C"


def format_temperature_delta(delta_k: float) -> str:
    delta_c = delta_k
    return f"{delta_k:+.3f} K / {delta_c:+.3f} deg C"


def artifact_metric_lookup(artifact: dict) -> dict[str, dict[str, float]]:
    metrics = artifact.get("metrics", [])
    if isinstance(metrics, list):
        return {row["Target"]: row for row in metrics if "Target" in row}
    return {}


def render_prediction_cards(results: dict, pilstm_hot_outlet: float | None = None) -> None:
    ml_hot_c = kelvin_to_celsius(results["predicted_hot_outlet_k_ml"])
    hybrid_hot_c = kelvin_to_celsius(results["predicted_hot_outlet_k_hybrid"])
    ml_cold_c = kelvin_to_celsius(results["predicted_cold_outlet_k_ml"])
    hybrid_cold_c = kelvin_to_celsius(results["predicted_cold_outlet_k_hybrid"])

    cards = [
        dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">ML Heat Load</div>
                <div class="prediction-value warm-value">{results['predicted_heat_load_kw_ml']:.2f} kW</div>
                <div class="prediction-note">Baseline model prediction learned directly from the dataset.</div>
            </div>
            """
        ).strip(),
        dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Hybrid Heat Load</div>
                <div class="prediction-value gold-value">{results['predicted_heat_load_kw_hybrid']:.2f} kW</div>
                <div class="prediction-note">Fluid-adjusted estimate after applying the physics-aware correction layer.</div>
            </div>
            """
        ).strip(),
        dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">ML Hot Outlet</div>
                <div class="prediction-value teal-value">{results['predicted_hot_outlet_k_ml']:.2f} K</div>
                <div class="prediction-subvalue">{ml_hot_c:.2f} deg C</div>
                <div class="prediction-note">Hot-stream exit temperature predicted by the trained ML model.</div>
            </div>
            """
        ).strip(),
        dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Hybrid Hot Outlet</div>
                <div class="prediction-value blue-value">{results['predicted_hot_outlet_k_hybrid']:.2f} K</div>
                <div class="prediction-subvalue">{hybrid_hot_c:.2f} deg C</div>
                <div class="prediction-note">Hot-stream exit temperature after fluid-property adjustment.</div>
            </div>
            """
        ).strip(),
        dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">ML Cold Outlet</div>
                <div class="prediction-value teal-value">{results['predicted_cold_outlet_k_ml']:.2f} K</div>
                <div class="prediction-subvalue">{ml_cold_c:.2f} deg C</div>
                <div class="prediction-note">Cold-stream exit temperature predicted by the trained ML model.</div>
            </div>
            """
        ).strip(),
        dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Hybrid Cold Outlet</div>
                <div class="prediction-value blue-value">{results['predicted_cold_outlet_k_hybrid']:.2f} K</div>
                <div class="prediction-subvalue">{hybrid_cold_c:.2f} deg C</div>
                <div class="prediction-note">Cold-stream exit temperature after fluid-property adjustment.</div>
            </div>
            """
        ).strip(),
    ]

    if pilstm_hot_outlet is not None:
        pilstm_hot_c = kelvin_to_celsius(pilstm_hot_outlet)
        cards.append(
            dedent(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">PI-LSTM Hot Outlet</div>
                    <div class="prediction-value purple-value">{pilstm_hot_outlet:.2f} K</div>
                    <div class="prediction-subvalue">{pilstm_hot_c:.2f} deg C</div>
                    <div class="prediction-note">Sequence-aware, physics-constrained hot-outlet estimate for direct comparison with the baseline model.</div>
                </div>
                """
            ).strip()
        )

    cards_html = '<div class="prediction-grid">' + "".join(cards) + "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def prediction_compare_chart(results: dict, pilstm_hot_outlet: float | None = None) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Heat Load Comparison (kW)", "Hot Outlet Comparison (K)"),
        horizontal_spacing=0.16,
    )

    fig.add_trace(
        go.Bar(
            x=["LinearRegressionGD", "Hybrid"],
            y=[
                results["predicted_heat_load_kw_ml"],
                results["predicted_heat_load_kw_hybrid"],
            ],
            text=[
                f"{results['predicted_heat_load_kw_ml']:.2f}",
                f"{results['predicted_heat_load_kw_hybrid']:.2f}",
            ],
            textposition="outside",
            marker_color=["#c2410c", "#d97706"],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    hot_models = ["LinearRegressionGD", "Hybrid"]
    hot_values = [
        results["predicted_hot_outlet_k_ml"],
        results["predicted_hot_outlet_k_hybrid"],
    ]
    hot_colors = ["#0f766e", "#1d4ed8"]
    if pilstm_hot_outlet is not None:
        hot_models.append("PI-LSTM")
        hot_values.append(pilstm_hot_outlet)
        hot_colors.append("#7c3aed")

    fig.add_trace(
        go.Bar(
            x=hot_models,
            y=hot_values,
            text=[f"{value:.2f}" for value in hot_values],
            textposition="outside",
            marker_color=hot_colors,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Current Scenario Model Comparison",
        paper_bgcolor="rgba(255,255,255,0.85)",
        plot_bgcolor="rgba(255,255,255,0.75)",
        margin=dict(l=20, r=20, t=70, b=20),
        height=430,
    )
    fig.update_yaxes(title_text="Heat load (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Hot outlet (K)", row=1, col=2)
    return fig


def build_model_comparison_df(
    artifact: dict,
    results: dict,
    pilstm_hot_outlet: float | None = None,
    pilstm_artifact: dict | None = None,
) -> pd.DataFrame:
    metric_lookup = artifact_metric_lookup(artifact)
    hot_metrics = metric_lookup.get("hot_outlet_temperature_k", {})

    rows = [
        {
            "Model": "LinearRegressionGD",
            "Family": "Traditional ML",
            "Predicts": "Heat load + hot outlet",
            "Current heat load": f"{results['predicted_heat_load_kw_ml']:.3f} kW",
            "Current hot outlet": format_temperature(results["predicted_hot_outlet_k_ml"]),
            "Hot-outlet delta vs ML": format_temperature_delta(0.0),
            "Validation view": (
                f"Hot-outlet MAPE {hot_metrics.get('MAPE', float('nan')):.4f}%"
                if hot_metrics
                else "Not available"
            ),
            "Why use it": "Best pure tabular baseline on the saved dataset.",
        },
        {
            "Model": "Hybrid",
            "Family": "ML + physics adjustment",
            "Predicts": "Heat load + hot outlet",
            "Current heat load": f"{results['predicted_heat_load_kw_hybrid']:.3f} kW",
            "Current hot outlet": format_temperature(results["predicted_hot_outlet_k_hybrid"]),
            "Hot-outlet delta vs ML": format_temperature_delta(
                results["predicted_hot_outlet_k_hybrid"] - results["predicted_hot_outlet_k_ml"]
            ),
            "Validation view": "Derived from the ML baseline plus fluid-property corrections.",
            "Why use it": "Best choice when you change fluids and want a fluid-aware scenario estimate.",
        },
    ]

    if pilstm_hot_outlet is not None:
        pi_metrics = (pilstm_artifact or {}).get("metrics", {}).get("hot_outlet", {})
        rows.append(
            {
                "Model": "PI-LSTM",
                "Family": "Physics-informed deep learning",
                "Predicts": "Hot outlet only",
                "Current heat load": "Not predicted",
                "Current hot outlet": format_temperature(pilstm_hot_outlet),
                "Hot-outlet delta vs ML": format_temperature_delta(
                    pilstm_hot_outlet - results["predicted_hot_outlet_k_ml"]
                ),
                "Validation view": (
                    f"Hot-outlet MAPE {pi_metrics.get('MAPE', float('nan')):.4f}%"
                    if pi_metrics
                    else "Not available"
                ),
                "Why use it": "Shows the sequence-aware, physics-constrained alternative for hot outlet prediction.",
            }
        )

    return pd.DataFrame(rows)


def build_pilstm_story(
    artifact: dict,
    results: dict,
    pilstm_hot_outlet: float,
    pilstm_artifact: dict,
) -> str:
    metric_lookup = artifact_metric_lookup(artifact)
    ml_hot_mape = metric_lookup.get("hot_outlet_temperature_k", {}).get("MAPE")
    pi_hot_metrics = pilstm_artifact.get("metrics", {}).get("hot_outlet", {})
    pi_hot_mape = pi_hot_metrics.get("MAPE")
    delta_vs_ml = pilstm_hot_outlet - results["predicted_hot_outlet_k_ml"]
    delta_vs_hybrid = pilstm_hot_outlet - results["predicted_hot_outlet_k_hybrid"]

    if ml_hot_mape is not None and pi_hot_mape is not None and pi_hot_mape <= ml_hot_mape:
        performance_line = (
            f"On the saved validation metrics, PI-LSTM beats LinearRegressionGD on hot outlet "
            f"MAPE ({pi_hot_mape:.4f}% vs {ml_hot_mape:.4f}%)."
        )
    elif ml_hot_mape is not None and pi_hot_mape is not None:
        performance_line = (
            f"On the saved validation metrics, LinearRegressionGD still has lower hot-outlet "
            f"MAPE than PI-LSTM ({ml_hot_mape:.4f}% vs {pi_hot_mape:.4f}%)."
        )
    else:
        performance_line = "Saved validation metrics are not fully available for a direct error comparison."

    return dedent(
        f"""
        **Why PI-LSTM is included**

        - {performance_line}
        - In the current scenario, PI-LSTM predicts {format_temperature(pilstm_hot_outlet)}.
        - That is {format_temperature_delta(delta_vs_ml)} relative to LinearRegressionGD and {format_temperature_delta(delta_vs_hybrid)} relative to the hybrid estimate.
        - PI-LSTM is still important in the dashboard because it is the only model here that carries temporal memory and physics constraints together.
        - That makes it useful for explaining the next step beyond static regression: moving toward transient, sequential heat-exchanger behavior instead of only pointwise tabular predictions.
        """
    ).strip()


def sensitivity_chart(artifact: dict, hot_props: dict[str, float], cold_props: dict[str, float], base_inputs: dict[str, float]) -> go.Figure:
    rows = []
    for hot_temp in range(
        int(base_inputs["hot_inlet_temperature_k"] - 20),
        int(base_inputs["hot_inlet_temperature_k"] + 25),
        5,
    ):
        scenario = predict_scenario(
            artifact=artifact,
            hot_inlet_temperature_k=float(hot_temp),
            hot_inlet_temperature_k_noisy=float(hot_temp + base_inputs["hot_sensor_bias_k"]),
            cold_inlet_temperature_k=base_inputs["cold_inlet_temperature_k"],
            cold_inlet_mass_flow_kg_s=base_inputs["cold_inlet_mass_flow_kg_s"],
            cold_inlet_mass_flow_kg_s_noisy=base_inputs["cold_inlet_mass_flow_kg_s"] + base_inputs["cold_flow_sensor_bias_kg_s"],
            hot_props=hot_props,
            cold_props=cold_props,
        )
        rows.append(
            {
                "Hot inlet temperature (K)": hot_temp,
                "Hybrid heat load (kW)": scenario["predicted_heat_load_kw_hybrid"],
                "Hybrid hot outlet (K)": scenario["predicted_hot_outlet_k_hybrid"],
                "Hybrid cold outlet (K)": scenario["predicted_cold_outlet_k_hybrid"],
            }
        )

    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Hot inlet temperature (K)"],
            y=df["Hybrid heat load (kW)"],
            mode="lines+markers",
            name="Hybrid heat load",
            line=dict(color="#d97706", width=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Hot inlet temperature (K)"],
            y=df["Hybrid hot outlet (K)"],
            mode="lines+markers",
            name="Hybrid hot outlet",
            line=dict(color="#0f766e", width=4),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Hot inlet temperature (K)"],
            y=df["Hybrid cold outlet (K)"],
            mode="lines+markers",
            name="Hybrid cold outlet",
            line=dict(color="#1d4ed8", width=4),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Temperature Sweep Simulation",
        xaxis_title="Hot inlet temperature (K)",
        yaxis=dict(title="Heat load (kW)"),
        yaxis2=dict(title="Outlet temperature (K)", overlaying="y", side="right"),
        paper_bgcolor="rgba(255,255,255,0.85)",
        plot_bgcolor="rgba(255,255,255,0.75)",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def physics_bar_chart(results: dict) -> go.Figure:
    labels = [
        "Delta T noisy",
        "C_hot",
        "C_cold",
        "C_ratio",
        "Energy proxy",
        "Response factor",
    ]
    values = [
        results["delta_t_in_noisy_k"],
        results["capacity_rate_hot_kw_per_k"],
        results["capacity_rate_cold_noisy_kw_per_k"],
        results["capacity_rate_ratio_noisy"],
        results["energy_proxy_noisy_kw"],
        results["thermal_response_factor"],
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=["#c2410c", "#d97706", "#0f766e", "#1d4ed8", "#7c3aed", "#be185d"]
            ),
        )
    )
    fig.update_layout(
        title="Physics Summary",
        paper_bgcolor="rgba(255,255,255,0.85)",
        plot_bgcolor="rgba(255,255,255,0.75)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def render_explanation_panels() -> None:
    st.markdown(
        dedent(
            """
            <div class="explain-grid">
                <div class="explain-card">
                    <div class="mini-title">What Goes In</div>
                    <div class="body-copy">
                        You enter the hot inlet temperature, cold-side mass flow, sensor noise assumptions,
                        and thermophysical properties for the hot and cold liquids.
                    </div>
                </div>
                <div class="explain-card">
                    <div class="mini-title">What We Predict</div>
                    <div class="body-copy">
                        We predict two process outputs: <b>heat load</b>, which is the total heat transferred
                        by the exchanger, and <b>hot outlet temperature</b>, which is the temperature of the
                        hot stream after it leaves the exchanger.
                    </div>
                </div>
                <div class="explain-card">
                    <div class="mini-title">How To Read It</div>
                    <div class="body-copy">
                        <b>ML baseline</b> is the direct learned prediction from the training dataset.
                        <b>Hybrid fluid-adjusted</b> applies a physics-informed correction so you can explore
                        other liquids beyond the original dataset more realistically.
                    </div>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_process_flow(results: dict) -> None:
    hot_in_k = results['delta_t_in_noisy_k'] + 293.15
    hot_out_k = results['predicted_hot_outlet_k_hybrid']
    st.markdown(
        dedent(
            f"""
            <div class="info-card">
                <div class="section-title">What This Simulation Is Doing</div>
                <div class="flow-strip">
                    <div class="flow-node">
                        <div class="mini-title">Hot Stream In</div>
                        <div class="big-number">{hot_in_k:.2f} K</div>
                        <div class="body-copy">{kelvin_to_celsius(hot_in_k):.2f} °C</div>
                        <div class="body-copy">Measured hot-side inlet temperature</div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-node">
                        <div class="mini-title">Exchanger Core</div>
                        <div class="big-number">{results['predicted_heat_load_kw_hybrid']:.2f} kW</div>
                        <div class="body-copy">Estimated heat transferred from hot side to cold side</div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-node">
                        <div class="mini-title">Hot Stream Out</div>
                        <div class="big-number">{hot_out_k:.2f} K</div>
                        <div class="body-copy">{kelvin_to_celsius(hot_out_k):.2f} °C</div>
                        <div class="body-copy">Estimated hot-fluid exit temperature after heat exchange</div>
                    </div>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_hero_card(subtitle: str, chips: list[str]) -> None:
    chips_html = "\n".join(chips)
    st.markdown(
        dedent(
            f"""
            <div class="hero-card">
                <div class="hero-title">Heat Exchanger Digital Twin</div>
                <div class="hero-subtitle">{subtitle}</div>
                <div style="margin-top:0.9rem;">
                    {chips_html}
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


inject_styles()
artifact = load_artifact()
pilstm_artifact = load_pilstm_artifact()

# Default hero content
hero_subtitle = (
    "A polished prediction and simulation dashboard built around the winning "
    "<b>LinearRegressionGD</b> models for <b>heat load</b> and "
    "<b>hot outlet temperature</b>. The dashboard combines machine-learning "
    "predictions with a physics-adjusted scenario layer so you can explore "
    "different liquids and operating conditions."
)

hero_chips = [
    '<span class="note-chip">Best model: LinearRegressionGD</span>',
    '<span class="note-chip warm-chip">Primary prediction: hx_1_heat_load_kw</span>',
    '<span class="note-chip teal-chip">Secondary prediction: hot_outlet_temperature_k</span>',
    '<span class="note-chip blue-chip">Mode: ML baseline + fluid-aware hybrid simulation</span>',
]

# Update if PI-LSTM is available
if pilstm_artifact is not None:
    # Get dynamic accuracy from artifact metrics (only hot outlet is reliable)
    hot_accuracy = 100.0 - pilstm_artifact['metrics']['hot_outlet']['MAPE']
    
    hero_subtitle = (
        "Advanced comparison dashboard combining traditional ML "
        "(<b>LinearRegressionGD</b>), a fluid-aware hybrid layer, and "
        "<b>Physics-Informed LSTM</b>. This lets you compare a strong tabular "
        "baseline against a sequence-aware, physics-constrained model for hot "
        "outlet temperature prediction."
    )
    hero_chips = [
        '<span class="note-chip">Traditional ML: LinearRegressionGD</span>',
        '<span class="note-chip blue-chip">Hybrid: fluid-aware correction layer</span>',
        '<span class="note-chip purple-chip">Deep Learning: Physics-Informed LSTM</span>',
        f'<span class="note-chip warm-chip">PI-LSTM Hot Outlet: {hot_accuracy:.2f}% hot-outlet accuracy</span>',
    ]

render_hero_card(hero_subtitle, hero_chips)

render_explanation_panels()

with st.sidebar:
    st.header("Scenario Inputs")
    st.caption(
        "The ML core is trained on the current dataset. Fluid changes are handled through a "
        "hybrid physics layer, so far-away fluids should be treated as engineering estimates."
    )

    hot_fluid = fluid_input_block("Hot fluid", "hot", FLUID_PRESETS)
    cold_fluid = fluid_input_block("Cold fluid", "cold", FLUID_PRESETS)

    st.markdown("---")
    st.markdown("### Temperature Inputs")
    hot_inlet_temperature_k = st.slider(
        "Hot inlet temperature (K)",
        min_value=320.0,
        max_value=650.0,
        value=473.15,
        step=1.0,
    )
    cold_inlet_temperature_k = st.slider(
        "Cold inlet temperature (K)",
        min_value=280.0,
        max_value=350.0,
        value=293.15,
        step=1.0,
        help="Temperature of cold fluid entering the heat exchanger"
    )
    
    st.markdown("### Flow Rate Inputs")
    cold_inlet_mass_flow_kg_s = st.slider(
        "Cold inlet mass flow (kg/s)",
        min_value=0.30,
        max_value=6.00,
        value=2.75,
        step=0.01,
    )
    
    st.markdown("### Sensor Noise (Optional)")
    hot_sensor_bias_k = st.slider(
        "Hot temperature sensor bias (K)",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
    )
    cold_flow_sensor_bias_kg_s = st.slider(
        "Cold flow sensor bias (kg/s)",
        min_value=-0.50,
        max_value=0.50,
        value=0.0,
        step=0.01,
    )

    run_prediction = st.button("Run Prediction", use_container_width=True, type="primary")


if not run_prediction:
    st.info("Choose your fluids and operating conditions in the sidebar, then click `Run Prediction`.")
    st.stop()

results = predict_scenario(
    artifact=artifact,
    hot_inlet_temperature_k=hot_inlet_temperature_k,
    hot_inlet_temperature_k_noisy=hot_inlet_temperature_k + hot_sensor_bias_k,
    cold_inlet_temperature_k=cold_inlet_temperature_k,
    cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
    cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s + cold_flow_sensor_bias_kg_s,
    hot_props=hot_fluid,
    cold_props=cold_fluid,
)

pilstm_hot_outlet = None
if pilstm_artifact is not None:
    pilstm_hot_outlet = predict_pilstm(
        pilstm_artifact,
        hot_inlet_temperature_k,
        cold_inlet_temperature_k,
        cold_inlet_mass_flow_kg_s,
        results["predicted_heat_load_kw_hybrid"],
    )

base_inputs = {
    "hot_inlet_temperature_k": hot_inlet_temperature_k,
    "cold_inlet_temperature_k": cold_inlet_temperature_k,
    "cold_inlet_mass_flow_kg_s": cold_inlet_mass_flow_kg_s,
    "hot_sensor_bias_k": hot_sensor_bias_k,
    "cold_flow_sensor_bias_kg_s": cold_flow_sensor_bias_kg_s,
}

st.markdown("---")
st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)
render_prediction_cards(results, pilstm_hot_outlet)

render_process_flow(results)

tab_labels = ["What We Predict", "Scenario Dashboard", "Simulation", "Model Details"]
if pilstm_artifact is not None:
    tab_labels.append("PI-LSTM Performance")

tabs = st.tabs(tab_labels)

with tabs[0]:
    left, right = st.columns([0.95, 1.05])
    with left:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>How To Read The Model Outputs</div>", unsafe_allow_html=True)
        st.write(
            "This app predicts the performance of the heat exchanger for the operating condition you enter."
        )
        st.write(
            "`hx_1_heat_load_kw` means the total rate of heat transferred inside the exchanger, measured in kilowatts. "
            "A higher value means the exchanger is transferring more thermal energy from the hot stream to the cold stream."
        )
        st.write(
            "`hot_outlet_temperature_k` means the exit temperature of the hot fluid after it passes through the exchanger. "
            "A lower hot-outlet temperature usually means more heat was removed from the hot stream."
        )
        st.write(
            "So, in simple words: you give inlet conditions and fluid properties, and the app estimates how much heat exchange happens "
            "and what the hot stream temperature will be after the exchanger."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.plotly_chart(
            prediction_compare_chart(results, pilstm_hot_outlet),
            use_container_width=True,
        )

    comparison_df = build_model_comparison_df(
        artifact,
        results,
        pilstm_hot_outlet=pilstm_hot_outlet,
        pilstm_artifact=pilstm_artifact,
    )
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model Comparison Across Approaches</div>", unsafe_allow_html=True)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.caption(
        "PI-LSTM is shown here alongside the traditional models as a proper comparison model, while staying explicit that it predicts only the hot outlet temperature."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if pilstm_hot_outlet is not None and pilstm_artifact is not None:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Why PI-LSTM Is Included</div>", unsafe_allow_html=True)
        st.markdown(build_pilstm_story(artifact, results, pilstm_hot_outlet, pilstm_artifact))
        st.caption(
            "This explanation is presentation-friendly: it gives both the numerical comparison and the reason PI-LSTM matters conceptually."
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    left, right = st.columns([1.05, 0.95])
    with left:
        st.plotly_chart(
            physics_bar_chart(results),
            use_container_width=True,
        )
    with right:
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Hot fluid",
                    "Cold fluid",
                    "Delta T (noisy)",
                    "Heat-capacity rate hot",
                    "Heat-capacity rate cold",
                    "Capacity-rate ratio",
                    "Energy proxy",
                    "Prandtl hot",
                    "Prandtl cold",
                    "Thermal response factor",
                    "Effectiveness proxy",
                    "Hybrid adjustment factor",
                ],
                "Value": [
                    hot_fluid["name"],
                    cold_fluid["name"],
                    f"{results['delta_t_in_noisy_k']:.3f} K",
                    f"{results['capacity_rate_hot_kw_per_k']:.3f} kW/K",
                    f"{results['capacity_rate_cold_noisy_kw_per_k']:.3f} kW/K",
                    f"{results['capacity_rate_ratio_noisy']:.3f}",
                    f"{results['energy_proxy_noisy_kw']:.3f} kW",
                    f"{results['prandtl_hot']:.3f}",
                    f"{results['prandtl_cold']:.3f}",
                    f"{results['thermal_response_factor']:.3f}",
                    f"{results['effectiveness_proxy']:.3f}",
                    f"{results['hybrid_factor']:.3f}",
                ],
            }
        )
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.plotly_chart(
        sensitivity_chart(artifact, hot_fluid, cold_fluid, base_inputs),
        use_container_width=True,
    )

    st.caption(
        "This sweep varies hot inlet temperature while keeping the chosen fluid pair and flow inputs fixed. "
        "It gives a quick process-response view for the current scenario."
    )
    st.caption(
        f"Current hybrid hot-outlet prediction: {results['predicted_hot_outlet_k_hybrid']:.2f} K / "
        f"{kelvin_to_celsius(results['predicted_hot_outlet_k_hybrid']):.2f} deg C"
    )

with tabs[3]:
    metrics_df = pd.DataFrame(artifact["metrics"])
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.subheader("Saved model metrics")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.subheader("Engineering note")
    st.write(
        "The machine-learning core was trained on the provided simulation dataset and the winning "
        "family was `LinearRegressionGD` for both targets. Because the source data does not vary fluid "
        "properties directly, the different-liquid workflow here uses a hybrid layer that combines the "
        "trained ML baseline with physics-based correction factors derived from Cp, density, viscosity, "
        "conductivity, capacity rates, and inlet temperature driving force."
    )
    st.markdown("</div>", unsafe_allow_html=True)

if pilstm_artifact is not None and len(tabs) > 4:
    with tabs[4]:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Physics-Informed LSTM Performance</div>", unsafe_allow_html=True)
        
        metrics = pilstm_artifact['metrics']
        traditional_hot_metrics = artifact_metric_lookup(artifact).get("hot_outlet_temperature_k", {})
        
        st.markdown("### Hot Outlet Temperature Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{metrics['hot_outlet']['MAE']:.4f} K")
            st.metric("RMSE", f"{metrics['hot_outlet']['RMSE']:.4f} K")
        with col2:
            st.metric("R^2", f"{metrics['hot_outlet']['R2']:.6f}")
            st.metric("MAPE", f"{metrics['hot_outlet']['MAPE']:.4f}%")
        with col3:
            accuracy_hot = 100.0 - metrics['hot_outlet']['MAPE']
            traditional_hot_accuracy = (
                100.0 - traditional_hot_metrics["MAPE"] if traditional_hot_metrics else None
            )
            accuracy_delta = (
                f"{accuracy_hot - traditional_hot_accuracy:+.2f}% vs LinearRegressionGD"
                if traditional_hot_accuracy is not None
                else None
            )
            st.metric("Accuracy", f"{accuracy_hot:.2f}%", delta=accuracy_delta)

        st.warning(
            dedent(
                """
                **Note on Cold Outlet Predictions:** The dataset contains erroneous cold outlet temperature data
                (constant at 2011.51 K / 1738 deg C), which is physically impossible for a heat exchanger.
                Therefore, only hot outlet temperature predictions are shown in this dashboard.
                PI-LSTM is therefore used here only as a hot-outlet comparison model.
                """
            ).strip()
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("### About Physics-Informed LSTM")
        st.markdown(
            dedent(
                f"""
                The Physics-Informed LSTM model combines deep learning with thermodynamic principles:

                **Key Features:**
                - Captures temporal patterns in heat exchanger operation
                - Incorporates physics-based constraints (energy conservation, temperature bounds)
                - Works effectively with smaller datasets through sequence learning
                - Learns from sequential process data

                **Architecture:**
                - 3 stacked LSTM layers (64 -> 32 -> 16 units)
                - Dropout regularization to prevent overfitting
                - Custom physics-informed loss function
                - Sequence-based learning for temporal dependencies

                **Advantages over Traditional ML:**
                1. Captures temporal dependencies in process data
                2. Incorporates thermodynamic constraints directly in loss function
                3. Better generalization with limited data through sequence learning
                4. Physically meaningful predictions that respect energy conservation

                **Hot Outlet Performance:**
                - Accuracy: {accuracy_hot:.2f}%
                - MAPE: {metrics['hot_outlet']['MAPE']:.4f}%
                - RMSE: {metrics['hot_outlet']['RMSE']:.4f} K
                - R^2: {metrics['hot_outlet']['R2']:.6f}

                This model is included to show the sequence-aware, physics-constrained alternative.
                On this saved dataset, it should be interpreted as a comparison model rather than the current
                overall winner against the tabular LinearRegressionGD baseline.
                """
            ).strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)
