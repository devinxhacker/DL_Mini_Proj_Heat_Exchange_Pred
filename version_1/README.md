# Heat Exchanger Digital Twin with PI-LSTM

## Quick Start

Run these commands from the repository root:

```bash
# Run Dashboard
streamlit run version_1/streamlit_app.py

# Train Models (if needed)
python version_1/train_best_heat_exchanger_artifact.py
python version_1/train_pilstm_artifact.py
```

## Files

**Main Application:**
- `streamlit_app.py` - Dashboard (5 prediction cards)
- `heat_exchanger_best_model.py` - ML utilities
- `physics_informed_lstm.py` - PI-LSTM model

**Training Scripts:**
- `train_best_heat_exchanger_artifact.py` - Train traditional ML
- `train_pilstm_artifact.py` - Train PI-LSTM

**Data & Models:**
- `heat_exchanger_dataset.csv` - Dataset
- `best_heat_exchanger_models.joblib` - Traditional ML models
- `pilstm_artifact.joblib` - PI-LSTM configuration
- `pilstm_model.weights.h5` - PI-LSTM weights

**Notebook:**
- `heat_exchanger_research_from_scratch_colab.ipynb` - Complete analysis

## Models

| Model | Hot Outlet Accuracy |
|-------|-------------------|
| Traditional ML | ~98% |
| Hybrid | ~98.5% |
| **PI-LSTM** | **99.39%** |

## Dashboard Features

- 5 prediction cards (ML, Hybrid, PI-LSTM)
- Dynamic accuracy display
- Realistic predictions
- Professional styling
- Data quality warning

## Requirements

```bash
pip install streamlit pandas numpy plotly joblib tensorflow scikit-learn
```
