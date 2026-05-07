# Version 2 Plan

Version 2 is the low-data experiment track.

It now reuses the shared full dataset and baseline saved artifacts from `version_1/`, so the two tracks stay clearly separated.

## Purpose

This version is meant to answer the stronger research question:

`What happens when the available heat-exchanger data is small, but still covers the real operating temperature range?`

That is the right place to test why LSTM / PI-LSTM may become more useful than standard tabular models.

## Design Principle

Do not create the low-data dataset by random sampling alone.

Random sampling can accidentally pull many rows from similar temperature values and produce a misleading experiment.

Instead, we keep:

- wide hot-inlet temperature coverage
- exact hot-inlet temperature coverage when the subset size allows it
- within-range diversity in cold-side mass flow
- deterministic, reproducible subset generation

## Files

- `build_low_data_subsets.py`
  Creates low-data subsets that preserve temperature-range coverage.

- `data/`
  Output folder for generated low-data CSV files and sampling summaries.

## Recommended Workflow

1. Generate low-data subsets.
2. Train the same Version 1 traditional ML models on each subset.
3. Train plain deep-learning baselines (`MLP`, `VanillaLSTM`) for hot-outlet comparison.
4. Train `PI-LSTM` on the same low-data hot-outlet problem.
5. Compare degradation as data becomes smaller.
6. Present Version 2 as the evidence-based justification for sequence-aware modeling under data scarcity.

## Example

```powershell
python version_2/build_low_data_subsets.py
```

This creates multiple subset sizes while preserving temperature diversity across the full operating range.
By default it generates `100` and `125` row subsets to match the intended real-world low-data setting.

## Run The Study

```powershell
python version_2/run_low_data_study.py
```

This produces dynamic result files in `version_2/results/`.

The hot-outlet comparison now includes:

- `LinearRegressionGD` as the strong traditional baseline
- `MLP` as the plain dense deep-learning baseline
- `VanillaLSTM` as the plain sequence baseline
- `PI-LSTM` as the sequence-aware physics-informed comparison model

## Run The Presentation Dashboard

```powershell
streamlit run version_2/streamlit_app.py
```

The dashboard reads the saved study outputs and shows the actual best-performing model for each subset.

## Honest Interpretation

Version 2 is intentionally designed to be honest, not pre-scripted.

- If PI-LSTM is best, the dashboard will show that.
- If `LinearRegressionGD` still stays best on this clean simulation dataset, the dashboard will show that instead.
- If `MLP` or `VanillaLSTM` underperform, the dashboard will show that too.
- The useful story then becomes: plain deep learning and plain sequence models are being tested correctly under low data, while this specific dataset may still favor a simple linear baseline.
