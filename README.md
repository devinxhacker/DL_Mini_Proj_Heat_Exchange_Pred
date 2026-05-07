# Heat Exchanger Digital Twin

The repository is now separated cleanly by version:

- `version_1/` contains the original dashboard, training scripts, dataset, notebook, and saved artifacts.
- `version_2/` contains the low-data experiment track and reads the shared baseline assets from `version_1/`.

## 🎯 NEW: Physics-Based Verification Mode

Both versions now include **instructor-verifiable physics calculations**:

- ✅ **Input:** Hot inlet temp + Hot outlet temp
- ✅ **Output:** Hot temp, Cold temp, Heat exchanged
- ✅ **Verification:** All calculations can be verified by hand using standard heat exchanger equations
- ✅ **Energy Balance:** Guaranteed correct (error ~0)

**Quick Test:**
```bash
python test_physics_calculations.py
```

**Documentation:**
- `PHYSICS_ENHANCEMENTS.md` - Technical details
- `INSTRUCTOR_VERIFICATION_GUIDE.md` - Manual verification guide
- `CHANGES_SUMMARY.md` - Complete summary of changes

## Quick Start

```bash
streamlit run version_1/streamlit_app.py
streamlit run version_2/streamlit_app.py
```

## Training And Study Commands

```bash
python version_1/train_best_heat_exchanger_artifact.py
python version_1/train_pilstm_artifact.py
python version_2/build_low_data_subsets.py
python version_2/run_low_data_study.py
```

## Folder Guide

- `version_1/`: baseline project files and presentation-ready app
- `version_2/`: low-data research workflow, subsets, and comparison dashboard
- `temp/`: scratch area
