# Optimising Commercial Building Energy Consumption Using LSTM

MSc Human-Centred AI thesis project, Technological University Dublin (TU Dublin)
**Author:** Dikshitha Kalva Vinod
**Supervisors:** Dr. Na Li, Dr. Rajesh Jaiswal
**Industry partner:** Electric Blue Technology

## Overview

Short-term energy forecasting for a commercial frozen food warehouse using IoT sensor data and LSTM networks, combined with Explainable AI (SHAP, LIME, Permutation Feature Importance) so that predictions can be understood and trusted by facility managers.

## Research Questions

- **RQ1:** To what extent can an AI model predict short-term commercial building energy consumption using high-resolution IoT sensor data?
- **RQ2:** How can XAI techniques such as SHAP and LIME improve the transparency of LSTM-based forecasting models and identify the key features behind predictions?

## Dataset

- Real-world IoT data provided by Electric Blue Technology
- 12,587 raw observations at about 15-minute intervals, reduced to 8,571 after cleaning
- Target variable: `real_power_total_kW`
- Predictors: current channels 1-4, voltage, power factor, battery, signal, and time features (hour, cyclical hour encoding, day of week, weekend)

> **Note:** The dataset is confidential and was shared for academic use only, so it is **not included** in this repository. 

## Methodology

1. Data preprocessing (JSON expansion, timestamp conversion, missing-value handling)
2. Feature analysis and selection
3. Comparison of six models: Basic LSTM, Regularised LSTM, Stacked LSTM, Deep Stacked LSTM, BiLSTM, GRU
4. Hyperparameter tuning of the Stacked LSTM
5. Ablation study
6. Explainability analysis (SHAP, LIME, Permutation Importance)

A chronological train/validation/test split was used to avoid data leakage, with a 24-hour input window.

## Results

Final model: Stacked LSTM (64 → 32 units, dropout 0.2, batch size 32, learning rate 0.01)

| Metric | Value |
|--------|-------|
| RMSE   | 7.6142 kW |
| MAE    | 5.0484 kW |
| sMAPE  | 48.73% |
| R²     | 0.6216 |

**Key finding:** SHAP, LIME and Permutation Importance all identified `current_chan1`, `current_chan2` and `current_chan3` as the dominant predictors, followed by time-of-day features. This is physically plausible, since electrical current is directly linked to power consumption.

## Getting Started

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
jupyter notebook notebooks/Final_Code_Thesis.ipynb
```

To run the Streamlit app [if added]:

```bash
streamlit run app.py
```

## Acknowledgements

Thanks to Electric Blue Technology for providing the dataset, and to my supervisors at TU Dublin.
