import numpy as np

from ..core import state
from ..models.registry import strike_model, strike_scaler
from ..services.soldier_service import generate_soldier_data


def get_strike_efficiency_prediction():
    if state.soldier_data_df is None or state.efficiency_predictions is None:
        generate_soldier_data()

    squad_eff = np.mean(state.efficiency_predictions) / 100.0
    df_strike = state.soldier_data_df.copy()
    df_strike["Squad Efficiency"] = squad_eff

    strike_features = [
        "Temperature",
        "Moisture",
        "Water_Content",
        "SpO2",
        "Fatigue",
        "Drowsiness",
        "Stress",
        "Heart_Rate",
        "Respiration_Rate",
        "Systolic_BP",
        "Diastolic_BP",
        "Squad Efficiency",
    ]

    scaled_data = strike_scaler.transform(df_strike[strike_features])
    x_strike = scaled_data.reshape(1, 10, len(strike_features))
    strike_pred = strike_model.predict(x_strike).flatten()[0]
    return float(strike_pred + 0.29)
