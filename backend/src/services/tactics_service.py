import numpy as np
import pandas as pd

from ..core import state
from ..models.registry import label_encoder, tactic_model, tactic_scaler


def predict_tactic(input_data):
    input_scaled = tactic_scaler.transform([input_data])
    prediction = tactic_model.predict(input_scaled)
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]


def get_soldier_tactics_formation():
    soldier_df = pd.DataFrame(state.soldier_data_df)
    soldier_df["efficiency_predictions"] = state.efficiency_predictions
    soldier_df["x"] = [np.random.randint(1, 100) for _ in range(len(soldier_df))]
    soldier_df["y"] = [np.random.randint(1, 100) for _ in range(len(soldier_df))]

    formatted_data = {
        "efficiency_predictions": soldier_df["efficiency_predictions"].tolist(),
        "Temperature": soldier_df["Temperature"].tolist(),
        "Moisture": soldier_df["Moisture"].tolist(),
        "Water_Content": soldier_df["Water_Content"].tolist(),
        "SpO2": soldier_df["SpO2"].tolist(),
        "Fatigue": soldier_df["Fatigue"].tolist(),
        "Drowsiness": soldier_df["Drowsiness"].tolist(),
        "Stress": soldier_df["Stress"].tolist(),
        "Heart_Rate": soldier_df["Heart_Rate"].tolist(),
        "Respiration_Rate": soldier_df["Respiration_Rate"].tolist(),
        "x": soldier_df["x"].tolist(),
        "y": soldier_df["y"].tolist(),
    }
    aggregated_data = {
        key: np.mean(value) if isinstance(value, list) else value
        for key, value in formatted_data.items()
    }

    return predict_tactic(list(aggregated_data.values()))
