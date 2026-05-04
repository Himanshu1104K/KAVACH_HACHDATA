import os
from pathlib import Path

# Disable GPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
TRAINED_MODELS_DIR = BACKEND_DIR / "Trained_Models"

MODEL_FILE = TRAINED_MODELS_DIR / "Efficiency_Model.keras"
SCALER_FILE = TRAINED_MODELS_DIR / "Efficiency_Scaler.pkl"
STRIKE_FILE = TRAINED_MODELS_DIR / "Surgical_Model.keras"
STRIKE_SCALER_FILE = TRAINED_MODELS_DIR / "Surgical_Scaler.pkl"
SOLDIER_TACTICS_FILE = TRAINED_MODELS_DIR / "soldier_tactics_model.keras"
SOLDIER_TACTICS_SCALER = TRAINED_MODELS_DIR / "Tactics_Scaler.pkl"
SOLDIER_TACTICS_LABEL_ENCODER = TRAINED_MODELS_DIR / "Tactics_Label_Encoder.pkl"

SOLDIER_COLUMNS = [
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
]
