import joblib as jb
import tensorflow as tf

from ..core.config import (
    MODEL_FILE,
    SCALER_FILE,
    SOLDIER_TACTICS_FILE,
    SOLDIER_TACTICS_LABEL_ENCODER,
    SOLDIER_TACTICS_SCALER,
    STRIKE_FILE,
    STRIKE_SCALER_FILE,
)

model = tf.keras.models.load_model(MODEL_FILE)
scaler = jb.load(SCALER_FILE)

strike_model = tf.keras.models.load_model(STRIKE_FILE)
strike_scaler = jb.load(STRIKE_SCALER_FILE)

tactic_model = tf.keras.models.load_model(SOLDIER_TACTICS_FILE)
tactic_scaler = jb.load(SOLDIER_TACTICS_SCALER)
label_encoder = jb.load(SOLDIER_TACTICS_LABEL_ENCODER)
