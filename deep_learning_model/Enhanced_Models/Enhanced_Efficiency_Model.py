import os
import tensorflow as tf
from tensorflow.keras import regularizers, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input,
    concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {len(gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found, using CPU")

# Create output directories if they don't exist
os.makedirs("output", exist_ok=True)

# Load the enhanced realistic data
print("Loading data...")
data = pd.read_csv("../Training_Data/realistic_soldier_data.csv")

# Display basic info about the dataset
print(f"Dataset shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

# Convert timestamp to datetime
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

# Feature engineering
print("\nPerforming feature engineering...")

# Extract time-based features
data["Day_of_Week"] = data["Timestamp"].dt.dayofweek
data["Is_Weekend"] = (data["Day_of_Week"] >= 5).astype(int)

# Create cyclical time features for better representation
data["Hour_sin"] = np.sin(data["Hour_of_Day"] * (2 * np.pi / 24))
data["Hour_cos"] = np.cos(data["Hour_of_Day"] * (2 * np.pi / 24))
data["Shift_sin"] = np.sin(data["Hours_into_Shift"] * (2 * np.pi / 8))
data["Shift_cos"] = np.cos(data["Hours_into_Shift"] * (2 * np.pi / 8))

# Physiological stress index
data["Physio_Stress_Index"] = (
    ((data["Heart_Rate"] - 60) / 60)
    + ((data["Respiration_Rate"] - 12) / 12)
    + ((data["Systolic_BP"] - 110) / 30)
) / 3

# Hydration index
data["Hydration_Index"] = (data["Water_Content"] / 100) * (data["Moisture"] / 70)

# Define feature sets for different aspects of performance
vital_features = [
    "Temperature",
    "SpO2",
    "Heart_Rate",
    "Respiration_Rate",
    "Systolic_BP",
    "Diastolic_BP",
]
environmental_features = ["Moisture", "Water_Content", "Hydration_Index"]
psychological_features = ["Fatigue", "Drowsiness", "Stress", "Physio_Stress_Index"]
temporal_features = ["Hour_sin", "Hour_cos", "Shift_sin", "Shift_cos", "Is_Weekend"]

# Combine all features
all_features = (
    vital_features + environmental_features + psychological_features + temporal_features
)

# Analyze feature correlations
print("\nAnalyzing feature correlations...")
plt.figure(figsize=(12, 10))
selected_features = (
    vital_features[:3]
    + environmental_features[:2]
    + psychological_features[:3]
    + ["Efficiency"]
)
correlation = data[selected_features].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.tight_layout()
plt.savefig("output/feature_correlations.png")
plt.close()

# Prepare data for modeling
print("\nPreparing data for modeling...")
X = data[all_features]
y = data["Efficiency"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create efficiency bins for stratified sampling
y_bins = pd.cut(y, bins=10, labels=False)

# Split into train/validation/test sets with stratification
X_train_val, X_test, y_train_val, y_test, bins_train_val, bins_test = train_test_split(
    X_scaled, y, y_bins, test_size=0.15, random_state=42, stratify=y_bins
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=bins_train_val,
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# Build multi-branch model with feature representation learning
def build_enhanced_model(input_shape):
    # Main input
    main_input = Input(shape=(input_shape,), name="main_input")

    # Split input into different feature groups
    # Since we're using a Dense model, we'll use slicing to separate feature groups
    vital_count = len(vital_features)
    env_count = len(environmental_features)
    psych_count = len(psychological_features)
    temp_count = len(temporal_features)

    # Branch 1: Vitals processing
    vitals_input = layers.Lambda(lambda x: x[:, :vital_count])(main_input)
    vitals = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        vitals_input
    )
    vitals = BatchNormalization()(vitals)
    vitals = Dropout(0.2)(vitals)
    vitals = Dense(16, activation="relu")(vitals)

    # Branch 2: Environmental processing
    env_input = layers.Lambda(lambda x: x[:, vital_count : vital_count + env_count])(
        main_input
    )
    env = Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        env_input
    )
    env = BatchNormalization()(env)
    env = Dropout(0.2)(env)
    env = Dense(8, activation="relu")(env)

    # Branch 3: Psychological processing
    psych_input = layers.Lambda(
        lambda x: x[:, vital_count + env_count : vital_count + env_count + psych_count]
    )(main_input)
    psych = Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        psych_input
    )
    psych = BatchNormalization()(psych)
    psych = Dropout(0.3)(psych)
    psych = Dense(8, activation="relu")(psych)

    # Branch 4: Temporal processing
    temp_input = layers.Lambda(lambda x: x[:, -temp_count:])(main_input)
    temp = Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        temp_input
    )
    temp = BatchNormalization()(temp)
    temp = Dropout(0.2)(temp)
    temp = Dense(8, activation="relu")(temp)

    # Branch 5: Direct pathway for residual connections
    direct = Dense(
        32, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)
    )(main_input)
    direct = Dropout(0.2)(direct)

    # Merge all branches
    merged = concatenate([vitals, env, psych, temp, direct])

    # Deep representation learning
    x = Dense(128, activation="relu")(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)

    # Output layer
    output = Dense(1, activation="linear", name="output")(x)

    # Create and return model
    model = Model(inputs=main_input, outputs=output)
    return model


# Create the model
print("\nBuilding enhanced model...")
model = build_enhanced_model(X_train.shape[1])
model.summary()

# Define callbacks for better training
checkpoint_path = "best_efficiency_model.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=7, min_lr=0.00001, verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae", "mse"])

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

# Plot training history
print("\nPlotting training history...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("Mean Absolute Error Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.savefig("output/training_history.png")
plt.close()

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
test_rmse = np.sqrt(test_mse)

print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Make predictions and visualize results
y_pred = model.predict(X_test).flatten()

# Calculate R²
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Efficiency")
plt.ylabel("Predicted Efficiency")
plt.title(f"Predicted vs Actual Efficiency (R² = {r2:.4f})")
plt.grid(True)
plt.savefig("output/prediction_results.png")
plt.close()

# Feature importance analysis using permutation importance
from sklearn.inspection import permutation_importance

print("\nCalculating feature importance...")


# Wrapper for sklearn compatibility
def predict_wrapper(X):
    return model.predict(X).flatten()


result = permutation_importance(
    predict_wrapper, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Sort features by importance
feature_importance = pd.DataFrame(
    {"Feature": all_features, "Importance": result.importances_mean}
).sort_values("Importance", ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(15))
plt.title("Feature Importance (Permutation Method)")
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.close()

# Error distribution analysis
errors = y_pred - y_test
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(errors, bins=30, alpha=0.7)
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.scatter(y_pred, errors, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="-")
plt.title("Prediction Error vs Predicted Value")
plt.xlabel("Predicted Value")
plt.ylabel("Error")

plt.tight_layout()
plt.savefig("output/error_analysis.png")
plt.close()

# Save the model and scaler
print("\nSaving model and preprocessing components...")
model.save("enhanced_efficiency_model.h5")

with open("efficiency_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_list.pkl", "wb") as f:
    pickle.dump(all_features, f)

print("\nModel training and evaluation complete!")
print(f"Model saved to enhanced_efficiency_model.h5")
print(f"Scaler saved to efficiency_scaler.pkl")
print(f"Feature list saved to feature_list.pkl")


# Create a prediction function for deployment
def predict_efficiency(new_data, model_path, scaler_path, feature_list_path):
    """
    Predict efficiency for new soldier data

    Parameters:
    new_data (DataFrame): DataFrame containing required features
    model_path: Path to the saved model
    scaler_path: Path to the saved scaler
    feature_list_path: Path to the saved feature list

    Returns:
    Array of predicted efficiencies
    """
    # Load model and preprocessing components
    model = tf.keras.models.load_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(feature_list_path, "rb") as f:
        feature_list = pickle.load(f)

    # Process data
    processed_data = new_data.copy()

    # Process timestamp if present
    if "Timestamp" in processed_data.columns:
        processed_data["Timestamp"] = pd.to_datetime(processed_data["Timestamp"])
        if "Hour_of_Day" not in processed_data.columns:
            processed_data["Hour_of_Day"] = (
                processed_data["Timestamp"].dt.hour
                + processed_data["Timestamp"].dt.minute / 60
            )

        processed_data["Day_of_Week"] = processed_data["Timestamp"].dt.dayofweek
        processed_data["Is_Weekend"] = (processed_data["Day_of_Week"] >= 5).astype(int)

        # Create cyclical features
        processed_data["Hour_sin"] = np.sin(
            processed_data["Hour_of_Day"] * (2 * np.pi / 24)
        )
        processed_data["Hour_cos"] = np.cos(
            processed_data["Hour_of_Day"] * (2 * np.pi / 24)
        )

    # Create shift features if needed
    if "Hours_into_Shift" in processed_data.columns:
        processed_data["Shift_sin"] = np.sin(
            processed_data["Hours_into_Shift"] * (2 * np.pi / 8)
        )
        processed_data["Shift_cos"] = np.cos(
            processed_data["Hours_into_Shift"] * (2 * np.pi / 8)
        )

    # Create derived features
    if all(
        f in processed_data.columns
        for f in ["Heart_Rate", "Respiration_Rate", "Systolic_BP"]
    ):
        processed_data["Physio_Stress_Index"] = (
            ((processed_data["Heart_Rate"] - 60) / 60)
            + ((processed_data["Respiration_Rate"] - 12) / 12)
            + ((processed_data["Systolic_BP"] - 110) / 30)
        ) / 3

    if all(f in processed_data.columns for f in ["Water_Content", "Moisture"]):
        processed_data["Hydration_Index"] = (processed_data["Water_Content"] / 100) * (
            processed_data["Moisture"] / 70
        )

    # Check for missing features
    missing_features = [f for f in feature_list if f not in processed_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")

    # Select and scale features
    X_new = processed_data[feature_list]
    X_new_scaled = scaler.transform(X_new)

    # Predict
    predictions = model.predict(X_new_scaled).flatten()

    return predictions


# Display a sample prediction
print("\nTesting prediction function with sample data...")
sample_data = data.iloc[:5].copy()
sample_predictions = predict_efficiency(
    sample_data,
    "enhanced_efficiency_model.h5",
    "efficiency_scaler.pkl",
    "feature_list.pkl",
)

print("\nSample predictions:")
for i, (actual, pred) in enumerate(zip(sample_data["Efficiency"], sample_predictions)):
    print(f"Sample {i+1}: Actual Efficiency = {actual:.4f}, Predicted = {pred:.4f}")

print("\nEnhanced Efficiency Model pipeline complete!")
