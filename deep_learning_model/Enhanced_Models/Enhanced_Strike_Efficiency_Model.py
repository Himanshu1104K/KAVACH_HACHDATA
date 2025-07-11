import os
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input, concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
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

# Display basic info
print(f"Dataset shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

# Convert timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Feature engineering
print("\nPerforming feature engineering...")

# Extract time-based features
data['Day_of_Week'] = data['Timestamp'].dt.dayofweek
data['Is_Weekend'] = (data['Day_of_Week'] >= 5).astype(int)

# Create cyclical time features
data['Hour_sin'] = np.sin(data['Hour_of_Day'] * (2 * np.pi / 24))
data['Hour_cos'] = np.cos(data['Hour_of_Day'] * (2 * np.pi / 24))
data['Shift_sin'] = np.sin(data['Hours_into_Shift'] * (2 * np.pi / 8))
data['Shift_cos'] = np.cos(data['Hours_into_Shift'] * (2 * np.pi / 8))

# Physiological stress index
data['Physio_Stress_Index'] = (
    ((data['Heart_Rate'] - 60) / 60) + 
    ((data['Respiration_Rate'] - 12) / 12) + 
    ((data['Systolic_BP'] - 110) / 30)
) / 3

# Hydration index
data['Hydration_Index'] = (data['Water_Content'] / 100) * (data['Moisture'] / 70)

# Calculate Squad Efficiency (average of nearby soldiers)
# Sort by timestamp to simulate squad formation
data = data.sort_values('Timestamp').reset_index(drop=True)

# Create squad ID (every 10 soldiers form a squad)
data['Squad_ID'] = data.index // 10

# Calculate squad-level metrics
squad_metrics = data.groupby('Squad_ID').agg({
    'Efficiency': 'mean',
    'Heart_Rate': 'mean',
    'Fatigue': 'mean',
    'Hydration_Index': 'mean',
    'Physio_Stress_Index': 'mean'
}).rename(columns={
    'Efficiency': 'Squad_Efficiency',
    'Heart_Rate': 'Squad_Heart_Rate',
    'Fatigue': 'Squad_Fatigue',
    'Hydration_Index': 'Squad_Hydration',
    'Physio_Stress_Index': 'Squad_Stress'
})

# Merge squad metrics back to individual data
data = data.merge(squad_metrics, on='Squad_ID')

# Define target variable - Calculate Strike Efficiency
# Strike Efficiency is a combination of individual and squad efficiency
# with emphasis on physiological factors
data['Strike_Efficiency'] = (
    0.6 * data['Efficiency'] + 
    0.4 * data['Squad_Efficiency'] - 
    0.1 * data['Physio_Stress_Index'] + 
    0.1 * data['Hydration_Index']
)

# Normalize Strike_Efficiency to 0-1 range
data['Strike_Efficiency'] = (data['Strike_Efficiency'] - data['Strike_Efficiency'].min()) / (data['Strike_Efficiency'].max() - data['Strike_Efficiency'].min())

# Define feature sets for different aspects of performance
vital_features = ['Temperature', 'SpO2', 'Heart_Rate', 'Respiration_Rate', 'Systolic_BP', 'Diastolic_BP']
environmental_features = ['Moisture', 'Water_Content', 'Hydration_Index']
psychological_features = ['Fatigue', 'Drowsiness', 'Stress', 'Physio_Stress_Index']
temporal_features = ['Hour_sin', 'Hour_cos', 'Shift_sin', 'Shift_cos', 'Is_Weekend']
squad_features = ['Squad_Efficiency', 'Squad_Heart_Rate', 'Squad_Fatigue', 'Squad_Hydration', 'Squad_Stress']

# Base features for time series
base_features = vital_features + environmental_features + psychological_features

# Print feature importance for understanding
print("\nCorrelation with Strike Efficiency:")
for feature in base_features:
    corr = data[feature].corr(data['Strike_Efficiency'])
    print(f"{feature}: {corr:.4f}")

# Function to create time series sequences
def create_sequences(data, features, target, seq_length=10):
    """
    Create time series sequences for LSTM model
    """
    X, y = [], []
    for squad in data['Squad_ID'].unique():
        squad_data = data[data['Squad_ID'] == squad]
        if len(squad_data) < seq_length:
            continue
            
        # Get features and target
        squad_features = squad_data[features].values
        squad_target = squad_data[target].values
        
        # Create sequences
        for i in range(len(squad_data) - seq_length + 1):
            X.append(squad_features[i:i+seq_length])
            y.append(squad_target[i+seq_length-1])  # Target is the last value in sequence
    
    return np.array(X), np.array(y)

# Create time series sequences
seq_length = 5  # Consider 5 time steps
print(f"\nCreating time series sequences with length {seq_length}...")

X_seq, y_seq = create_sequences(
    data, 
    base_features, 
    'Strike_Efficiency', 
    seq_length=seq_length
)

# For individual-level features that don't change in the sequence
# We'll use them as additional inputs alongside the sequence data
ind_features = temporal_features + squad_features
X_ind = np.zeros((X_seq.shape[0], len(ind_features)))

# Populate individual features (using the last record of each sequence)
for i, squad in enumerate(data['Squad_ID'].unique()):
    if len(data[data['Squad_ID'] == squad]) < seq_length:
        continue
        
    start_idx = i * (len(data[data['Squad_ID'] == squad]) - seq_length + 1)
    end_idx = start_idx + (len(data[data['Squad_ID'] == squad]) - seq_length + 1)
    
    for j in range(start_idx, end_idx):
        if j < len(X_ind):  # Ensure we don't go out of bounds
            seq_end_idx = (j - start_idx) + seq_length - 1
            sqd_idx = data[data['Squad_ID'] == squad].index[seq_end_idx]
            X_ind[j] = data.loc[sqd_idx, ind_features].values

print(f"Sequence data shape: {X_seq.shape}")
print(f"Individual features shape: {X_ind.shape}")
print(f"Target shape: {y_seq.shape}")

# Scale the data
scaler_seq = StandardScaler()
n_samples, n_timesteps, n_features = X_seq.shape
X_seq_reshaped = X_seq.reshape(n_samples * n_timesteps, n_features)
X_seq_scaled = scaler_seq.fit_transform(X_seq_reshaped).reshape(n_samples, n_timesteps, n_features)

scaler_ind = StandardScaler()
X_ind_scaled = scaler_ind.fit_transform(X_ind)

# Split the data
print("\nSplitting data into train/validation/test sets...")
X_seq_train_val, X_seq_test, X_ind_train_val, X_ind_test, y_train_val, y_test = train_test_split(
    X_seq_scaled, X_ind_scaled, y_seq, test_size=0.15, random_state=42
)

X_seq_train, X_seq_val, X_ind_train, X_ind_val, y_train, y_val = train_test_split(
    X_seq_train_val, X_ind_train_val, y_train_val, test_size=0.15/0.85, random_state=42
)

print(f"Training set: {X_seq_train.shape[0]} samples")
print(f"Validation set: {X_seq_val.shape[0]} samples")
print(f"Test set: {X_seq_test.shape[0]} samples")

# Build advanced hybrid model with LSTM for time series and Dense for individual features
def build_hybrid_strike_model(seq_shape, ind_shape):
    # Time Series Branch (LSTM)
    seq_input = Input(shape=seq_shape, name='sequence_input')
    
    # Bidirectional LSTM layer
    lstm = Bidirectional(LSTM(64, return_sequences=True, 
                         kernel_regularizer=regularizers.l2(0.001)))(seq_input)
    lstm = Dropout(0.3)(lstm)
    
    # Additional LSTM layers
    lstm = Bidirectional(LSTM(32, return_sequences=False))(lstm)
    lstm = Dropout(0.3)(lstm)
    lstm = Dense(32, activation='relu')(lstm)
    
    # CNN for sequence feature extraction
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu', 
                padding='same')(seq_input)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=32, kernel_size=3, activation='relu', 
                padding='same')(cnn)
    cnn = GlobalAveragePooling1D()(cnn)
    cnn = Dense(32, activation='relu')(cnn)
    
    # Individual Features Branch
    ind_input = Input(shape=ind_shape, name='individual_input')
    ind = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(ind_input)
    ind = BatchNormalization()(ind)
    ind = Dropout(0.2)(ind)
    ind = Dense(16, activation='relu')(ind)
    
    # Merge all branches
    merged = concatenate([lstm, cnn, ind])
    
    # Deep representation learning
    x = Dense(64, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid', name='output')(x)
    
    # Create and compile model
    model = Model(inputs=[seq_input, ind_input], outputs=output)
    return model

# Create model
print("\nBuilding hybrid strike efficiency model...")
model = build_hybrid_strike_model(
    seq_shape=(X_seq_train.shape[1], X_seq_train.shape[2]),
    ind_shape=(X_ind_train.shape[1],)
)

model.summary()

# Define callbacks
checkpoint_path = "best_strike_efficiency_model.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor='val_loss',
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=0.00001,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, 
    loss="mean_squared_error", 
    metrics=["mae"]
)

# Train model
print("\nTraining model...")
history = model.fit(
    [X_seq_train, X_ind_train], y_train,
    epochs=100,
    batch_size=32,
    validation_data=([X_seq_val, X_ind_val], y_val),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
print("\nPlotting training history...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('output/strike_training_history.png')
plt.close()

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_mae = model.evaluate([X_seq_test, X_ind_test], y_test, verbose=0)
test_mse = test_loss  # Since we're using MSE as the loss
test_rmse = np.sqrt(test_mse)

print(f"Test Loss (MSE): {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Make predictions
y_pred = model.predict([X_seq_test, X_ind_test]).flatten()

# Calculate R²
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Strike Efficiency')
plt.ylabel('Predicted Strike Efficiency')
plt.title(f'Predicted vs Actual Strike Efficiency (R² = {r2:.4f})')
plt.grid(True)
plt.savefig('output/strike_prediction_results.png')
plt.close()

# Error distribution analysis
errors = y_pred - y_test
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(errors, bins=30, alpha=0.7)
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.scatter(y_pred, errors, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Error vs Predicted Value')
plt.xlabel('Predicted Value')
plt.ylabel('Error')

plt.tight_layout()
plt.savefig('output/strike_error_analysis.png')
plt.close()

# Save models and preprocessing components
print("\nSaving model and preprocessing components...")
model.save('enhanced_strike_efficiency_model.h5')

with open('strike_seq_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_seq, f)

with open('strike_ind_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_ind, f)

with open('strike_base_features.pkl', 'wb') as f:
    pickle.dump(base_features, f)

with open('strike_ind_features.pkl', 'wb') as f:
    pickle.dump(ind_features, f)

print("\nModel training and evaluation complete!")
print(f"Model saved to enhanced_strike_efficiency_model.h5")

# Create a prediction function for deployment
def predict_strike_efficiency(new_data, model_path, seq_scaler_path, ind_scaler_path, 
                             base_features_path, ind_features_path, seq_length=5):
    """
    Predict strike efficiency for new soldier data
    
    Parameters:
    new_data (DataFrame): DataFrame containing required features and sorted by timestamp
    model_path: Path to the saved model
    seq_scaler_path: Path to the sequence scaler
    ind_scaler_path: Path to the individual features scaler
    base_features_path: Path to the base features list
    ind_features_path: Path to the individual features list
    seq_length: Sequence length used in the model
    
    Returns:
    Array of predicted strike efficiencies
    """
    # Load model and preprocessing components
    model = tf.keras.models.load_model(model_path)
    
    with open(seq_scaler_path, 'rb') as f:
        seq_scaler = pickle.load(f)
    
    with open(ind_scaler_path, 'rb') as f:
        ind_scaler = pickle.load(f)
    
    with open(base_features_path, 'rb') as f:
        base_features = pickle.load(f)
    
    with open(ind_features_path, 'rb') as f:
        ind_features = pickle.load(f)
    
    # Process the data
    processed_data = new_data.copy()
    
    # Ensure timestamp is datetime
    if 'Timestamp' in processed_data.columns:
        processed_data['Timestamp'] = pd.to_datetime(processed_data['Timestamp'])
        
        # Extract time features if they don't exist
        if 'Hour_of_Day' not in processed_data.columns:
            processed_data['Hour_of_Day'] = processed_data['Timestamp'].dt.hour + processed_data['Timestamp'].dt.minute/60
        
        processed_data['Day_of_Week'] = processed_data['Timestamp'].dt.dayofweek
        processed_data['Is_Weekend'] = (processed_data['Day_of_Week'] >= 5).astype(int)
        
        # Create cyclical features
        processed_data['Hour_sin'] = np.sin(processed_data['Hour_of_Day'] * (2 * np.pi / 24))
        processed_data['Hour_cos'] = np.cos(processed_data['Hour_of_Day'] * (2 * np.pi / 24))
    
    # Create squad features if they don't exist
    if 'Squad_ID' not in processed_data.columns:
        # Assume every 10 soldiers form a squad
        processed_data['Squad_ID'] = processed_data.index // 10
    
    # Calculate derived features
    if 'Hydration_Index' not in processed_data.columns and all(f in processed_data.columns for f in ['Water_Content', 'Moisture']):
        processed_data['Hydration_Index'] = (processed_data['Water_Content'] / 100) * (processed_data['Moisture'] / 70)
    
    if 'Physio_Stress_Index' not in processed_data.columns and all(f in processed_data.columns for f in ['Heart_Rate', 'Respiration_Rate', 'Systolic_BP']):
        processed_data['Physio_Stress_Index'] = (
            ((processed_data['Heart_Rate'] - 60) / 60) + 
            ((processed_data['Respiration_Rate'] - 12) / 12) + 
            ((processed_data['Systolic_BP'] - 110) / 30)
        ) / 3
    
    # Calculate squad metrics if they don't exist
    required_squad_metrics = ['Squad_Efficiency', 'Squad_Heart_Rate', 'Squad_Fatigue', 'Squad_Hydration', 'Squad_Stress']
    if not all(metric in processed_data.columns for metric in required_squad_metrics):
        # Calculate squad-level metrics
        if 'Efficiency' in processed_data.columns:
            squad_metrics = processed_data.groupby('Squad_ID').agg({
                'Efficiency': 'mean',
                'Heart_Rate': 'mean',
                'Fatigue': 'mean',
                'Hydration_Index': 'mean',
                'Physio_Stress_Index': 'mean'
            }).rename(columns={
                'Efficiency': 'Squad_Efficiency',
                'Heart_Rate': 'Squad_Heart_Rate',
                'Fatigue': 'Squad_Fatigue',
                'Hydration_Index': 'Squad_Hydration',
                'Physio_Stress_Index': 'Squad_Stress'
            })
            
            # Merge squad metrics
            processed_data = processed_data.merge(squad_metrics, on='Squad_ID')
        else:
            raise ValueError("Efficiency data required to calculate squad metrics")
    
    # Check for missing features
    missing_base = [f for f in base_features if f not in processed_data.columns]
    missing_ind = [f for f in ind_features if f not in processed_data.columns]
    
    if missing_base or missing_ind:
        raise ValueError(f"Missing features: Base features: {missing_base}, Individual features: {missing_ind}")
    
    # Create sequences
    X_seq, _ = create_sequences(processed_data, base_features, 'Efficiency', seq_length=seq_length)
    
    # For individual features
    X_ind = np.zeros((X_seq.shape[0], len(ind_features)))
    
    # Populate individual features
    for i, squad in enumerate(processed_data['Squad_ID'].unique()):
        if len(processed_data[processed_data['Squad_ID'] == squad]) < seq_length:
            continue
            
        start_idx = i * (len(processed_data[processed_data['Squad_ID'] == squad]) - seq_length + 1)
        end_idx = start_idx + (len(processed_data[processed_data['Squad_ID'] == squad]) - seq_length + 1)
        
        for j in range(start_idx, end_idx):
            if j < len(X_ind):
                seq_end_idx = (j - start_idx) + seq_length - 1
                sqd_idx = processed_data[processed_data['Squad_ID'] == squad].index[seq_end_idx]
                X_ind[j] = processed_data.loc[sqd_idx, ind_features].values
    
    # Scale the data
    n_samples, n_timesteps, n_features = X_seq.shape
    X_seq_reshaped = X_seq.reshape(n_samples * n_timesteps, n_features)
    X_seq_scaled = seq_scaler.transform(X_seq_reshaped).reshape(n_samples, n_timesteps, n_features)
    
    X_ind_scaled = ind_scaler.transform(X_ind)
    
    # Predict
    predictions = model.predict([X_seq_scaled, X_ind_scaled]).flatten()
    
    return predictions

# Test the prediction function
print("\nTesting prediction function with sample data...")
# Create a sample sequence from the test data
sample_size = 5  # Use 5 samples
sample_seq = X_seq_test[:sample_size]
sample_ind = X_ind_test[:sample_size]
sample_y = y_test[:sample_size]

# Inverse transform the scaled data
sample_seq_reshaped = sample_seq.reshape(sample_size * seq_length, len(base_features))
sample_seq_orig = scaler_seq.inverse_transform(sample_seq_reshaped).reshape(sample_size, seq_length, len(base_features))
sample_ind_orig = scaler_ind.inverse_transform(sample_ind)

# Create a sample DataFrame
sample_dfs = []
for i in range(sample_size):
    # Create sequence of records
    for j in range(seq_length):
        sample_record = pd.DataFrame({feature: [sample_seq_orig[i, j, k]] for k, feature in enumerate(base_features)})
        
        # Add individual features from the last record in sequence
        if j == seq_length - 1:
            for k, feature in enumerate(ind_features):
                sample_record[feature] = sample_ind_orig[i, k]
        
        sample_record['Squad_ID'] = i
        sample_record['Timestamp'] = pd.Timestamp('2023-01-01') + pd.Timedelta(hours=j)
        
        # If we need Efficiency for squad metrics
        sample_record['Efficiency'] = 0.5  # Placeholder
        
        sample_dfs.append(sample_record)

sample_data = pd.concat(sample_dfs, ignore_index=True)

# Calculate squad metrics
squad_metrics = sample_data.groupby('Squad_ID').agg({
    'Efficiency': 'mean',
    'Heart_Rate': 'mean',
    'Fatigue': 'mean',
    'Hydration_Index': 'mean',
    'Physio_Stress_Index': 'mean'
}).rename(columns={
    'Efficiency': 'Squad_Efficiency',
    'Heart_Rate': 'Squad_Heart_Rate',
    'Fatigue': 'Squad_Fatigue',
    'Hydration_Index': 'Squad_Hydration',
    'Physio_Stress_Index': 'Squad_Stress'
})

# Merge squad metrics
sample_data = sample_data.merge(squad_metrics, on='Squad_ID')

# Make predictions using our function
sample_predictions = predict_strike_efficiency(
    sample_data,
    'enhanced_strike_efficiency_model.h5',
    'strike_seq_scaler.pkl',
    'strike_ind_scaler.pkl',
    'strike_base_features.pkl',
    'strike_ind_features.pkl',
    seq_length=seq_length
)

# Compare with actual values
print("\nSample predictions:")
for i, (actual, pred) in enumerate(zip(sample_y, sample_predictions)):
    print(f"Sample {i+1}: Actual = {actual:.4f}, Predicted = {pred:.4f}")

print("\nEnhanced Strike Efficiency Model pipeline complete!") 