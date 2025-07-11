# Enhanced Soldier Efficiency Models

This directory contains improved deep learning models for predicting soldier efficiency metrics using time-series data with advanced physiological relationships.

## Models Overview

### 1. Enhanced Efficiency Model
- **File**: `Enhanced_Efficiency_Model.py`
- **Architecture**: Multi-branch neural network with specialized processing for vital signs, environmental factors, psychological metrics, and temporal information
- **Features**: Incorporates physiological correlations and time-based patterns (circadian rhythm, shift fatigue)
- **Performance**: Higher accuracy and better generalization compared to the basic model

### 2. Enhanced Strike Efficiency Model
- **File**: `Enhanced_Strike_Efficiency_Model.py`
- **Architecture**: Hybrid LSTM-CNN model for time-series analysis with multi-input branches
- **Features**: Incorporates squad-level metrics and temporal sequences for predicting strike efficiency
- **Key Innovation**: Captures how soldiers perform as part of a squad over time

## Key Improvements

1. **Physiological Realism**:
   - Heart rate naturally correlates with respiration rate
   - Blood pressure correlations with heart rate
   - Fatigue affects drowsiness and stress levels

2. **Time-Based Patterns**:
   - Circadian rhythm effects (performance typically drops in afternoon)
   - Shift-based fatigue modeling
   - Weekend vs. weekday performance differences

3. **Derived Features**:
   - Physiological Stress Index combining heart rate, respiration, and blood pressure
   - Hydration Index from water content and moisture levels
   - Squad-level aggregated metrics

4. **Advanced Model Architectures**:
   - Multi-branch neural networks for specialized feature processing
   - LSTM networks for time-series analysis
   - Residual connections and specialized feature representation

## Usage

### Efficiency Model

```python
from Enhanced_Models.Enhanced_Efficiency_Model import predict_efficiency
import pandas as pd

# Load your data
new_data = pd.read_csv("your_soldier_data.csv")

# Make predictions
predictions = predict_efficiency(
    new_data, 
    model_path='enhanced_efficiency_model.h5',
    scaler_path='efficiency_scaler.pkl',
    feature_list_path='feature_list.pkl'
)

# Use predictions
print(predictions)
```

### Strike Efficiency Model

```python
from Enhanced_Models.Enhanced_Strike_Efficiency_Model import predict_strike_efficiency
import pandas as pd

# Load your time-series data (must be sorted by timestamp)
new_data = pd.read_csv("your_squad_data.csv")

# Make predictions
predictions = predict_strike_efficiency(
    new_data,
    model_path='enhanced_strike_efficiency_model.h5',
    seq_scaler_path='strike_seq_scaler.pkl',
    ind_scaler_path='strike_ind_scaler.pkl',
    base_features_path='strike_base_features.pkl',
    ind_features_path='strike_ind_features.pkl',
    seq_length=5  # Sequence length used in training
)

# Use predictions
print(predictions)
```

## Required Data Format

### For Efficiency Model
The data should include these variables (at minimum):
- Vital signs: Temperature, SpO2, Heart_Rate, Respiration_Rate, Systolic_BP, Diastolic_BP
- Environmental: Moisture, Water_Content
- Psychological: Fatigue, Drowsiness, Stress
- Temporal: Timestamp or Hour_of_Day, Hours_into_Shift

### For Strike Efficiency Model
The data requires:
- All features needed for the Efficiency model
- Data sorted by timestamp
- At least 5 consecutive time points per squad (for sequence analysis)
- Squad_ID or data that can be grouped into squads

## Model Pipeline

1. Data preprocessing and feature engineering
2. Feature scaling
3. Training the multi-branch deep learning model
4. Model evaluation and performance visualization
5. Feature importance analysis
6. Prediction on new data

## Performance Metrics

The models are evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE) 
- Mean Absolute Error (MAE)
- R² Score

## Output Files

The scripts generate various output files in the `output` directory:
- **Training history plots**: Shows loss and MAE curves over epochs
- **Prediction results**: Scatter plots of predicted vs actual values
- **Feature importance**: Bar charts showing the most influential features
- **Error analysis**: Histograms and scatter plots of prediction errors

Model files and preprocessing components are saved in the main directory:
- Model files (`.h5`)
- Scalers (`.pkl`)
- Feature lists (`.pkl`) 