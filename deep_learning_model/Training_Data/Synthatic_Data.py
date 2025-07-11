import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples in each efficiency category
num_samples_per_group = 1700  # Total ~5100 records

# Define physiologically plausible correlations
def add_realistic_noise(data, noise_level=0.05):
    """Add small amount of realistic noise to data"""
    return data + np.random.normal(0, noise_level * np.std(data), size=data.shape)

def generate_correlated_vitals(base_heart_rate, num_samples):
    """Generate physiologically correlated vital signs"""
    # Heart rate has natural correlation with respiration rate (~4:1 ratio)
    respiration_rate = base_heart_rate / 4 + np.random.normal(0, 1, num_samples)
    
    # Blood pressure correlations
    systolic = 100 + (base_heart_rate * 0.3) + np.random.normal(0, 5, num_samples)
    diastolic = systolic * 0.65 + np.random.normal(0, 3, num_samples)
    
    return respiration_rate, systolic, diastolic

# Create timestamps for time-based patterns (e.g., circadian rhythm effects)
start_date = datetime(2023, 1, 1)
timestamps = [start_date + timedelta(minutes=30*i) for i in range(num_samples_per_group*3)]

# **1. Generate Low Efficiency Data**
# Core temperature slightly elevated due to exertion/dehydration
low_temperature = np.random.uniform(36.8, 38.2, num_samples_per_group)
low_temperature = add_realistic_noise(low_temperature, 0.03)

# Low moisture and water content (dehydration)
low_moisture = np.random.uniform(30, 50, num_samples_per_group)
# Correlation between moisture and water content
low_water_content = 35 + (low_moisture * 0.5) + np.random.normal(0, 3, num_samples_per_group)
low_water_content = np.clip(low_water_content, 40, 60)

# Low SpO2 due to exertion
low_spo2 = np.random.uniform(85, 90, num_samples_per_group)
low_spo2 = add_realistic_noise(low_spo2, 0.02)

# High fatigue and related factors
low_fatigue = np.random.beta(7, 3, num_samples_per_group)  # Beta distribution for more realistic distribution
# Correlation between fatigue and drowsiness
low_drowsiness = low_fatigue * 0.8 + np.random.beta(5, 3, num_samples_per_group) * 0.2
# Correlation between fatigue and stress
low_stress = low_fatigue * 0.7 + np.random.beta(8, 3, num_samples_per_group) * 0.3

# Elevated heart rate due to exertion/stress
low_heart_rate = np.random.uniform(110, 140, num_samples_per_group)
# Generate correlated vitals
low_respiration_rate, low_systolic, low_diastolic = generate_correlated_vitals(low_heart_rate, num_samples_per_group)
low_systolic = np.clip(low_systolic, 130, 160).astype(int)
low_diastolic = np.clip(low_diastolic, 85, 100).astype(int)
low_respiration_rate = np.clip(low_respiration_rate, 20, 30)

# Add some circadian rhythm effect - worst in afternoon heat
time_of_day = np.array([(t.hour + t.minute/60) for t in timestamps[:num_samples_per_group]])
circadian_effect = 0.05 * np.sin((time_of_day - 14) * np.pi / 12)  # Worst at 2 PM
low_efficiency_base = (
    (0.3 * low_spo2 / 100 + 0.2 * low_water_content / 100) -
    (0.3 * low_fatigue + 0.2 * low_drowsiness + 0.25 * low_stress) -
    (0.15 * (low_heart_rate - 80) / 60) - 
    (0.1 * (low_systolic - 120) / 30)
)
low_efficiency = low_efficiency_base - circadian_effect

# **2. Generate Medium Efficiency Data**
med_temperature = np.random.uniform(36.2, 37.8, num_samples_per_group)
med_temperature = add_realistic_noise(med_temperature, 0.03)

med_moisture = np.random.uniform(40, 60, num_samples_per_group)
# Correlation between moisture and water content
med_water_content = 45 + (med_moisture * 0.6) + np.random.normal(0, 3, num_samples_per_group)
med_water_content = np.clip(med_water_content, 50, 80)

med_spo2 = np.random.uniform(90, 96, num_samples_per_group)
med_spo2 = add_realistic_noise(med_spo2, 0.01)

med_fatigue = np.random.beta(5, 5, num_samples_per_group)  # Centered around 0.5
# Correlation between fatigue and drowsiness
med_drowsiness = med_fatigue * 0.8 + np.random.beta(4, 4, num_samples_per_group) * 0.2
# Correlation between fatigue and stress
med_stress = med_fatigue * 0.7 + np.random.beta(5, 4, num_samples_per_group) * 0.3

med_heart_rate = np.random.uniform(80, 110, num_samples_per_group)
# Generate correlated vitals
med_respiration_rate, med_systolic, med_diastolic = generate_correlated_vitals(med_heart_rate, num_samples_per_group)
med_systolic = np.clip(med_systolic, 115, 130).astype(int)
med_diastolic = np.clip(med_diastolic, 75, 85).astype(int)
med_respiration_rate = np.clip(med_respiration_rate, 16, 25)

# Medium circadian effect
time_of_day = np.array([(t.hour + t.minute/60) for t in timestamps[num_samples_per_group:2*num_samples_per_group]])
circadian_effect = 0.03 * np.sin((time_of_day - 14) * np.pi / 12)  # Moderate effect at 2 PM
med_efficiency_base = (
    (0.3 * med_spo2 / 100 + 0.2 * med_water_content / 100) -
    (0.3 * med_fatigue + 0.2 * med_drowsiness + 0.25 * med_stress) -
    (0.15 * (med_heart_rate - 80) / 60) -
    (0.1 * (med_systolic - 120) / 30)
)
med_efficiency = med_efficiency_base - circadian_effect

# **3. Generate High Efficiency Data**
high_temperature = np.random.uniform(36.0, 37.2, num_samples_per_group)
high_temperature = add_realistic_noise(high_temperature, 0.02)

high_moisture = np.random.uniform(50, 70, num_samples_per_group)
# Correlation between moisture and water content
high_water_content = 60 + (high_moisture * 0.5) + np.random.normal(0, 2, num_samples_per_group)
high_water_content = np.clip(high_water_content, 70, 90)

high_spo2 = np.random.uniform(96, 100, num_samples_per_group)
high_spo2 = add_realistic_noise(high_spo2, 0.005)

high_fatigue = np.random.beta(3, 7, num_samples_per_group)  # Beta distribution skewed toward lower values
# Correlation between fatigue and drowsiness
high_drowsiness = high_fatigue * 0.8 + np.random.beta(3, 9, num_samples_per_group) * 0.2
# Correlation between fatigue and stress
high_stress = high_fatigue * 0.7 + np.random.beta(2, 8, num_samples_per_group) * 0.3

high_heart_rate = np.random.uniform(60, 90, num_samples_per_group)
# Generate correlated vitals
high_respiration_rate, high_systolic, high_diastolic = generate_correlated_vitals(high_heart_rate, num_samples_per_group)
high_systolic = np.clip(high_systolic, 110, 125).astype(int)
high_diastolic = np.clip(high_diastolic, 70, 80).astype(int)
high_respiration_rate = np.clip(high_respiration_rate, 12, 20)

# Minimal circadian effect for high efficiency
time_of_day = np.array([(t.hour + t.minute/60) for t in timestamps[2*num_samples_per_group:3*num_samples_per_group]])
circadian_effect = 0.01 * np.sin((time_of_day - 14) * np.pi / 12)  # Minimal effect
high_efficiency_base = (
    (0.3 * high_spo2 / 100 + 0.2 * high_water_content / 100) -
    (0.3 * high_fatigue + 0.2 * high_drowsiness + 0.25 * high_stress) -
    (0.15 * (high_heart_rate - 80) / 60) -
    (0.1 * (high_systolic - 120) / 30)
)
high_efficiency = high_efficiency_base - circadian_effect

# Normalize to range [0.1, 0.9]
low_efficiency = np.interp(low_efficiency, (low_efficiency.min(), low_efficiency.max()), (0.1, 0.3))
med_efficiency = np.interp(med_efficiency, (med_efficiency.min(), med_efficiency.max()), (0.3, 0.6))
high_efficiency = np.interp(high_efficiency, (high_efficiency.min(), high_efficiency.max()), (0.6, 0.9))

# Add timestamps to the data
low_timestamps = timestamps[:num_samples_per_group]
med_timestamps = timestamps[num_samples_per_group:2*num_samples_per_group]
high_timestamps = timestamps[2*num_samples_per_group:3*num_samples_per_group]

# **Combine Data**
df_low = pd.DataFrame({
    "Timestamp": low_timestamps,
    "Temperature": low_temperature, "Moisture": low_moisture, "Water_Content": low_water_content,
    "SpO2": low_spo2, "Fatigue": low_fatigue, "Drowsiness": low_drowsiness, "Stress": low_stress,
    "Heart_Rate": low_heart_rate, "Respiration_Rate": low_respiration_rate, 
    "Systolic_BP": low_systolic, "Diastolic_BP": low_diastolic, "Efficiency": low_efficiency
})

df_med = pd.DataFrame({
    "Timestamp": med_timestamps,
    "Temperature": med_temperature, "Moisture": med_moisture, "Water_Content": med_water_content,
    "SpO2": med_spo2, "Fatigue": med_fatigue, "Drowsiness": med_drowsiness, "Stress": med_stress,
    "Heart_Rate": med_heart_rate, "Respiration_Rate": med_respiration_rate, 
    "Systolic_BP": med_systolic, "Diastolic_BP": med_diastolic, "Efficiency": med_efficiency
})

df_high = pd.DataFrame({
    "Timestamp": high_timestamps,
    "Temperature": high_temperature, "Moisture": high_moisture, "Water_Content": high_water_content,
    "SpO2": high_spo2, "Fatigue": high_fatigue, "Drowsiness": high_drowsiness, "Stress": high_stress,
    "Heart_Rate": high_heart_rate, "Respiration_Rate": high_respiration_rate, 
    "Systolic_BP": high_systolic, "Diastolic_BP": high_diastolic, "Efficiency": high_efficiency
})

# Merge All Data
df_final = pd.concat([df_low, df_med, df_high], ignore_index=True)

# Add some outliers (0.5% of data)
outlier_indices = np.random.choice(len(df_final), size=int(0.005 * len(df_final)), replace=False)
for idx in outlier_indices:
    # Randomly choose which feature to create an outlier for
    feature = np.random.choice(['Temperature', 'SpO2', 'Heart_Rate', 'Systolic_BP'])
    if feature == 'Temperature':
        df_final.loc[idx, 'Temperature'] = np.random.uniform(38.5, 40)  # Fever
    elif feature == 'SpO2':
        df_final.loc[idx, 'SpO2'] = np.random.uniform(80, 85)  # Very low oxygen
    elif feature == 'Heart_Rate':
        df_final.loc[idx, 'Heart_Rate'] = np.random.uniform(140, 160)  # Tachycardia
    elif feature == 'Systolic_BP':
        df_final.loc[idx, 'Systolic_BP'] = np.random.uniform(160, 180)  # Hypertension
        df_final.loc[idx, 'Diastolic_BP'] = df_final.loc[idx, 'Systolic_BP'] * 0.65  # Maintain relationship

# Recalculate efficiency for outliers to maintain physiological relationships
for idx in outlier_indices:
    row = df_final.loc[idx]
    new_efficiency = (
        (0.3 * row['SpO2'] / 100 + 0.2 * row['Water_Content'] / 100) -
        (0.3 * row['Fatigue'] + 0.2 * row['Drowsiness'] + 0.25 * row['Stress']) -
        (0.15 * (row['Heart_Rate'] - 80) / 60) - 
        (0.1 * (row['Systolic_BP'] - 120) / 30)
    )
    # Keep within the original category's range
    if row['Efficiency'] <= 0.3:
        df_final.loc[idx, 'Efficiency'] = np.clip(new_efficiency, 0.1, 0.3)
    elif row['Efficiency'] <= 0.6:
        df_final.loc[idx, 'Efficiency'] = np.clip(new_efficiency, 0.3, 0.6)
    else:
        df_final.loc[idx, 'Efficiency'] = np.clip(new_efficiency, 0.6, 0.9)

# Add feature: time since shift started (in hours)
df_final['Hour_of_Day'] = df_final['Timestamp'].apply(lambda x: x.hour + x.minute/60)
# Create shift patterns (8-hour shifts)
df_final['Hours_into_Shift'] = df_final['Timestamp'].apply(
    lambda x: (x.hour % 8) + x.minute/60
)

# Shuffle the data
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Save as CSV
df_final.to_csv("realistic_soldier_data.csv", index=False)

# Display Summary
print("Data Summary Statistics:")
print(df_final["Efficiency"].describe())
print("\nCorrelation Matrix:")
print(df_final[['Temperature', 'Water_Content', 'SpO2', 'Fatigue', 'Heart_Rate', 'Efficiency']].corr())

# Display first few rows
print("\nSample Data:")
print(df_final.head())