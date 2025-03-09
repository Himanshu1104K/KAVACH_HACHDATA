import numpy as np
import pandas as pd
import random

def generate_soldier_dataset(size=10, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    efficiency_predictions = np.random.randint(15, 80, size)  # Efficiency bounded realistically
    temperature = np.round(np.random.uniform(36.0, 37.8, size), 1)  # Normal human body temp
    moisture = np.round(np.random.uniform(35.0, 65.0, size), 1)  # Skin moisture range
    water_content = np.round(np.random.uniform(45.0, 75.0, size), 1)  # Hydration levels
    spo2 = np.round(np.random.uniform(92.0, 100.0, size), 1)  # Normal SpO2 levels
    fatigue = np.round(np.random.uniform(20.0, 65.0, size), 1)  # Fatigue score based on activity
    drowsiness = np.round(np.random.uniform(15.0, 50.0, size), 1)  # Sleepiness score
    stress = np.round(np.random.uniform(15.0, 50.0, size), 1)  # Stress levels under operation
    heart_rate = np.round(np.random.uniform(60.0, 100.0, size), 1)  # Normal HR range
    respiration_rate = np.round(np.random.uniform(12.0, 22.0, size), 1)  # Normal RR range
    x = np.random.randint(0, 100, size)  # Position coordinates
    y = np.random.randint(0, 100, size)
    
    def determine_label(efficiency, fatigue, stress):
        if efficiency < 30 or fatigue > 55 or stress > 45:
            return "Defensive Position - Regroup and Hold"
        elif 30 <= efficiency < 60 and fatigue < 50 and stress < 40:
            return "Balanced Formation - Hold and Assess"
        else:
            return "Fierce Attack - Charge Forward!"
    
    label = [determine_label(eff, fat, st) for eff, fat, st in zip(efficiency_predictions, fatigue, stress)]
    
    data = {
        "efficiency_predictions": efficiency_predictions,
        "Temperature": temperature,
        "Moisture": moisture,
        "Water_Content": water_content,
        "SpO2": spo2,
        "Fatigue": fatigue,
        "Drowsiness": drowsiness,
        "Stress": stress,
        "Heart_Rate": heart_rate,
        "Respiration_Rate": respiration_rate,
        "x": x,
        "y": y,
        "label": label
    }
    
    df = pd.DataFrame(data)
    return df

# Generate dataset and save to CSV
dataset = generate_soldier_dataset(10)
dataset.to_csv("soldier_health_data.csv", index=False)
print("Dataset saved as 'soldier_health_data.csv'")
