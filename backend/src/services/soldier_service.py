import numpy as np
import pandas as pd

from ..core.config import SOLDIER_COLUMNS
from ..core import state
from ..models.registry import model, scaler


def generate_soldier_data(num_soldiers=10):
    data = []
    for _ in range(num_soldiers):
        category = np.random.choice(
            ["low", "medium", "high"], p=[0.001, 0.4995, 0.4995]
        )

        if category == "low":
            temp = np.random.uniform(38, 40)
            moisture = np.random.uniform(10, 30)
            water_content = np.random.uniform(20, 40)
            sp_o2 = np.random.uniform(80, 90)
            fatigue = np.random.uniform(80, 100)
            drowsiness = np.random.uniform(70, 100)
            stress = np.random.uniform(70, 100)
            heart_rate = np.random.uniform(100, 130)
            respiration_rate = np.random.uniform(25, 35)
            systolic = np.random.randint(130, 140)
            diastolic = np.random.randint(85, 90)

        elif category == "medium":
            temp = np.random.uniform(36, 38)
            moisture = np.random.uniform(30, 50)
            water_content = np.random.uniform(40, 60)
            sp_o2 = np.random.uniform(90, 95)
            fatigue = np.random.uniform(40, 70)
            drowsiness = np.random.uniform(30, 60)
            stress = np.random.uniform(30, 60)
            heart_rate = np.random.uniform(80, 100)
            respiration_rate = np.random.uniform(18, 25)
            systolic = np.random.randint(115, 130)
            diastolic = np.random.randint(75, 85)

        else:
            temp = np.random.uniform(35, 36.5)
            moisture = np.random.uniform(50, 70)
            water_content = np.random.uniform(60, 80)
            sp_o2 = np.random.uniform(95, 100)
            fatigue = np.random.uniform(10, 40)
            drowsiness = np.random.uniform(10, 30)
            stress = np.random.uniform(10, 30)
            heart_rate = np.random.uniform(60, 80)
            respiration_rate = np.random.uniform(12, 18)
            systolic = np.random.randint(110, 120)
            diastolic = np.random.randint(70, 80)

        data.append(
            [
                temp,
                moisture,
                water_content,
                sp_o2,
                fatigue,
                drowsiness,
                stress,
                heart_rate,
                respiration_rate,
                systolic,
                diastolic,
            ]
        )

    state.soldier_data_df = pd.DataFrame(data, columns=SOLDIER_COLUMNS)
    scaled_data = scaler.transform(state.soldier_data_df)
    state.efficiency_predictions = [
        int(model.predict(row.reshape(1, -1)).flatten()[0] * 100) for row in scaled_data
    ]
