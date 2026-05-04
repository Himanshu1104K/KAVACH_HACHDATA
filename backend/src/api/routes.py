from fastapi import APIRouter, HTTPException

from ..core import state
from ..services.soldier_service import generate_soldier_data
from ..services.strike_service import get_strike_efficiency_prediction
from ..services.tactics_service import get_soldier_tactics_formation

router = APIRouter()


@router.get("/")
def main_dashboard():
    generate_soldier_data()
    return {
        "efficiency_predictions": state.efficiency_predictions,
        "soldier_data": state.soldier_data_df.to_dict(),
    }


@router.get("/soldier/{index}")
def get_soldier_details(index: int):
    if state.soldier_data_df is None or state.efficiency_predictions is None:
        generate_soldier_data()

    if index < 0 or index >= len(state.soldier_data_df):
        raise HTTPException(status_code=404, detail="Soldier index out of range")

    soldier_info = state.soldier_data_df.iloc[index].to_dict()
    efficiency = state.efficiency_predictions[index]
    return {
        "soldier_index": index,
        "efficiency": efficiency,
        "health_metrics": soldier_info,
    }


@router.get("/strike_efficiency")
def get_strike_efficiency():
    return {"strike_success_probability": get_strike_efficiency_prediction()}


@router.get("/soldier_tacktics")
def get_soldier_tactics():
    return {"formation": get_soldier_tactics_formation()}
