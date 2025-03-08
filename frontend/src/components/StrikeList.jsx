import React, { useContext } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext } from "../MainComponent";

function StrikeList() {
  const { solData } = useContext(AuthContext);
  const navigate = useNavigate();

  // Ensure solData and efficiency_predictions exist before sorting
  const sortedSoldiers = solData?.efficiency_predictions
    ? [...solData.efficiency_predictions]
        .map((eff, index) => ({ id: index + 1, efficiency: eff }))
        .sort((a, b) => a.efficiency - b.efficiency)
    : [];

  return (
    <>
      {sortedSoldiers.length > 0
        ? sortedSoldiers.map((soldier) => (
            <div
              className="mainDH text-2xl tracking-widest bg-[#504b385c] justify-center rounded-xl p-5 m-3 text-[#504b38] cursor-pointer hover:bg-[#504b38] hover:text-white"
              key={soldier.id}
              onClick={() => navigate(`/SingleSol/${soldier.id}`)}
            >
              Soldier {soldier.id}: Efficiency {soldier.efficiency}
            </div>
          ))
        : "Loading or No Data"}
    </>
  );
}

export default StrikeList;
