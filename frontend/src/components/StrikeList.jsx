import React, { useContext } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext } from "../MainComponent";
import { getSoldierNameById } from "../constants/soldierNames";

function StrikeList() {
  const { solData } = useContext(AuthContext);
  const navigate = useNavigate();
  const getStatus = (efficiency) => {
    if (efficiency < 30) return "Danger";
    if (efficiency > 70) return "Stable";
    return "Watch";
  };

  // Ensure solData and efficiency_predictions exist before sorting
  const sortedSoldiers = solData?.efficiency_predictions
    ? [...solData.efficiency_predictions]
        .map((eff, index) => ({ id: index + 1, efficiency: eff }))
        .sort((a, b) => b.efficiency - a.efficiency) // Sort from high to low
    : [];

  return (
    <>
      <h2 className="ui-section-title text-white text-2xl font-bold mb-6">Soldier Efficiency Ranking</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sortedSoldiers.length > 0
          ? sortedSoldiers.map((soldier) => (
              <div
                className="cursor-pointer transform rounded-xl border border-emerald-500/12 bg-[linear-gradient(145deg,rgba(30,41,59,0.9),rgba(15,23,42,0.96))] p-4 shadow-lg transition-all duration-300 hover:border-emerald-400/35 hover:shadow-emerald-950/15 hover:scale-[1.02]"
                key={soldier.id}
                onClick={() => navigate(`/SingleSol/${soldier.id}`)}
              >
                <div className="flex items-center">
                  <div className={`flex items-center justify-center rounded-full w-12 h-12 mr-4 ${
                    soldier.efficiency < 30 
                      ? 'bg-gradient-to-br from-rose-500 to-rose-700 text-white' 
                      : soldier.efficiency > 70 
                        ? 'bg-gradient-to-br from-emerald-400 to-emerald-700 text-white' 
                        : 'bg-gradient-to-br from-amber-400 to-amber-600 text-white'
                  }`}>
                    {soldier.id}
                  </div>
                  <div className="flex-grow">
                    <div className="text-gray-lightest text-xl font-semibold">
                      {getSoldierNameById(soldier.id)}
                    </div>
                    <div className={`text-lg font-bold ${
                      soldier.efficiency < 30 
                        ? 'text-rose-300' 
                        : soldier.efficiency > 70 
                          ? 'text-emerald-300' 
                          : 'text-amber-300'
                    }`}>
                      Efficiency: {soldier.efficiency}%
                    </div>
                    <div className="mt-1">
                      <span className={`text-xs font-semibold px-2 py-1 rounded-md ${
                        soldier.efficiency < 30
                          ? "bg-[rgba(244,63,94,0.2)] text-rose-300"
                          : soldier.efficiency > 70
                            ? "bg-[rgba(16,185,129,0.2)] text-emerald-300"
                            : "bg-[rgba(245,158,11,0.2)] text-amber-300"
                      }`}>
                        {getStatus(soldier.efficiency)}
                      </span>
                    </div>
                  </div>
                  <div className="text-gray-light hover:text-white transition-colors duration-200 flex items-center">
                    <span className="mr-1">Details</span>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                    </svg>
                  </div>
                </div>
              </div>
            ))
          : <div className="text-gray-light text-xl p-6 text-center bg-[rgba(30,41,59,0.82)] rounded-xl border border-[rgba(110,231,183,0.2)]">Loading or No Data</div>
        }
      </div>
    </>
  );
}

export default StrikeList;
