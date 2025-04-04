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
        .sort((a, b) => b.efficiency - a.efficiency) // Sort from high to low
    : [];

  return (
    <>
      <h2 className="text-white text-2xl font-bold mb-6 border-b border-gray-dark pb-3">Soldier Efficiency Ranking</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sortedSoldiers.length > 0
          ? sortedSoldiers.map((soldier) => (
              <div
                className="bg-black-secondary rounded-xl p-4 border border-gray-dark hover:border-gray-light transition-all duration-300 shadow-lg transform hover:scale-105 cursor-pointer"
                key={soldier.id}
                onClick={() => navigate(`/SingleSol/${soldier.id}`)}
              >
                <div className="flex items-center">
                  <div className={`flex items-center justify-center rounded-full w-12 h-12 mr-4 ${
                    soldier.efficiency < 30 
                      ? 'bg-gradient-to-br from-red-800 to-red-600 text-white' 
                      : soldier.efficiency > 70 
                        ? 'bg-gradient-to-br from-green-800 to-green-600 text-white' 
                        : 'bg-gradient-to-br from-yellow-700 to-yellow-500 text-white'
                  }`}>
                    {soldier.id}
                  </div>
                  <div className="flex-grow">
                    <div className="text-gray-lightest text-xl font-semibold">
                      Soldier {soldier.id}
                    </div>
                    <div className={`text-lg font-bold ${
                      soldier.efficiency < 30 
                        ? 'text-red-400' 
                        : soldier.efficiency > 70 
                          ? 'text-green-400' 
                          : 'text-yellow-400'
                    }`}>
                      Efficiency: {soldier.efficiency}%
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
          : <div className="text-gray-light text-xl p-6 text-center bg-black-secondary rounded-xl border border-gray-dark">Loading or No Data</div>
        }
      </div>
    </>
  );
}

export default StrikeList;
