import { useState, useContext } from "react";
import { useParams } from "react-router-dom";
import NavBar from "../components/NavBar";
import CardUpper from "../components/CardUpper";
import DynamicGraph from "../components/DynamicGraph";
import { AuthContext } from "../MainComponent";

const SingleSol = () => {
  const { id } = useParams(); // Get soldier ID from route
  const soldierIndex = parseInt(id) == 0 ? 0 : parseInt(id) - 1; // Convert to zero-based index
  const { solData } = useContext(AuthContext);
  const [selectedMetric, setSelectedMetric] = useState("Temperature"); // Default metric

  // Ensure solData exists before accessing
  if (
    !solData?.efficiency_predictions ||
    !solData?.soldier_data ||
    isNaN(soldierIndex)
  ) {
    return (
      <div className="text-center text-2xl mt-10">Loading or No Data Found</div>
    );
  }

  const handleCardClick = (metric) => {
    setSelectedMetric(metric);
  };

  return (
    <div>
      <NavBar />
      <h2 className="mainDH text-4xl text-center m-5">
        Single Soldier Dashboard
      </h2>
      <div className="grid grid-cols-2 mb-5">
        <div className="w-[30vw] grid grid-cols-2 gap-3 m-5 ml-20">
          <div onClick={() => handleCardClick("efficiency_predictions")}>
            <CardUpper
              title="Efficiency"
              value={solData.efficiency_predictions[soldierIndex] ?? "N/A"}
            />
          </div>
          <div onClick={() => handleCardClick("Temperature")}>
            <CardUpper
              title="Temperature"
              value={Math.floor(
                solData.soldier_data["Temperature"]?.[soldierIndex] || 0
              )}
            />
          </div>
          <div onClick={() => handleCardClick("Moisture")}>
            <CardUpper
              title="Moisture"
              value={Math.floor(
                solData.soldier_data["Moisture"]?.[soldierIndex] || 0
              )}
            />
          </div>
          <div onClick={() => handleCardClick("Water_Content")}>
            <CardUpper
              title="Water Content"
              value={Math.floor(
                solData.soldier_data["Water_Content"]?.[soldierIndex] || 0
              )}
            />
          </div>
          <div onClick={() => handleCardClick("SpO2")}>
            <CardUpper
              title="SpO2"
              value={Math.floor(
                solData.soldier_data["SpO2"]?.[soldierIndex] || 0
              )}
            />
          </div>
          <div onClick={() => handleCardClick("Fatigue")}>
            <CardUpper
              title="Fatigue"
              value={Math.floor(
                solData.soldier_data["Fatigue"]?.[soldierIndex] || 0
              )}
            />
          </div>
        </div>
        <div className="absolute top-[180px] right-[70px] w-[52vw]">
          <DynamicGraph selectedMetric={selectedMetric} />
        </div>
      </div>
      <div className="grid grid-cols-6 p-5 ml-15 mr-8">
        <div onClick={() => handleCardClick("Drowsiness")}>
          <CardUpper
            title="Drowsiness"
            value={Math.floor(
              solData.soldier_data["Drowsiness"]?.[soldierIndex] || 0
            )}
          />
        </div>
        <div onClick={() => handleCardClick("Heart_Rate")}>
          <CardUpper
            title="Heart Rate"
            value={Math.floor(
              solData.soldier_data["Heart_Rate"]?.[soldierIndex] || 0
            )}
          />
        </div>
        <div onClick={() => handleCardClick("Stress")}>
          <CardUpper
            title="Stress"
            value={Math.floor(
              solData.soldier_data["Stress"]?.[soldierIndex] || 0
            )}
          />
        </div>
        <div onClick={() => handleCardClick("Respiration_Rate")}>
          <CardUpper
            title="Respiration Rate"
            value={Math.floor(
              solData.soldier_data["Respiration_Rate"]?.[soldierIndex] || 0
            )}
          />
        </div>
        <div onClick={() => handleCardClick("Systolic_BP")}>
          <CardUpper
            title="Systolic BP"
            value={Math.floor(
              solData.soldier_data["Systolic_BP"]?.[soldierIndex] || 0
            )}
          />
        </div>
        <div onClick={() => handleCardClick("Diastolic_BP")}>
          <CardUpper
            title="Diastolic BP"
            value={Math.floor(
              solData.soldier_data["Diastolic_BP"]?.[soldierIndex] || 0
            )}
          />
        </div>
      </div>
    </div>
  );
};

export default SingleSol;
