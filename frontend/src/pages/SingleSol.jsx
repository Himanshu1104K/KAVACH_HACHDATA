import { useState, useContext } from "react";
import { useParams } from "react-router-dom";
import NavBar from "../components/NavBar";
import CardUpper from "../components/CardUpper";
import DynamicGraph from "../components/DynamicGraph";
import PageTransition from "../components/PageTransition";
import { AuthContext } from "../MainComponent";

const SingleSol = () => {
  const { id } = useParams(); // Get soldier ID from route
  const soldierIndex = parseInt(id) === 0 ? 0 : parseInt(id) - 1; // Convert to zero-based index
  const strIndex = soldierIndex.toString(); // Convert to string for API data format
  const { solData } = useContext(AuthContext);
  const [selectedMetric, setSelectedMetric] = useState("Heart_Rate"); // Default metric

  // Ensure solData exists before accessing
  if (
    !solData?.efficiency_predictions ||
    !solData?.soldier_data ||
    isNaN(soldierIndex)
  ) {
    return (
      <div className="text-center text-2xl mt-24 pt-10 transition-opacity duration-1000 opacity-70">
        Loading or No Data Found
      </div>
    );
  }

  const handleCardClick = (metric) => {
    setSelectedMetric(metric);
  };

  // Helper function to get value from soldier data with proper formatting
  const getSoldierValue = (metricName) => {
    try {
      if (metricName === "efficiency_predictions") {
        return solData.efficiency_predictions[soldierIndex] ?? "N/A";
      } else {
        const value = solData.soldier_data[metricName]?.[strIndex];
        return value !== undefined ? Math.floor(value) : "N/A";
      }
    } catch (error) {
      console.error(`Error getting value for ${metricName}:`, error);
      return "N/A";
    }
  };

  return (
    <div className="min-h-screen bg-black-primary">
      <NavBar />
      <div className="container mx-auto px-6 pt-24 pb-8">
        <PageTransition>
          <h2 className="mainDH text-4xl text-center mb-10 text-white font-bold">
            SINGLE SOLDIER DASHBOARD
          </h2>
          
          <div className="flex flex-col lg:flex-row gap-8 mb-10">
            {/* Left side - Vital metrics grid */}
            <div className="w-full lg:w-2/5 grid grid-cols-2 gap-4">
              <div 
                className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
                onClick={() => handleCardClick("efficiency_predictions")}
              >
                <CardUpper
                  title="Efficiency"
                  value={getSoldierValue("efficiency_predictions")}
                />
              </div>
              <div 
                className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
                onClick={() => handleCardClick("Temperature")}
              >
                <CardUpper
                  title="Temperature"
                  value={getSoldierValue("Temperature")}
                />
              </div>
              <div 
                className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
                onClick={() => handleCardClick("Moisture")}
              >
                <CardUpper
                  title="Moisture"
                  value={getSoldierValue("Moisture")}
                />
              </div>
              <div 
                className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
                onClick={() => handleCardClick("Water_Content")}
              >
                <CardUpper
                  title="Water Content"
                  value={getSoldierValue("Water_Content")}
                />
              </div>
              <div 
                className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
                onClick={() => handleCardClick("SpO2")}
              >
                <CardUpper
                  title="SpO2"
                  value={getSoldierValue("SpO2")}
                />
              </div>
              <div 
                className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
                onClick={() => handleCardClick("Fatigue")}
              >
                <CardUpper
                  title="Fatigue"
                  value={getSoldierValue("Fatigue")}
                />
              </div>
            </div>

            {/* Right side - Graph */}
            <div className="w-full lg:w-3/5">
              <DynamicGraph selectedMetric={selectedMetric} />
            </div>
          </div>

          {/* Bottom metrics row */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mt-8">
            <div 
              className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
              onClick={() => handleCardClick("Drowsiness")}
            >
              <CardUpper
                title="Drowsiness"
                value={getSoldierValue("Drowsiness")}
              />
            </div>
            <div 
              className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
              onClick={() => handleCardClick("Heart_Rate")}
            >
              <CardUpper
                title="Heart Rate"
                value={getSoldierValue("Heart_Rate")}
              />
            </div>
            <div 
              className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
              onClick={() => handleCardClick("Stress")}
            >
              <CardUpper
                title="Stress"
                value={getSoldierValue("Stress")}
              />
            </div>
            <div 
              className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
              onClick={() => handleCardClick("Respiration_Rate")}
            >
              <CardUpper
                title="Respiration Rate"
                value={getSoldierValue("Respiration_Rate")}
              />
            </div>
            <div 
              className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
              onClick={() => handleCardClick("Systolic_BP")}
            >
              <CardUpper
                title="Systolic BP"
                value={getSoldierValue("Systolic_BP")}
              />
            </div>
            <div 
              className="cursor-pointer transform hover:scale-105 transition-transform duration-200"
              onClick={() => handleCardClick("Diastolic_BP")}
            >
              <CardUpper
                title="Diastolic BP"
                value={getSoldierValue("Diastolic_BP")}
              />
            </div>
          </div>
        </PageTransition>
      </div>
    </div>
  );
};

export default SingleSol;
