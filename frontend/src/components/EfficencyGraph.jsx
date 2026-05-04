import React from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);
import { AuthContext } from "../MainComponent";
import { useContext } from "react";
import { getSoldierNameById } from "../constants/soldierNames";

const EfficencyGraph = () => {
  const { solData } = useContext(AuthContext);
  
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "#d1fae5",
          font: {
            size: 13,
            weight: "bold",
            family: "'Inter', sans-serif"
          },
          boxWidth: 15,
          padding: 20
        }
      },
      title: {
        display: true,
        text: "Soldier Efficiency",
        color: "#d1fae5",
        font: {
          size: 18,
          weight: "bold",
          family: "'Inter', sans-serif"
        },
        padding: {
          bottom: 20
        }
      },
      tooltip: {
        backgroundColor: "rgba(30, 30, 30, 0.9)",
        titleFont: {
          size: 14,
          weight: "bold"
        },
        bodyFont: {
          size: 13
        },
        padding: 10,
        cornerRadius: 6,
        displayColors: false,
        callbacks: {
          label: function(context) {
            return `Efficiency: ${context.raw}%`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: "rgba(16, 185, 129, 0.16)",
          lineWidth: 1
        },
        ticks: {
          color: "#d1fae5",
          font: {
            size: 12
          },
          padding: 10
        },
        border: {
          display: false
        }
      },
      y: {
        grid: {
          color: "rgba(16, 185, 129, 0.16)",
          lineWidth: 1
        },
        ticks: {
          color: "#d1fae5",
          font: {
            size: 12
          },
          padding: 10
        },
        border: {
          display: false
        },
        beginAtZero: true,
        max: 100
      }
    },
    animation: {
      duration: 1000,
      easing: 'easeOutQuart'
    }
  };
  
  const data = {
    labels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((num) => getSoldierNameById(num)),
    datasets: [
      {
        label: "Efficiency (%)",
        data: solData?.efficiency_predictions || [],
        backgroundColor: solData?.efficiency_predictions.map((eff) => {
          return eff < 30 ? "rgba(251, 113, 133, 0.84)" :
                 eff > 70 ? "rgba(52, 211, 153, 0.84)" :
                 "rgba(250, 204, 21, 0.84)";
        }),
        borderColor: solData?.efficiency_predictions.map((eff) => {
          return eff < 30 ? "rgba(225, 29, 72, 1)" :
                 eff > 70 ? "rgba(5, 150, 105, 1)" :
                 "rgba(202, 138, 4, 1)";
        }),
        borderWidth: 2,
        borderRadius: 6,
        hoverBackgroundColor: solData?.efficiency_predictions.map((eff) => {
          return eff < 30 ? "rgba(251, 113, 133, 1)" : 
                 eff > 70 ? "rgba(52, 211, 153, 1)" : 
                 "rgba(250, 204, 21, 1)";
        }),
        barPercentage: 0.7,
        categoryPercentage: 0.8,
      },
    ],
  };
  
  return (
    <div className="ui-panel p-6 transition-all duration-300 hover:border-emerald-500/30">
      <h3 className="ui-section-title text-white text-xl font-semibold mb-6">Soldiers Performance Analytics</h3>
      <div className="h-[350px] w-full">
        <Bar options={options} data={data} />
      </div>
    </div>
  );
};

export default EfficencyGraph;
