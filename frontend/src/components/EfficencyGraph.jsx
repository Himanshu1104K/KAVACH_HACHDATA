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

const EfficencyGraph = () => {
  const { solData } = useContext(AuthContext);
  
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "#FFFFFF",
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
        color: "#FFFFFF",
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
          color: "rgba(187, 187, 187, 0.1)",
          lineWidth: 1
        },
        ticks: {
          color: "#E0E0E0",
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
          color: "rgba(187, 187, 187, 0.1)",
          lineWidth: 1
        },
        ticks: {
          color: "#E0E0E0",
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
    labels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(num => `Soldier ${num}`),
    datasets: [
      {
        label: "Efficiency (%)",
        data: solData?.efficiency_predictions || [],
        backgroundColor: solData?.efficiency_predictions.map((eff) => {
          return eff < 30 ? "rgba(248, 113, 113, 0.85)" : // More vibrant red
                 eff > 70 ? "rgba(74, 222, 128, 0.85)" : // More vibrant green
                 "rgba(250, 204, 21, 0.85)"; // More vibrant yellow
        }),
        borderColor: solData?.efficiency_predictions.map((eff) => {
          return eff < 30 ? "rgba(220, 38, 38, 1)" : // Red border
                 eff > 70 ? "rgba(22, 163, 74, 1)" : // Green border
                 "rgba(202, 138, 4, 1)"; // Yellow border
        }),
        borderWidth: 2,
        borderRadius: 6,
        hoverBackgroundColor: solData?.efficiency_predictions.map((eff) => {
          return eff < 30 ? "rgba(248, 113, 113, 1)" : 
                 eff > 70 ? "rgba(74, 222, 128, 1)" : 
                 "rgba(250, 204, 21, 1)";
        }),
        barPercentage: 0.7,
        categoryPercentage: 0.8,
      },
    ],
  };
  
  return (
    <div className="bg-gradient-to-br from-black-secondary to-gray-dark rounded-xl p-6 shadow-xl border border-gray-dark hover:border-gray-medium transition-all duration-300">
      <h3 className="text-white text-xl font-semibold mb-6 border-b border-gray-dark pb-3">Soldiers Performance Analytics</h3>
      <div className="h-[350px] w-full">
        <Bar options={options} data={data} />
      </div>
    </div>
  );
};

export default EfficencyGraph;
