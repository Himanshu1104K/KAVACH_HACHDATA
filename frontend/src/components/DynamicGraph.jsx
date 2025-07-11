import { useContext, useState, useEffect } from "react";
import { AuthContext } from "../MainComponent";
import { useParams } from "react-router-dom";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const DynamicGraph = ({ selectedMetric }) => {
  const { id } = useParams();
  const soldierIndex = parseInt(id) === 0 ? 0 : parseInt(id) - 1;
  const { solData } = useContext(AuthContext);
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (solData !== undefined) {
      try {
        setChartData((prevData) => {
          let newValue;
          
          // Special handling for efficiency_predictions which is at the root level
          if (selectedMetric === "efficiency_predictions") {
            newValue = solData.efficiency_predictions[soldierIndex];
          } 
          // Handle all other metrics which are in soldier_data object with nested structure
          else if (solData.soldier_data && solData.soldier_data[selectedMetric]) {
            // Convert soldierIndex to string since the API keys are strings
            const strIndex = soldierIndex.toString();
            newValue = Math.floor(solData.soldier_data[selectedMetric][strIndex] || 0);
          } else {
            newValue = 0;
          }
          
          const newData = [...prevData, newValue];
          if (newData.length > 10) newData.shift();
          return newData;
        });
      } catch (error) {
        console.error("Error updating chart data:", error);
        setChartData([0]); // Fallback to prevent UI breaking
      }
    }
  }, [solData, selectedMetric, soldierIndex]);

  // Generate labels for x-axis
  const labels = chartData.map((_, index) => index + 1);

  // Format the metric name for display
  const formatMetricName = (metric) => {
    if (metric === "efficiency_predictions") return "Efficiency";
    return metric.replace(/_/g, ' ');
  };

  const data = {
    labels: labels,
    datasets: [
      {
        label: formatMetricName(selectedMetric),
        data: chartData,
        fill: true,
        backgroundColor: "rgba(96, 108, 56, 0.25)",
        borderColor: "#606C38",
        borderWidth: 3,
        tension: 0.4,
        pointBackgroundColor: "#FEFAE0",
        pointBorderColor: "#606C38",
        pointRadius: 5,
        pointHoverRadius: 7,
        pointBorderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { 
        position: "top",
        labels: {
          color: "#FEFAE0",
          font: {
            size: 13,
            weight: "bold"
          },
          boxWidth: 15,
          padding: 20
        }
      },
      title: { 
        display: true, 
        text: `${formatMetricName(selectedMetric)} Over Time`,
        color: "#FEFAE0",
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
        backgroundColor: "rgba(46, 46, 46, 0.9)",
        titleFont: {
          size: 14,
          weight: "bold"
        },
        bodyFont: {
          size: 13
        },
        padding: 10,
        cornerRadius: 6,
        displayColors: false
      }
    },
    scales: {
      x: {
        grid: {
          color: "rgba(221, 161, 94, 0.1)",
          lineWidth: 1
        },
        ticks: {
          color: "#FEFAE0",
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
          color: "rgba(221, 161, 94, 0.1)",
          lineWidth: 1
        },
        ticks: {
          color: "#FEFAE0",
          font: {
            size: 12
          },
          padding: 10
        },
        border: {
          display: false
        }
      }
    },
    animation: {
      duration: 800,
      easing: 'easeOutQuart'
    }
  };

  return (
    <div className="bg-black-secondary rounded-xl p-6 shadow-lg border border-gray-dark hover:border-gray-medium transition-all duration-300 h-full">
      <h3 className="text-white text-xl font-semibold mb-6 border-b border-gray-dark pb-3">{formatMetricName(selectedMetric)} Metrics</h3>
      <div className="h-[350px] w-full">
        <Line options={options} data={data} />
      </div>
    </div>
  );
};

export default DynamicGraph;
