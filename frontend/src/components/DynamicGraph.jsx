import { useContext, useEffect } from "react";
import { AuthContext } from "../MainComponent";
import { GraphSeriesContext, graphSeriesCacheKey } from "../context/GraphSeriesContext";
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
  const { seriesMap, appendPoint } = useContext(GraphSeriesContext);
  const cacheKey = graphSeriesCacheKey(soldierIndex, selectedMetric);
  const chartData = seriesMap[cacheKey] ?? [];

  useEffect(() => {
    if (solData !== undefined) {
      try {
        let newValue;

        if (selectedMetric === "efficiency_predictions") {
          newValue = solData.efficiency_predictions[soldierIndex];
        } else if (solData.soldier_data && solData.soldier_data[selectedMetric]) {
          const strIndex = soldierIndex.toString();
          newValue = Math.floor(solData.soldier_data[selectedMetric][strIndex] || 0);
        } else {
          newValue = 0;
        }

        appendPoint(cacheKey, newValue);
      } catch (error) {
        console.error("Error updating chart data:", error);
        appendPoint(cacheKey, 0);
      }
    }
  }, [solData, selectedMetric, soldierIndex, cacheKey, appendPoint]);

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
        backgroundColor: "rgba(16, 185, 129, 0.22)",
        borderColor: "#10b981",
        borderWidth: 3,
        tension: 0.4,
        pointBackgroundColor: "#d1fae5",
        pointBorderColor: "#10b981",
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
          color: "#d1fae5",
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
        }
      }
    },
    animation: {
      duration: 800,
      easing: 'easeOutQuart'
    }
  };

  return (
    <div className="ui-panel h-full p-6 transition-all duration-300 hover:border-emerald-500/30">
      <h3 className="ui-section-title text-white text-xl font-semibold mb-6">{formatMetricName(selectedMetric)} Metrics</h3>
      <div className="h-[350px] w-full">
        <Line options={options} data={data} />
      </div>
    </div>
  );
};

export default DynamicGraph;
