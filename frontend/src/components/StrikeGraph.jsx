import React, { useState, useEffect } from "react";
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
import useFetchStrike from "../customHooks/useFetchStrike";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const StrikeGraph = () => {
  const { strike } = useFetchStrike(
    "https://kavach-backend-production.up.railway.app/strike_efficiency"
    // "http://127.0.0.1:8000/strike_efficiency"
  );

  // Initialize state to hold the strike data points
  const [chartData, setChartData] = useState([]);

  // Effect to update the chart data when strike changes
  useEffect(() => {
    if (strike?.strike_success_probability !== undefined) {
      setChartData((prevData) => {
        // Append new strike value
        const newData = [...prevData, strike?.strike_success_probability];
        // If the array length exceeds 10, remove the first element
        if (newData.length > 10) {
          newData.shift();
        }
        return newData;
      });
    }
  }, [strike?.strike_success_probability]);

  // Generate labels dynamically based on the chartData length or any other logic
  const labels = chartData.map((_, index) => `Update ${index + 1}`);

  const data = {
    labels: labels,
    datasets: [
      {
        label: "Strike Success Rate",
        data: chartData,
        backgroundColor: "rgba(16, 185, 129, 0.3)",
        borderColor: "rgba(16, 185, 129, 1)",
        borderWidth: 3,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: "#FFFFFF",
        pointBorderColor: "rgba(16, 185, 129, 1)",
        pointRadius: 5,
        pointHoverRadius: 7,
        pointBorderWidth: 2,
        pointHoverBackgroundColor: "#FFFFFF",
        pointHoverBorderColor: "rgba(16, 185, 129, 1)",
        pointHoverBorderWidth: 3,
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
          color: "#FFFFFF",
          font: {
            size: 13,
            weight: "bold",
            family: "'Inter', sans-serif",
          },
          boxWidth: 15,
          padding: 20,
        },
      },
      title: {
        display: true,
        text: "Strike Success Probability Over Time",
        color: "#FFFFFF",
        font: {
          size: 18,
          weight: "bold",
          family: "'Inter', sans-serif",
        },
        padding: {
          bottom: 20,
        },
      },
      tooltip: {
        backgroundColor: "rgba(30, 30, 30, 0.9)",
        titleFont: {
          size: 14,
          weight: "bold",
        },
        bodyFont: {
          size: 13,
        },
        padding: 10,
        cornerRadius: 6,
        displayColors: false,
        callbacks: {
          label: function (context) {
            return `Probability: ${(context.raw * 100).toFixed(2)}%`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: "rgba(187, 187, 187, 0.1)",
          lineWidth: 1,
        },
        ticks: {
          color: "#E0E0E0",
          font: {
            size: 12,
          },
          padding: 10,
        },
        border: {
          display: false,
        },
      },
      y: {
        grid: {
          color: "rgba(187, 187, 187, 0.1)",
          lineWidth: 1,
        },
        ticks: {
          color: "#E0E0E0",
          font: {
            size: 12,
          },
          padding: 10,
          callback: function (value) {
            return (value * 100).toFixed(0) + "%";
          },
        },
        border: {
          display: false,
        },
      },
    },
    animation: {
      duration: 1200,
      easing: "easeOutQuart",
    },
  };

  return (
    <div className="bg-gradient-to-br from-black-secondary to-gray-dark rounded-xl p-6 shadow-xl border border-gray-dark hover:border-gray-medium transition-all duration-300">
      <h3 className="text-white text-xl font-semibold mb-6 border-b border-gray-dark pb-3">
        Strike Success Probability
      </h3>
      <div className="h-[350px] w-full">
        <Line options={options} data={data} />
      </div>
    </div>
  );
};

export default StrikeGraph;
