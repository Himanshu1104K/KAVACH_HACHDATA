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
    "https://fastapi-backend-for-kavach-production.up.railway.app/strike_efficiency"
  );

  console.log(strike);
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
  const labels = chartData.map((_, index) => index + 1);

  const data = {
    labels: labels,
    datasets: [
      {
        label: "Strike Rate",
        data: chartData,
        fill: false,
        backgroundColor: "#504b38",
        borderColor: "#504b385c",
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Strike Rate Over Time",
      },
    },
  };

  return (
    <div>
      <Line
        options={options}
        data={data}
        className="bg-[#F8F3D9] rounded-md p-6"
      />
    </div>
  );
};

export default StrikeGraph;
