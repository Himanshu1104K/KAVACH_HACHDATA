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
  const soldierIndex = parseInt(id) == 0 ? 0 : parseInt(id) - 1;
  const { solData } = useContext(AuthContext);
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (solData !== undefined) {
      setChartData((prevData) => {
        const newValue =
          selectedMetric == "Efficiency"
            ? solData.efficiency_predictions[soldierIndex]
            : Math.floor(solData.soldier_data[selectedMetric][id]);
        const newData = [...prevData, newValue];

        if (newData.length > 10) newData.shift();
        return newData;
      });
    }
  }, [solData, selectedMetric]); // Re-run when `solData` or `selectedMetric` changes

  const labels = chartData.map((_, index) => index + 1);

  const data = {
    labels: labels,
    datasets: [
      {
        label: selectedMetric,
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
      legend: { position: "top" },
      title: { display: true, text: `${selectedMetric} Over Time` },
    },
  };

  return (
    <div>
      <Line
        options={options}
        data={data}
        className="bg-[#F8F3D9] rounded-md p-6 "
      />
    </div>
  );
};

export default DynamicGraph;
