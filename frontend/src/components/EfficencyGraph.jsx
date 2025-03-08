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
  const options = {};
  const data = {
    labels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    datasets: [
      {
        label: "Soldier Efficency Graph",
        data: solData?.efficiency_predictions,
        backgroundColor: solData?.efficiency_predictions.map((eff) => {
          return eff < 20 ? "#a31d1d63" : "#504b385c";
        }),

        borderColor: "black",
        borderWidth: 1,
      },
    ],
  };
  return (
    <div>
      <Bar
        options={options}
        data={data}
        className="bg-[#F8F3D9] rounded-md p-6"
      />
    </div>
  );
};

export default EfficencyGraph;
