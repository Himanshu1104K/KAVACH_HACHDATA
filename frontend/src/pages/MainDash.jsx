import NavBar from "../components/NavBar";
import EfficencyGraph from "../components/EfficencyGraph";
import StrikeGraph from "../components/StrikeGraph";
import StrikeList from "../components/StrikeList";
const MainDash = () => {
  return (
    <div>
      <NavBar />
      <h1 className="mainDH text-center mt-5">Main Dashboard </h1>
      <div className="grid grid-cols-2 gap-5 m-8">
        <EfficencyGraph />
        <StrikeGraph />
      </div>
      <div className="bg-[#F8F3D9] rounded-xl m-8 p-8">
        <StrikeList />
      </div>
    </div>
  );
};
export default MainDash;
