import NavBar from "../components/NavBar";
import EfficencyGraph from "../components/EfficencyGraph";
import StrikeGraph from "../components/StrikeGraph";
import StrikeList from "../components/StrikeList";

const MainDash = () => {
  return (
    <div className="bg-gradient-to-br from-black-primary to-black-secondary min-h-screen">
      <NavBar />
      <div className="container mx-auto px-4 pt-24 pb-8">
        <h1 className="mainDH text-center text-5xl font-bold text-white mb-10">
          MAIN DASHBOARD
        </h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
          <EfficencyGraph />
          <StrikeGraph />
        </div>
        
        <div className="bg-gradient-to-br from-black-secondary to-gray-dark rounded-xl p-8 border border-gray-dark shadow-xl hover:border-gray-medium transition-all duration-300">
          <StrikeList />
        </div>
      </div>
    </div>
  );
};

export default MainDash;
