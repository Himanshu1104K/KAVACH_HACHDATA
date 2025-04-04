import NavBar from "../components/NavBar";
import EfficencyGraph from "../components/EfficencyGraph";
import StrikeGraph from "../components/StrikeGraph";
import StrikeList from "../components/StrikeList";
import PageTransition from "../components/PageTransition";

const MainDash = () => {
  return (
    <div className="bg-gradient-to-br from-black-primary to-black-secondary min-h-screen">
      <NavBar />
      <div className="container mx-auto px-4 pt-24 pb-8">
        <PageTransition>
          <h1 className="mainDH text-center text-5xl font-bold text-white mb-10">
            MAIN DASHBOARD
          </h1>
          
          {/* Quick Navigation - Demonstrates smooth scrolling */}
          <div className="flex justify-center mb-8">
            <div className="bg-black-secondary rounded-lg shadow-md p-2 inline-flex gap-3">
              <a href="#efficiency" className="text-gray-lightest hover:text-white hover:bg-gray-dark px-4 py-2 rounded-md transition-all duration-200">Efficiency</a>
              <a href="#strikes" className="text-gray-lightest hover:text-white hover:bg-gray-dark px-4 py-2 rounded-md transition-all duration-200">Strikes</a>
              <a href="#strike-list" className="text-gray-lightest hover:text-white hover:bg-gray-dark px-4 py-2 rounded-md transition-all duration-200">Strike List</a>
            </div>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
            <div id="efficiency" className="scroll-mt-32">
              <EfficencyGraph />
            </div>
            <div id="strikes" className="scroll-mt-32">
              <StrikeGraph />
            </div>
          </div>
          
          <div id="strike-list" className="bg-gradient-to-br from-black-secondary to-gray-dark rounded-xl p-8 border border-gray-dark shadow-xl hover:border-gray-medium transition-all duration-300 scroll-mt-32">
            <StrikeList />
          </div>
          
          {/* Back to top button */}
          <div className="flex justify-center mt-10">
            <a href="#" className="bg-gray-medium hover:bg-gray-light text-white rounded-full p-3 shadow-lg transition-all duration-300 hover:shadow-xl">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path>
              </svg>
            </a>
          </div>
        </PageTransition>
      </div>
    </div>
  );
};

export default MainDash;
