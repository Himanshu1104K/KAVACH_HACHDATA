import { useState, useEffect } from "react";
import NavBar from "../components/NavBar";
import useFetchTactics from "../customHooks/useFetchTactics";
import useFetchStrike from "../customHooks/useFetchStrike";

const Tactics = () => {
  const { formation, isLoading, error } = useFetchTactics(
    "http://127.0.0.1:8000/soldier_tacktics"
  );
  const { strike, isLoading: strikeLoading } = useFetchStrike(
    "http://127.0.0.1:8000/strike_efficiency"
  );
  const [animateIn, setAnimateIn] = useState(false);

  useEffect(() => {
    // Trigger animation after component mount
    const timer = setTimeout(() => {
      setAnimateIn(true);
    }, 300);
    return () => clearTimeout(timer);
  }, []);

  // Generate random positions for the soldier dots formation
  const generateFormationPositions = (count = 10) => {
    const positions = [];
    const formationPattern = [
      { x: 50, y: 20 }, { x: 30, y: 35 }, { x: 70, y: 35 },
      { x: 20, y: 50 }, { x: 50, y: 50 }, { x: 80, y: 50 },
      { x: 30, y: 65 }, { x: 70, y: 65 }, { x: 40, y: 80 }, { x: 60, y: 80 }
    ];

    for (let i = 0; i < count; i++) {
      const position = formationPattern[i] || { x: 50, y: 50 + i * 5 };
      positions.push({
        id: i + 1,
        x: position.x,
        y: position.y,
        efficiency: Math.floor(Math.random() * 100)
      });
    }
    return positions;
  };

  const soldierPositions = generateFormationPositions();

  // Function to determine soldier dot color based on efficiency
  const getSoldierColor = (efficiency) => {
    if (efficiency < 30) return "bg-red-500";
    if (efficiency > 70) return "bg-green-500";
    return "bg-yellow-500";
  };

  // Format strike success probability
  const formatStrikeSuccess = () => {
    if (strikeLoading || !strike || !strike.strike_success_probability) {
      return "Calculating...";
    }
    return `${(strike.strike_success_probability * 100).toFixed(1)}%`;
  };

  // Calculate strike success width percentage for progress bar
  const strikeSuccessWidth = () => {
    if (strikeLoading || !strike || !strike.strike_success_probability) {
      return "50%"; // Default width while loading
    }
    return `${(strike.strike_success_probability * 100).toFixed(1)}%`;
  };

  // Determine strike success color based on percentage
  const strikeSuccessColor = () => {
    if (strikeLoading || !strike || !strike.strike_success_probability) {
      return "bg-gray-500";
    }
    const successRate = strike.strike_success_probability * 100;
    if (successRate < 30) return "bg-red-500";
    if (successRate > 70) return "bg-green-500";
    return "bg-yellow-500";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black-primary to-black-secondary">
      <NavBar />
      
      <div className="container mx-auto px-4 pt-24 pb-8">
        <div className={`transition-opacity duration-1000 ${animateIn ? 'opacity-100' : 'opacity-0'}`}>
          <h1 className="text-5xl font-bold text-center text-white mb-6 tracking-wider">
            BATTLE FORMATION
          </h1>
          
          <div className="bg-gradient-to-br from-black-secondary to-gray-dark rounded-xl p-8 shadow-xl border border-gray-dark mb-8">
            <div className="flex flex-col md:flex-row items-center justify-between mb-6 pb-6 border-b border-gray-medium">
              <div>
                <h2 className="text-3xl font-bold text-white mb-2">
                  {isLoading ? "Loading..." : error ? "Formation Not Available" : formation?.formation || "Standard Formation"}
                </h2>
                <p className="text-gray-lightest">
                  Optimal positioning based on soldier metrics and terrain analysis
                </p>
              </div>
              
              <div className="mt-4 md:mt-0">
                <button className="bg-gray-medium hover:bg-gray-light text-white font-bold py-3 px-6 rounded-lg transition duration-300 flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                  </svg>
                  Refresh Tactics
                </button>
              </div>
            </div>
            
            {/* Tactical Field Representation */}
            <div className="relative w-full h-[60vh] bg-gray-dark bg-opacity-40 rounded-lg border border-gray-medium overflow-hidden">
              {/* Terrain overlay - grid pattern */}
              <div className="absolute inset-0 grid grid-cols-10 grid-rows-10">
                {Array.from({ length: 100 }).map((_, i) => (
                  <div key={i} className="border border-gray-dark border-opacity-10"></div>
                ))}
              </div>
              
              {/* Coordinate grid lines with more transparency */}
              <div className="absolute inset-0">
                {/* Horizontal grid lines */}
                {Array.from({ length: 10 }).map((_, i) => (
                  <div 
                    key={`h-${i}`} 
                    className="absolute w-full border-t border-gray-light border-opacity-5"
                    style={{ top: `${(i + 1) * 10}%` }}
                  ></div>
                ))}
                
                {/* Vertical grid lines */}
                {Array.from({ length: 10 }).map((_, i) => (
                  <div 
                    key={`v-${i}`} 
                    className="absolute h-full border-l border-gray-light border-opacity-5"
                    style={{ left: `${(i + 1) * 10}%` }}
                  ></div>
                ))}
              </div>
              
              {/* Tactical indicators - like compass and markers */}
              <div className="absolute top-4 left-4 bg-black-primary bg-opacity-60 rounded-lg p-2 text-gray-lightest text-sm">
                N ↑
              </div>
              
              {/* Soldier positions */}
              {soldierPositions.map((soldier) => (
                <div 
                  key={soldier.id}
                  className={`absolute w-6 h-6 rounded-full ${getSoldierColor(soldier.efficiency)} shadow-lg transform hover:scale-125 transition-transform duration-300 flex items-center justify-center text-xs font-bold text-white border-2 border-white`}
                  style={{ 
                    left: `${soldier.x}%`, 
                    top: `${soldier.y}%`,
                    transform: 'translate(-50%, -50%)',
                    zIndex: 10
                  }}
                  title={`Soldier ${soldier.id}: ${soldier.efficiency}% Efficiency`}
                >
                  {soldier.id}
                </div>
              ))}
              
              {/* Formation lines connecting soldiers */}
              <svg className="absolute inset-0 w-full h-full" style={{ zIndex: 5 }}>
                <defs>
                  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#BBBBBB" />
                  </marker>
                </defs>
                
                {/* Connect soldiers with lines - just examples */}
                <line x1="50%" y1="20%" x2="30%" y2="35%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
                <line x1="50%" y1="20%" x2="70%" y2="35%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
                <line x1="30%" y1="35%" x2="20%" y2="50%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
                <line x1="70%" y1="35%" x2="80%" y2="50%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
                <line x1="30%" y1="35%" x2="50%" y2="50%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
                <line x1="70%" y1="35%" x2="50%" y2="50%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
                <line x1="20%" y1="50%" x2="30%" y2="65%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
                <line x1="80%" y1="50%" x2="70%" y2="65%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
                <line x1="30%" y1="65%" x2="40%" y2="80%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
                <line x1="70%" y1="65%" x2="60%" y2="80%" stroke="#BBBBBB" strokeWidth="1" strokeDasharray="4" strokeOpacity="0.4" />
              </svg>
              
              {/* Direction indicator */}
              <div className="absolute bottom-4 right-4 bg-black-primary bg-opacity-60 rounded-full p-3">
                <svg className="w-8 h-8 text-gray-lightest" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 11l3-3m0 0l3 3m-3-3v8m0-13a9 9 0 110 18 9 9 0 010-18z"></path>
                </svg>
              </div>
            </div>
          </div>
          
          {/* Battle Insights Panel */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-black-secondary rounded-xl p-6 shadow-lg border border-gray-dark hover:border-gray-medium transition-all duration-300">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                <svg className="w-5 h-5 mr-2 text-red-400" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                  <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"></path>
                  <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd"></path>
                </svg>
                Terrain Analysis
              </h3>
              <p className="text-gray-lightest mb-3">High ground toward the north, providing tactical advantage for ranged soldiers.</p>
              <div className="w-full bg-gray-dark rounded-full h-2 mb-1">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '75%' }}></div>
              </div>
              <p className="text-xs text-gray-light">75% favorable conditions</p>
            </div>
            
            <div className="bg-black-secondary rounded-xl p-6 shadow-lg border border-gray-dark hover:border-gray-medium transition-all duration-300">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                <svg className="w-5 h-5 mr-2 text-yellow-400" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                  <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd"></path>
                </svg>
                Strike Efficiency
              </h3>
              <p className="text-gray-lightest mb-3">Formation optimized for maximum strike efficiency based on soldier capabilities.</p>
              <div className="w-full bg-gray-dark rounded-full h-2 mb-1">
                <div className={`${strikeSuccessColor()} h-2 rounded-full transition-all duration-500`} style={{ width: strikeSuccessWidth() }}></div>
              </div>
              <p className="text-xs text-gray-light">{formatStrikeSuccess()} strike success probability</p>
            </div>
            
            <div className="bg-black-secondary rounded-xl p-6 shadow-lg border border-gray-dark hover:border-gray-medium transition-all duration-300">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                <svg className="w-5 h-5 mr-2 text-blue-400" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2h-1V9a1 1 0 00-1-1H9z" clipRule="evenodd"></path>
                </svg>
                Formation Details
              </h3>
              <p className="text-gray-lightest mb-3">Diamond formation with reinforced flanks to maximize coverage and defense.</p>
              <ul className="text-gray-lightest space-y-1 text-sm">
                <li>• Front line: 3 soldiers</li>
                <li>• Middle position: 4 soldiers</li>
                <li>• Rear support: 3 soldiers</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Tactics;
