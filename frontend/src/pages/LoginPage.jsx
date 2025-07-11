import { useContext, useState, useEffect } from "react";
import { AuthContext } from "../MainComponent";

const LoginPage = () => {
  const { userName, password, setAutenticated } = useContext(AuthContext);
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [animateIn, setAnimateIn] = useState(false);

  // Trigger animation after component mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimateIn(true);
    }, 300);
    return () => clearTimeout(timer);
  }, []);

  // Clear error message when user or password changes
  useEffect(() => {
    if (error) setError("");
  }, [user, pass]);

  const handleLogin = (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    
    // Simulate network request with timeout
    setTimeout(() => {
      if (user === userName && pass === password) {
        setAutenticated(true);
      } else {
        setError("Invalid username or password. Please try again.");
      }
      setLoading(false);
    }, 800);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-black-primary">
      <div className={`flex flex-col items-center transition-opacity duration-1000 ${animateIn ? 'opacity-100' : 'opacity-0'}`}>
        <div className="mb-12 text-center">
          <h1 className="text-6xl font-bold text-white">KAVACH</h1>
        </div>
        
        <div className="w-[500px] bg-gray-dark rounded-lg overflow-hidden shadow-2xl shadow-[#343a40]">
          <div className="p-8">
            <h2 className="text-2xl font-bold text-center text-white mb-8">SECURE AUTHENTICATION</h2>
            
            <form onSubmit={handleLogin} className="space-y-6">
              <div className="space-y-2">
                <label className="block text-white text-left">Username</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                    </svg>
                  </div>
                  <input
                    type="text"
                    value={user}
                    onChange={(e) => setUser(e.target.value)}
                    placeholder="Enter Username"
                    className="bg-black-secondary text-white w-full pl-10 pr-3 py-3 rounded-md focus:outline-none focus:ring-1 focus:ring-gray-light"
                    required
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <label className="block text-white text-left">Password</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                    </svg>
                  </div>
                  <input
                    type={showPassword ? "text" : "password"}
                    value={pass}
                    onChange={(e) => setPass(e.target.value)}
                    placeholder="Enter Password"
                    className="bg-black-secondary text-white w-full pl-10 pr-10 py-3 rounded-md focus:outline-none focus:ring-1 focus:ring-gray-light"
                    required
                  />
                  <button 
                    type="button"
                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    <svg className="h-5 w-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d={showPassword 
                        ? "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                        : "M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l18 18"
                      }></path>
                    </svg>
                  </button>
                </div>
              </div>
              
              {error && (
                <div className="text-center">
                  <p className="text-red-500 text-sm">{error}</p>
                </div>
              )}
              
              <div className="pt-4">
                <button 
                  type="submit" 
                  disabled={loading}
                  className="w-full py-3 bg-gray-medium hover:bg-gray-light text-white font-semibold rounded-md transition-all duration-200"
                >
                  {loading ? (
                    <svg className="animate-spin h-5 w-5 mx-auto text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  ) : (
                    "Login"
                  )}
                </button>
              </div>
            </form>
            
            <div className="text-center mt-8">
              <p className="text-gray-accent text-sm">Secure military-grade authentication system</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
