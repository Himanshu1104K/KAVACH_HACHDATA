import { useState } from "react";
import "./App.css";
import LoginPage from "./pages/LoginPage";
import Tactics from "./pages/Tactics";
import { createContext } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import MainDash from "./pages/MainDash";
import SingleSol from "./pages/SingleSol";
import useFetchData from "./customHooks/useFetchData";
import KavachLandingPage from "./components/ui/fin-tech-landing-page";
export const AuthContext = createContext();

function MainComponent() {
  const [isAutenticated, setAutenticated] = useState(false);
  const userName = "admin";
  const password = "0";
  const { solData, error, isLoading } = useFetchData(
    // "https://kavach-backend-production.up.railway.app/"
    // "http://127.0.0.1:8000/"
    "https://welcomed-wildcat-actively.ngrok-free.app/"
  );
  return (
    <>
      <Router>
        <div className="App">
          <AuthContext.Provider
            value={{
              isAutenticated,
              userName,
              password,
              setAutenticated,
              solData,
            }}
          >
            <Routes>
              <Route path="/" element={<KavachLandingPage />} />
              <Route
                path="/login"
                element={isAutenticated ? <Navigate to="/dashboard" replace /> : <LoginPage />}
              />
              <Route
                path="/dashboard"
                element={isAutenticated ? <MainDash /> : <Navigate to="/login" replace />}
              />
              <Route
                path="/SingleSol/:id"
                element={isAutenticated ? <SingleSol /> : <Navigate to="/login" replace />}
              />
              <Route
                path="/tactics"
                element={isAutenticated ? <Tactics /> : <Navigate to="/login" replace />}
              />
            </Routes>
          </AuthContext.Provider>
        </div>
      </Router>
    </>
  );
}

export default MainComponent;
