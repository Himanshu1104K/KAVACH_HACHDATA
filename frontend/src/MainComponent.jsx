import { useState } from "react";
import "./App.css";
import LoginPage from "./pages/LoginPage";
import Tactics from "./pages/Tactics";
import { createContext } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MainDash from "./pages/MainDash";
import SingleSol from "./pages/SingleSol";
import useFetchData from "./customHooks/useFetchData";
export const AuthContext = createContext();

function MainComponent() {
  const [isAutenticated, setAutenticated] = useState(false);
  const userName = "admin";
  const password = "0";
  const { solData, error, isLoading } = useFetchData(
    "https://kavach-backend-production.up.railway.app/"
    // "http://127.0.0.1:8000/"
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
              <Route
                path="/"
                element={isAutenticated ? <MainDash /> : <LoginPage />}
              />
              <Route
                path="/SingleSol/:id"
                element={isAutenticated ? <SingleSol /> : <LoginPage />}
              />
              <Route
                path="/tactics"
                element={isAutenticated ? <Tactics /> : <LoginPage />}
              />
            </Routes>
          </AuthContext.Provider>
        </div>
      </Router>
    </>
  );
}

export default MainComponent;
