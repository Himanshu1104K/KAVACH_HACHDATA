import { useContext, useState } from "react";
import { AuthContext } from "../MainComponent";

const LoginPage = () => {
  const { userName, password, setAutenticated } = useContext(AuthContext);
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");

  const handleLogin = (e) => {
    e.preventDefault(); // Prevent default form submission behavior
    if (user === userName && pass === password) {
      setAutenticated(true);
    } else {
      alert("Invalid Credentials");
    }
  };

  return (
    <div className="logP">
      <h1>Welcome to Kavach</h1>
      <div className="logBox">
        <h2>Authenticate Yourself</h2>
        <input
          type="text"
          value={user}
          onChange={(e) => setUser(e.target.value)}
          placeholder="Enter Username"
        />
        <input
          type="password"
          value={pass}
          onChange={(e) => setPass(e.target.value)}
          placeholder="Enter Password"
        />
        <button onClick={handleLogin}>Login</button> {/* Use button instead */}
      </div>
    </div>
  );
};

export default LoginPage;
