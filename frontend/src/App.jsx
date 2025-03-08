import "./App.css";
import { createContext } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MainComponent from "./MainComponent";

function App() {
  const client = new QueryClient();
  return (
    <>
      <QueryClientProvider client={client}>
        <MainComponent />
      </QueryClientProvider>
    </>
  );
}

export default App;
