import { useQuery } from "@tanstack/react-query";
import axios from "axios";

const useFetchTactics = (url) => {
  const {
    data: formation,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["formation"],
    queryFn: async () => {
      try {
        const response = await axios.get(url);
        return response.data;
      } catch (err) {
        console.error("Error fetching data:", err);
        throw err; // Re-throw the error so it can be handled by React Query
      }
    },
    refetchInterval: 30000,
  });

  return { formation, error, isLoading };
};

export default useFetchTactics;
