import { useQuery } from "@tanstack/react-query";
import axios from "axios";

const useFetchStrike = (url) => {
  const {
    data: strike,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["strike"],
    queryFn: async () => {
      try {
        const response = await axios.get(url);
        return response.data;
      } catch (err) {
        console.error("Error fetching data:", err);
        throw err; // Re-throw the error so it can be handled by React Query
      }
    },
    refetchInterval: 5000,
  });

  return { strike, error, isLoading };
};

export default useFetchStrike;
