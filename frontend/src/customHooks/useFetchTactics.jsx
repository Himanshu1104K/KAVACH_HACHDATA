import { useQuery } from "@tanstack/react-query";
import axios from "axios";

/**
 * @param {string} url
 * @param {{ refetchInterval?: number | false }} [options] — default: no auto-refetch (use `useFormationWebhookPush` for 10s cadence)
 */
const useFetchTactics = (url, options = {}) => {
  const { refetchInterval = false } = options;

  const {
    data: formation,
    isLoading,
    error,
    refetch,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ["formation", url],
    queryFn: async () => {
      const response = await axios.get(url);
      return response.data;
    },
    refetchInterval,
    refetchOnWindowFocus: true,
  });

  return { formation, error, isLoading, refetch, dataUpdatedAt };
};

export default useFetchTactics;
