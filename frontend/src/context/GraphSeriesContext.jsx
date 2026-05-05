import { createContext, useCallback, useRef, useState } from "react";

export const GraphSeriesContext = createContext(null);

const MAX_POINTS = 10;

export function GraphSeriesProvider({ children }) {
  const [seriesMap, setSeriesMap] = useState({});
  const [snapshots, setSnapshots] = useState({});
  const lastQueryUpdateRef = useRef({});

  const appendPoint = useCallback((cacheKey, value) => {
    setSeriesMap((prev) => {
      const prevSeries = prev[cacheKey] ?? [];
      const next = [...prevSeries, value];
      while (next.length > MAX_POINTS) next.shift();
      return { ...prev, [cacheKey]: next };
    });
  }, []);

  /** Append once per React Query fetch (avoids duplicate point when remounting route). */
  const appendPointAfterQuery = useCallback((cacheKey, value, queryDataUpdatedAt) => {
    if (value === undefined || queryDataUpdatedAt == null) return;
    if (lastQueryUpdateRef.current[cacheKey] === queryDataUpdatedAt) return;
    lastQueryUpdateRef.current[cacheKey] = queryDataUpdatedAt;
    setSeriesMap((prev) => {
      const prevSeries = prev[cacheKey] ?? [];
      const next = [...prevSeries, value];
      while (next.length > MAX_POINTS) next.shift();
      return { ...prev, [cacheKey]: next };
    });
  }, []);

  const setSnapshot = useCallback((cacheKey, value) => {
    setSnapshots((prev) => ({ ...prev, [cacheKey]: value }));
  }, []);

  return (
    <GraphSeriesContext.Provider
      value={{
        seriesMap,
        appendPoint,
        appendPointAfterQuery,
        snapshots,
        setSnapshot,
      }}
    >
      {children}
    </GraphSeriesContext.Provider>
  );
}

export function graphSeriesCacheKey(soldierIndex, metric) {
  return `${soldierIndex}::${metric}`;
}

/** In-memory keys for dashboard charts (cleared on full page reload). */
export const DASHBOARD_STRIKE_SERIES_KEY = "dashboard::strike_success";
export const DASHBOARD_EFFICIENCY_SNAPSHOT_KEY = "dashboard::efficiency_predictions";
