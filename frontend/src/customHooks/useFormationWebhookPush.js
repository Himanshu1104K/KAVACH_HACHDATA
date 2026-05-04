import { useEffect, useRef } from "react";

const DEFAULT_INTERVAL_MS = 10_000;

/**
 * Client-side cadence similar to receiving a formation webhook every `intervalMs`:
 * bumps layout (random disposition) and calls `refetch()` for latest API payload.
 * (A real webhook POSTs to your server; the browser still needs polling, SSE, or WS.)
 *
 * @param {() => Promise<unknown> | unknown} refetch - e.g. React Query `refetch`
 * @param {() => void} onLayoutPush - e.g. increment layout tick for new coordinates
 * @param {number} [intervalMs]
 * @param {{ enabled?: boolean }} [options]
 */
export function useFormationWebhookPush(
  refetch,
  onLayoutPush,
  intervalMs = DEFAULT_INTERVAL_MS,
  { enabled = true } = {}
) {
  const refetchRef = useRef(refetch);
  const pushRef = useRef(onLayoutPush);

  useEffect(() => {
    refetchRef.current = refetch;
  }, [refetch]);

  useEffect(() => {
    pushRef.current = onLayoutPush;
  }, [onLayoutPush]);

  useEffect(() => {
    if (!enabled) return undefined;
    if (typeof refetchRef.current !== "function" || typeof pushRef.current !== "function") {
      return undefined;
    }

    const tick = () => {
      pushRef.current();
      void refetchRef.current();
    };

    const id = setInterval(tick, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, enabled]);
}

export { DEFAULT_INTERVAL_MS as FORMATION_WEBHOOK_INTERVAL_MS };
