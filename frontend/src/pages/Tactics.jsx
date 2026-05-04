import { useCallback, useMemo, useState } from "react";
import NavBar from "../components/NavBar";
import PageTransition from "../components/PageTransition";
import FormationInfographic from "../components/FormationInfographic";
import useFetchTactics from "../customHooks/useFetchTactics";
import useFetchStrike from "../customHooks/useFetchStrike";
import {
  FORMATION_WEBHOOK_INTERVAL_MS,
  useFormationWebhookPush,
} from "../customHooks/useFormationWebhookPush";
import { generateRandomFormation } from "../lib/formationLayout";

const Tactics = () => {
  const { formation, isLoading, error, refetch: refetchFormation } = useFetchTactics(
    "http://127.0.0.1:8000/soldier_tacktics"
  );
  const { strike, isLoading: strikeLoading } = useFetchStrike(
    "http://127.0.0.1:8000/strike_efficiency"
  );

  const [layoutTick, setLayoutTick] = useState(0);
  const soldierPositions = useMemo(() => generateRandomFormation(), [layoutTick]);

  const bumpLayout = useCallback(() => {
    setLayoutTick((n) => n + 1);
  }, []);

  useFormationWebhookPush(refetchFormation, bumpLayout, FORMATION_WEBHOOK_INTERVAL_MS);

  const formatStrikeSuccess = () => {
    if (strikeLoading || !strike || !strike.strike_success_probability) {
      return "Calculating...";
    }
    return `${(strike.strike_success_probability * 100).toFixed(1)}%`;
  };

  const strikeSuccessWidth = () => {
    if (strikeLoading || !strike || !strike.strike_success_probability) {
      return "50%";
    }
    return `${(strike.strike_success_probability * 100).toFixed(1)}%`;
  };

  const strikeSuccessColor = () => {
    if (strikeLoading || !strike || !strike.strike_success_probability) {
      return "bg-slate-500";
    }
    const successRate = strike.strike_success_probability * 100;
    if (successRate < 30) return "bg-rose-500";
    if (successRate > 70) return "bg-emerald-500";
    return "bg-amber-500";
  };

  const formationTitle = isLoading
    ? "Loading formation…"
    : error
      ? "Formation unavailable"
      : formation?.formation || "Defend weakest sector";

  return (
    <div className="ui-shell min-h-screen">
      <NavBar />

      <div className="mx-auto max-w-7xl px-4 pb-10 pt-16 sm:px-5 sm:pt-20">
        <PageTransition>
          <header className="mb-8 text-center sm:mb-10">
            <h1 className="mx-auto max-w-2xl text-2xl font-semibold tracking-tight sm:text-3xl md:text-4xl">
              <span className="bg-gradient-to-r from-white via-[#e8f2ff] to-emerald-200/85 bg-clip-text text-transparent">
                Battle formation
              </span>
            </h1>
            <div
              className="mx-auto mt-4 h-px w-20 bg-gradient-to-r from-transparent via-emerald-400/45 to-transparent sm:mt-5 sm:w-24"
              aria-hidden
            />
          </header>

          <section
            id="formation"
            className="ui-panel mb-8 scroll-mt-28 p-5 transition hover:border-emerald-500/30 sm:p-7"
          >
            <div className="mb-6 flex flex-col items-center gap-4 border-b border-emerald-500/15 pb-6 sm:flex-row sm:justify-between">
              <div className="max-w-2xl text-center sm:text-left">
                <h2 className="text-lg font-semibold tracking-tight text-white sm:text-xl md:text-2xl">
                  {formationTitle}
                </h2>
                <p className="mt-1.5 text-sm leading-relaxed text-slate-400">
                  Live layout: depth bands, convex hull, anchor spokes — hover a node for name and vitals.{" "}
                  <span className="text-emerald-400/80">
                    Webhook-style push every {FORMATION_WEBHOOK_INTERVAL_MS / 1000}s
                  </span>{" "}
                  (re-fetch + new disposition).
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  bumpLayout();
                  void refetchFormation();
                }}
                className="inline-flex shrink-0 items-center gap-2 rounded-full border border-[#2b4a70] px-4 py-2 text-sm font-medium text-[#b8d7ff] transition hover:bg-[#123059] hover:text-[#e8f2ff]"
              >
                <svg className="size-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                Refresh layout
              </button>
            </div>

            <FormationInfographic key={layoutTick} soldiers={soldierPositions} />
          </section>

          <div id="insights" className="mb-8 grid scroll-mt-28 grid-cols-1 gap-6 md:grid-cols-3">
            <div className="ui-panel p-6 transition-all duration-300 hover:border-emerald-500/30">
              <h3 className="mb-4 flex items-center text-xl font-semibold text-white">
                <svg className="mr-2 h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20" aria-hidden>
                  <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                  <path
                    fillRule="evenodd"
                    d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                    clipRule="evenodd"
                  />
                </svg>
                Terrain analysis
              </h3>
              <p className="mb-3 text-gray-lightest">
                High ground toward the north, providing tactical advantage for ranged soldiers.
              </p>
              <div className="mb-1 h-2 w-full rounded-full bg-gray-dark">
                <div className="h-2 rounded-full bg-green-500" style={{ width: "75%" }} />
              </div>
              <p className="text-xs text-gray-light">75% favorable conditions</p>
            </div>

            <div className="ui-panel p-6 transition-all duration-300 hover:border-emerald-500/30">
              <h3 className="mb-4 flex items-center text-xl font-semibold text-white">
                <svg className="mr-2 h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20" aria-hidden>
                  <path
                    fillRule="evenodd"
                    d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z"
                    clipRule="evenodd"
                  />
                </svg>
                Strike efficiency
              </h3>
              <p className="mb-3 text-gray-lightest">
                Formation optimized for maximum strike efficiency based on soldier capabilities.
              </p>
              <div className="mb-1 h-2 w-full rounded-full bg-gray-dark">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${strikeSuccessColor()}`}
                  style={{ width: strikeSuccessWidth() }}
                />
              </div>
              <p className="text-xs text-gray-light">{formatStrikeSuccess()} strike success probability</p>
            </div>

            <div
              id="details"
              className="ui-panel scroll-mt-28 p-6 transition-all duration-300 hover:border-emerald-500/30"
            >
              <h3 className="mb-4 flex items-center text-xl font-semibold text-white">
                <svg className="mr-2 h-5 w-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20" aria-hidden>
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2h-1V9a1 1 0 00-1-1H9z"
                    clipRule="evenodd"
                  />
                </svg>
                Formation details
              </h3>
              <p className="mb-3 text-gray-lightest">
                Diamond shell with reinforced flanks: tip at slot 1, anchor at core (5), rear pair (9–10) holds the
                base.
              </p>
              <ul className="space-y-1 text-sm text-gray-lightest">
                <li>• Tip + wings: slots 1–3 (forward screen)</li>
                <li>• Core line: 4–5–6 (anchor + lateral reach)</li>
                <li>• Base: 7–8–9–10 (support and withdrawal corridor)</li>
              </ul>
            </div>
          </div>
        </PageTransition>
      </div>
    </div>
  );
};

export default Tactics;
