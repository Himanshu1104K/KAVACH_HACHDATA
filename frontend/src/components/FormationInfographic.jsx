import { useMemo } from "react";
import { motion } from "framer-motion";
import { getSoldierNameById } from "../constants/soldierNames";
import { convexHull } from "../lib/formationLayout";

const CENTER_ID = 5;

const BAND_LABELS = ["Tip", "Forward", "Core", "Support", "Rear"];

/** SVG user-space springs — reads as smooth drift, not a hard tick */
const spring = { type: "spring", stiffness: 155, damping: 20, mass: 0.78 };
const lineSpring = { type: "spring", stiffness: 140, damping: 22, mass: 0.72 };
const colorTween = { duration: 0.38, ease: [0.33, 1, 0.68, 1] };

function statusForEfficiency(efficiency) {
  if (efficiency < 30) {
    return {
      stroke: "#fb7185",
      fill: "rgba(251, 113, 133, 0.28)",
      label: "Critical",
    };
  }
  if (efficiency > 70) {
    return {
      stroke: "#34d399",
      fill: "rgba(52, 211, 153, 0.26)",
      label: "Stable",
    };
  }
  return {
    stroke: "#fbbf24",
    fill: "rgba(251, 191, 36, 0.22)",
    label: "Watch",
  };
}

/**
 * Field diagram: depth bands + hub-and-spoke from anchor (center slot).
 * Coordinates are 0–100 (percent of view box).
 */
export default function FormationInfographic({ soldiers }) {
  const center = soldiers.find((s) => s.id === CENTER_ID);

  const bands = useMemo(() => {
    const sortedY = [...soldiers.map((s) => s.y)].sort((a, b) => a - b);
    if (sortedY.length === 0) return [];
    const idx = (t) => sortedY[Math.min(sortedY.length - 1, Math.round(t * (sortedY.length - 1)))];
    return BAND_LABELS.map((label, i) => ({
      label,
      y: idx((i + 0.5) / BAND_LABELS.length),
    }));
  }, [soldiers]);

  const hullPointsAttr = useMemo(() => {
    const pts = soldiers.map((s) => ({ x: s.x, y: s.y }));
    const hull = convexHull(pts);
    if (hull.length < 3) return null;
    return hull.map((p) => `${p.x},${p.y}`).join(" ");
  }, [soldiers]);

  return (
    <div className="flex flex-col gap-5 lg:flex-row lg:gap-6">
      <div className="min-w-0 flex-1 overflow-hidden rounded-2xl border border-emerald-500/20 bg-gradient-to-b from-slate-950/95 via-[#0a1628] to-[#050d14] p-2 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] sm:p-3">
        <svg
          viewBox="0 0 100 100"
          className="mx-auto block h-auto w-full max-h-[min(56vh,500px)] select-none"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="Soldier formation: depth bands and links to anchor position"
        >
          <defs>
            <linearGradient id="formation-field" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0f172a" stopOpacity="0.5" />
              <stop offset="55%" stopColor="#0c1929" stopOpacity="0.2" />
              <stop offset="100%" stopColor="#020617" stopOpacity="0.65" />
            </linearGradient>
          </defs>

          <rect width="100" height="100" fill="url(#formation-field)" rx="1" />

          {/* Depth bands: label + faint guide */}
          {bands.map(({ y, label }, i) => (
            <g key={`band-${i}-${label}`}>
              <motion.line
                initial={false}
                animate={{ y1: y, y2: y }}
                transition={{ y1: lineSpring, y2: lineSpring }}
                x1="14"
                x2="98"
                stroke="#94a3b8"
                strokeWidth="0.12"
                strokeOpacity="0.18"
                strokeDasharray="1.2 1.2"
              />
              <motion.text
                initial={false}
                animate={{ y }}
                transition={{ y: lineSpring }}
                x="3"
                dominantBaseline="middle"
                fill="#64748b"
                style={{ fontSize: "2.6px", fontWeight: 600 }}
                letterSpacing="0.04em"
              >
                {label}
              </motion.text>
            </g>
          ))}

          {/* Hub spokes: anchor to each outer slot */}
          {center &&
            soldiers
              .filter((s) => s.id !== CENTER_ID)
              .map((s) => (
                <motion.line
                  key={`spoke-${s.id}`}
                  initial={false}
                  animate={{
                    x1: center.x,
                    y1: center.y,
                    x2: s.x,
                    y2: s.y,
                  }}
                  transition={{
                    x1: lineSpring,
                    y1: lineSpring,
                    x2: lineSpring,
                    y2: lineSpring,
                  }}
                  stroke="#2dd4bf"
                  strokeWidth="0.32"
                  strokeOpacity="0.35"
                  strokeDasharray="1.5 1.2"
                  strokeLinecap="round"
                />
              ))}

          {/* Perimeter: convex hull of current positions (updates every layout) */}
          {hullPointsAttr ? (
            <polygon
              fill="none"
              stroke="#64748b"
              strokeWidth="0.28"
              strokeOpacity="0.35"
              strokeDasharray="1.8 1.2"
              strokeLinejoin="round"
              points={hullPointsAttr}
            />
          ) : null}

          {soldiers.map((s) => {
            const st = statusForEfficiency(s.efficiency);
            const isCenter = s.id === CENTER_ID;
            const r = isCenter ? 4.6 : 3.35;
            const name = getSoldierNameById(s.id);
            return (
              <g key={s.id}>
                <title>{`${name} · ${s.efficiency}% vitals (${st.label})`}</title>
                <motion.circle
                  r={r + 1.1}
                  fill="none"
                  strokeWidth="0.42"
                  strokeOpacity={0.55}
                  initial={false}
                  animate={{ cx: s.x, cy: s.y, stroke: st.stroke }}
                  transition={{ cx: spring, cy: spring, stroke: colorTween }}
                />
                <motion.circle
                  r={r}
                  strokeWidth="0.4"
                  initial={false}
                  animate={{ cx: s.x, cy: s.y, fill: st.fill, stroke: st.stroke }}
                  transition={{
                    cx: spring,
                    cy: spring,
                    fill: colorTween,
                    stroke: colorTween,
                  }}
                />
                <motion.text
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill="#f8fafc"
                  style={{ fontSize: isCenter ? "3.1px" : "2.75px", fontWeight: 700 }}
                  initial={false}
                  animate={{ x: s.x, y: s.y }}
                  transition={{ x: spring, y: spring }}
                >
                  {s.id}
                </motion.text>
                {isCenter ? (
                  <motion.text
                    textAnchor="middle"
                    fill="#99f6e4"
                    style={{ fontSize: "2.1px", fontWeight: 600 }}
                    opacity={0.92}
                    initial={false}
                    animate={{ x: s.x, y: s.y + r + 3.2 }}
                    transition={{ x: spring, y: spring }}
                  >
                    Anchor
                  </motion.text>
                ) : null}
              </g>
            );
          })}
        </svg>

        <p className="mt-2 px-1 text-center text-[11px] leading-snug text-slate-500 sm:text-xs">
          <span className="text-slate-400">Bands</span> follow this layout&apos;s vertical spread.{" "}
          <span className="text-slate-400">Dashed ring</span> is the convex hull (outer envelope).{" "}
          <span className="text-slate-400">Teal lines</span> link every soldier to anchor (5). Use{" "}
          <span className="text-slate-400">Refresh layout</span> for a new random disposition.
        </p>
      </div>

      <aside className="flex w-full shrink-0 flex-col justify-center gap-3 rounded-2xl border border-[#2b4a70]/70 bg-[#0a1628]/80 p-4 text-sm lg:w-56">
        <p className="text-xs font-semibold uppercase tracking-wider text-[#95e5ff]">Legend</p>
        <ul className="space-y-2.5 text-slate-300">
          <li className="flex items-start gap-2">
            <span className="mt-0.5 size-2.5 shrink-0 rounded-full bg-emerald-400 ring-2 ring-emerald-400/30" />
            <span>
              <span className="font-medium text-emerald-100/90">Stable</span>
              <span className="block text-xs text-slate-500">&gt; 70% efficiency</span>
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="mt-0.5 size-2.5 shrink-0 rounded-full bg-amber-400 ring-2 ring-amber-400/25" />
            <span>
              <span className="font-medium text-amber-100/90">Watch</span>
              <span className="block text-xs text-slate-500">30–70%</span>
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="mt-0.5 size-2.5 shrink-0 rounded-full bg-rose-400 ring-2 ring-rose-400/25" />
            <span>
              <span className="font-medium text-rose-100/90">Critical</span>
              <span className="block text-xs text-slate-500">&lt; 30%</span>
            </span>
          </li>
        </ul>
        <div className="mt-1 border-t border-emerald-500/15 pt-3 text-xs leading-relaxed text-slate-500">
          <span className="font-medium text-slate-400">Anchor (5):</span> hub for spokes; hull tightens when the unit
          clusters. Hover a disc for name and vitals.
        </div>
      </aside>
    </div>
  );
}
