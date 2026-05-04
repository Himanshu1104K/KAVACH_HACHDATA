import React from "react";
import { Link } from "react-router-dom";
import { KavachMark } from "../KavachMark";
import { KavachFooter } from "./kavach-footer";

const IconShieldCheck = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path d="M12 3l7 3v6c0 5-3.5 8-7 9-3.5-1-7-4-7-9V6l7-3z" />
    <path d="M9 12l2 2 4-4" />
  </svg>
);
const IconActivity = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path d="M3 12h4l2-4 4 8 2-4h6" />
  </svg>
);
const IconRadar = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <circle cx="12" cy="12" r="8" />
    <path d="M12 12l5-5" />
    <circle cx="12" cy="12" r="1.5" />
  </svg>
);
const IconMapPinned = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path d="M12 21s6-5.2 6-10a6 6 0 1 0-12 0c0 4.8 6 10 6 10z" />
    <circle cx="12" cy="11" r="2.2" />
  </svg>
);
const IconBrain = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path d="M9 4a3 3 0 0 0-3 3v1a3 3 0 0 0 0 6v1a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3v-1a3 3 0 0 0 0-6V7a3 3 0 0 0-3-3H9z" />
  </svg>
);
const IconFingerprint = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path d="M12 4a6 6 0 0 1 6 6v3M6 13v-3a6 6 0 0 1 6-6M9 20c2.5-1.5 3-3.5 3-6v-2" />
  </svg>
);
const IconLock = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <rect x="5" y="10" width="14" height="10" rx="2" />
    <path d="M8 10V8a4 4 0 1 1 8 0v2" />
  </svg>
);
const IconArrowUpRight = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <path d="M7 17L17 7" />
    <path d="M9 7h8v8" />
  </svg>
);

function MiniSparkline({ stroke }: { stroke: string }) {
  return (
    <svg viewBox="0 0 120 36" className="mt-3 h-9 w-full" aria-hidden>
      <defs>
        <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={stroke} stopOpacity="0.35" />
          <stop offset="100%" stopColor={stroke} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path
        d="M0 28 L20 22 L40 26 L58 10 L78 18 L98 8 L120 14 L120 36 L0 36 Z"
        fill="url(#sparkFill)"
      />
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth="2"
        strokeLinecap="round"
        points="0,28 20,22 40,26 58,10 78,18 98,8 120,14"
      />
    </svg>
  );
}

function DonutRing({ pct, color }: { pct: number; color: string }) {
  const r = 16;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  return (
    <svg viewBox="0 0 48 48" className="h-14 w-14 shrink-0" aria-hidden>
      <circle cx="24" cy="24" r={r} fill="none" stroke="#1e3a5f" strokeWidth="5" />
      <circle
        cx="24"
        cy="24"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="5"
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        transform="rotate(-90 24 24)"
      />
    </svg>
  );
}

function ArcGauge({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded-xl bg-[#102746] p-4">
      <p className="text-xs font-medium text-[#95e5ff]">{label}</p>
      <div className="mt-2 flex items-center gap-3">
        <svg viewBox="0 0 64 40" className="h-12 w-20 shrink-0" aria-hidden>
          <path
            d="M 8 36 A 28 28 0 0 1 56 36"
            fill="none"
            stroke="#1e3a5f"
            strokeWidth="6"
            strokeLinecap="round"
          />
          <path
            d="M 8 36 A 28 28 0 0 1 56 36"
            fill="none"
            stroke="url(#gEmerald)"
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray="88 88"
          />
          <defs>
            <linearGradient id="gEmerald" x1="0" y1="0" x2="1" y2="0">
              <stop stopColor="#34d399" />
              <stop offset="1" stopColor="#059669" />
            </linearGradient>
          </defs>
        </svg>
        <div>
          <p className="text-2xl font-bold text-white">{value}</p>
          <p className="text-[11px] text-[#94a3b8]">{sub}</p>
        </div>
      </div>
    </div>
  );
}

const featureItems = [
  {
    icon: IconActivity,
    title: "Real-time vitals",
    desc: "HR, temp, SpO₂ streams.",
    spark: "#34d399",
  },
  {
    icon: IconBrain,
    title: "Predictive AI",
    desc: "Fatigue & stress scoring.",
    spark: "#2dd4bf",
  },
  {
    icon: IconMapPinned,
    title: "GPS context",
    desc: "Rescue & formation sync.",
    spark: "#4ade80",
  },
  {
    icon: IconRadar,
    title: "Unit dashboard",
    desc: "Multi-soldier readiness.",
    spark: "#6ee7b7",
  },
  {
    icon: IconShieldCheck,
    title: "Tactical support",
    desc: "Alert-led decisions.",
    spark: "#34d399",
  },
  {
    icon: IconLock,
    title: "Secure uplink",
    desc: "TLS + integrity ready.",
    spark: "#14b8a6",
  },
  {
    icon: IconFingerprint,
    title: "Biometric access",
    desc: "Face / fingerprint gate.",
    spark: "#5eead4",
  },
];

const stepItems = ["Wearables", "Secure Uplink", "AI Analysis", "Risk Alerts", "Command"];

const visionImages = [
  {
    title: "Digital twin health",
    src: "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=900&q=80",
    caption: "Correlate vitals with mission load in simulation.",
  },
  {
    title: "Command intelligence",
    src: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=900&q=80",
    caption: "Fuse telemetry, terrain, and AI risk in one view.",
  },
  {
    title: "Edge + cloud",
    src: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=900&q=80",
    caption: "Resilient pipelines from field sensors to secure cloud.",
  },
];

export default function KavachLandingPage() {
  return (
    <div className="min-h-screen w-full bg-[#08111f] text-[#e8f2ff]">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        .font-jakarta { font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui; }
      `}</style>

      <div className="font-jakarta">
        <nav className="mx-auto flex w-full max-w-7xl items-center justify-between px-5 py-6">
          <div className="flex items-center gap-3">
            <KavachMark />
            <span className="text-lg font-semibold tracking-wide text-[#e8f2ff]">Kavach</span>
          </div>
          <div className="flex items-center gap-2">
            <Link
              to="/login"
              className="rounded-full border border-[#2b4a70] px-4 py-2 text-sm text-[#b8d7ff] hover:bg-[#123059]"
            >
              Login
            </Link>
            <a
              href="#cta"
              className="rounded-full border border-[#2b4a70] px-4 py-2 text-sm text-[#b8d7ff] hover:bg-[#123059]"
            >
              Request demo
            </a>
          </div>
        </nav>

        {/* Hero — unchanged structure, Kavach copy */}
        <section className="mx-auto grid w-full max-w-7xl grid-cols-1 gap-10 px-5 pb-14 pt-6 md:grid-cols-2">
          <div className="space-y-6">
            <p className="inline-flex rounded-full border border-[#2b4a70] bg-[#0d203a] px-3 py-1 text-xs tracking-wider text-[#95e5ff]">
              KAVACH · SOLDIER HEALTH MONITORING
            </p>
            <h1 className="text-4xl font-extrabold leading-tight md:text-6xl">
              Every heartbeat tracked.
              <br />
              Every mission protected.
            </h1>
            <p className="max-w-xl text-base text-[#b4c9e6] md:text-lg">
              Kavach unifies wearables, GPS, and AI so command sees risk before it becomes
              casualty — in one calm, decisive picture.
            </p>
            <div className="flex items-center gap-3">
              <a
                href="#cta"
                className="inline-flex items-center rounded-full bg-emerald-800 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-950/40 hover:bg-emerald-700"
              >
                Secure pilot program <IconArrowUpRight className="ml-1 h-4 w-4" />
              </a>
            </div>
          </div>

          <div className="rounded-2xl border border-[#2b4a70] bg-gradient-to-b from-[#0f223f] to-[#0b1a31] p-6 shadow-2xl animate-[fadeInUp_0.65s_ease-out]">
            <h3 className="text-xl font-semibold">Live battlefield snapshot</h3>
            <div className="mt-5 grid grid-cols-2 gap-4">
              <div className="rounded-xl bg-[#102746] p-4">
                <p className="text-2xl font-bold text-[#95e5ff]">&lt;10s</p>
                <p className="text-xs text-[#a8bfdc]">alert latency</p>
              </div>
              <div className="rounded-xl bg-[#102746] p-4">
                <p className="text-2xl font-bold text-[#95e5ff]">24/7</p>
                <p className="text-xs text-[#a8bfdc]">telemetry</p>
              </div>
              <div className="rounded-xl bg-[#102746] p-4">
                <p className="text-2xl font-bold text-[#95e5ff]">GPS</p>
                <p className="text-xs text-[#a8bfdc]">position lock</p>
              </div>
              <div className="rounded-xl bg-[#102746] p-4">
                <p className="text-2xl font-bold text-[#95e5ff]">AI</p>
                <p className="text-xs text-[#a8bfdc]">risk score</p>
              </div>
            </div>
            <div className="mt-5 rounded-xl border border-[#27486d] bg-[#0c1d36] p-4">
              <p className="text-xs uppercase tracking-wider text-[#95e5ff]">Unit readiness mix</p>
              <div className="mt-3 flex items-end gap-3">
                {[78, 62, 90, 47, 71, 56].map((h, i) => (
                  <div key={i} className="flex-1">
                    <div
                      className="rounded-t-md bg-gradient-to-t from-emerald-600 to-emerald-300"
                      style={{ height: `${h}px` }}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Gap analysis — infographics + image */}
        <section className="mx-auto w-full max-w-7xl px-5 py-10">
          <div className="grid grid-cols-1 items-stretch gap-8 lg:grid-cols-2">
            <div className="grid gap-4">
              <div className="rounded-2xl border border-[#2b4a70] bg-[#0e213c] p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-wider text-rose-300">Legacy gap</p>
                    <p className="mt-1 text-sm text-[#c6d8ef]">Manual checks &amp; delayed reports stretch response time.</p>
                  </div>
                  <DonutRing pct={38} color="#fb7185" />
                </div>
                <MiniSparkline stroke="#fb7185" />
              </div>
              <div className="rounded-2xl border border-[#2b4a70] bg-[#0e213c] p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-wider text-amber-300">Field risk</p>
                    <p className="mt-1 text-sm text-[#c6d8ef]">Low visibility into stress &amp; fatigue before collapse.</p>
                  </div>
                  <DonutRing pct={62} color="#fbbf24" />
                </div>
                <MiniSparkline stroke="#fbbf24" />
              </div>
              <div className="rounded-2xl border border-emerald-500/30 bg-gradient-to-br from-[#0e213c] to-[#0a1628] p-5 ring-1 ring-emerald-500/20">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-wider text-emerald-300">Kavach closes the loop</p>
                    <p className="mt-1 text-sm text-[#c6d8ef]">Live vitals + AI alerts + GPS — one command surface.</p>
                  </div>
                  <DonutRing pct={91} color="#34d399" />
                </div>
                <MiniSparkline stroke="#34d399" />
              </div>
            </div>
            <div className="relative overflow-hidden rounded-2xl border border-[#2b4a70]">
              <img
                src="https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?auto=format&fit=crop&w=1000&q=80"
                alt="Clinical monitoring context for Kavach defence health platform"
                className="h-full min-h-[280px] w-full object-cover lg:min-h-full"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-[#08111f] via-[#08111f]/40 to-transparent" />
              <div className="absolute bottom-0 left-0 right-0 p-6">
                <p className="text-sm font-semibold text-white">Operational clarity</p>
                <p className="mt-1 max-w-md text-xs text-[#cbd5e1]">
                  Kavach turns fragmented vitals into a single, trusted operational picture for command.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Core capabilities — cards + spark */}
        <section className="mx-auto w-full max-w-7xl px-5 py-10">
          <h2 className="text-2xl font-bold md:text-3xl">Kavach capabilities</h2>
          <p className="mt-2 max-w-2xl text-sm text-[#94a3b8]">
            Each layer is built for mission tempo: fast scan, deep trust, minimal noise.
          </p>
          <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {featureItems.map((item) => (
              <div
                key={item.title}
                className="flex flex-col rounded-xl border border-[#2b4a70] bg-[#0e213c] p-4 transition hover:border-emerald-500/35"
              >
                <div className="flex items-center gap-2">
                  <item.icon className="h-5 w-5 text-emerald-300" />
                  <h3 className="font-semibold text-[#e8f2ff]">{item.title}</h3>
                </div>
                <p className="mt-1 text-xs text-[#94a3b8]">{item.desc}</p>
                <MiniSparkline stroke={item.spark} />
              </div>
            ))}
          </div>
        </section>

        {/* How it works — timeline infographic */}
        <section className="mx-auto w-full max-w-7xl px-5 py-10">
          <h2 className="text-2xl font-bold md:text-3xl">Signal path</h2>
          <p className="mt-2 text-sm text-[#94a3b8]">From body to decision — five beats, one chain.</p>
          <div className="relative mt-8 hidden md:block">
            <svg className="absolute left-0 right-0 top-8 h-2 w-full text-[#1e3a5f]" aria-hidden>
              <line x1="4%" y1="4" x2="96%" y2="4" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              <line x1="4%" y1="4" x2="72%" y2="4" stroke="url(#tlGrad)" strokeWidth="3" strokeLinecap="round" />
              <defs>
                <linearGradient id="tlGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop stopColor="#059669" />
                  <stop offset="1" stopColor="#34d399" />
                </linearGradient>
              </defs>
            </svg>
            <ol className="relative grid grid-cols-5 gap-2">
              {stepItems.map((step, idx) => (
                <li key={step} className="flex flex-col items-center text-center">
                  <span className="z-10 grid h-14 w-14 place-items-center rounded-full border-2 border-emerald-500/40 bg-[#0c1d36] text-lg font-bold text-emerald-300 shadow-lg">
                    {idx + 1}
                  </span>
                  <span className="mt-4 text-xs font-medium text-[#cbd5e1]">{step}</span>
                </li>
              ))}
            </ol>
          </div>
          <ol className="mt-6 grid grid-cols-1 gap-3 md:hidden">
            {stepItems.map((step, idx) => (
              <li
                key={step}
                className="flex items-center gap-3 rounded-xl border border-[#2b4a70] bg-[#0e213c] px-4 py-3"
              >
                <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-emerald-900/60 text-sm font-bold text-emerald-200">
                  {idx + 1}
                </span>
                <span className="text-sm text-[#cbd5e1]">{step}</span>
              </li>
            ))}
          </ol>
        </section>

        {/* Mission impact — gauges + mini chart strip */}
        <section className="mx-auto w-full max-w-7xl px-5 py-10">
          <h2 className="text-2xl font-bold md:text-3xl">Mission impact</h2>
          <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
            <ArcGauge label="Fatality risk window" value="↓" sub="Earlier intervention cues" />
            <ArcGauge label="Mission readiness" value="↑" sub="Live unit health index" />
            <ArcGauge label="Decision latency" value="⚡" sub="Alert-to-action pipeline" />
            <ArcGauge label="Trust layer" value="🔒" sub="TLS + biometrics" />
          </div>
          <div className="mt-6 rounded-xl border border-[#27486d] bg-[#0c1d36] p-4">
            <p className="text-xs uppercase tracking-wider text-[#95e5ff]">Illustrative response curve (minutes)</p>
            <svg viewBox="0 0 400 80" className="mt-2 h-20 w-full" aria-hidden>
              <rect width="400" height="80" fill="transparent" />
              {[0, 1, 2, 3, 4].map((i) => (
                <line
                  key={i}
                  x1={i * 80}
                  y1="0"
                  x2={i * 80}
                  y2="80"
                  stroke="#1e3a5f"
                  strokeWidth="1"
                />
              ))}
              <polyline
                fill="none"
                stroke="#34d399"
                strokeWidth="3"
                strokeLinecap="round"
                points="0,60 80,55 160,40 240,28 320,35 400,18"
              />
              <polyline
                fill="url(#respFill)"
                stroke="none"
                points="0,60 80,55 160,40 240,28 320,35 400,18 400,80 0,80"
              />
              <defs>
                <linearGradient id="respFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" stopOpacity="0.25" />
                  <stop offset="100%" stopColor="#34d399" stopOpacity="0" />
                </linearGradient>
              </defs>
            </svg>
          </div>
        </section>

        {/* Future vision — image cards */}
        <section className="mx-auto w-full max-w-7xl px-5 py-10">
          <h2 className="text-2xl font-bold md:text-3xl">Where Kavach is heading</h2>
          <div className="mt-6 grid grid-cols-1 gap-5 md:grid-cols-3">
            {visionImages.map((v) => (
              <article
                key={v.title}
                className="group relative overflow-hidden rounded-2xl border border-[#2b4a70] bg-[#0e213c]"
              >
                <img
                  src={v.src}
                  alt=""
                  className="aspect-[4/3] w-full object-cover transition duration-500 group-hover:scale-105"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#08111f] via-[#08111f]/55 to-transparent" />
                <div className="absolute bottom-0 p-4">
                  <h3 className="font-semibold text-white">{v.title}</h3>
                  <p className="mt-1 text-xs text-[#cbd5e1]">{v.caption}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section id="cta" className="mx-auto w-full max-w-7xl px-5 pb-8 pt-6">
          <div className="rounded-2xl border border-emerald-500/25 bg-gradient-to-br from-[#0f223f] to-[#0a1628] p-8 text-center shadow-[0_0_40px_rgba(16,185,129,0.12)]">
            <h2 className="text-2xl font-bold md:text-3xl">Protect the force before risk becomes loss.</h2>
            <p className="mx-auto mt-3 max-w-2xl text-sm text-[#94a3b8] md:text-base">
              Kavach is built for defence organisations, agencies, and research teams modernising soldier survivability.
            </p>
            <a
              href="#cta"
              className="mt-6 inline-flex items-center rounded-full bg-emerald-700 px-6 py-3 text-sm font-semibold text-white hover:bg-emerald-600 md:text-base"
            >
              Book a Kavach strategic demo <IconArrowUpRight className="ml-1 h-4 w-4" />
            </a>
          </div>
        </section>

        <KavachFooter
          brandName="Kavach"
          brandDescription="Soldier health monitoring — AI, IoT, and secure cloud for defence command."
        />
      </div>

      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(22px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
