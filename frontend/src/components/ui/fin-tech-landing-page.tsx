import React from "react";
import { Link } from "react-router-dom";
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

const featureItems = [
  {
    icon: IconActivity,
    title: "Real-time health monitoring",
    desc: "Heart rate, body temperature, SpO2, and vital streams are tracked continuously.",
  },
  {
    icon: IconBrain,
    title: "AI-based predictive alerts",
    desc: "Models flag fatigue, stress, and health risks before visible collapse.",
  },
  {
    icon: IconMapPinned,
    title: "GPS-based location tracking",
    desc: "Each soldier is geo-tagged for faster rescue and tactical coordination.",
  },
  {
    icon: IconRadar,
    title: "Multi-soldier command dashboard",
    desc: "Commanders visualize unit-wide health and readiness in one secure view.",
  },
  {
    icon: IconShieldCheck,
    title: "Tactical decision support",
    desc: "Risk heat, alerts, and live vitals support faster and safer battlefield calls.",
  },
  {
    icon: IconLock,
    title: "Secure transmission",
    desc: "SSL/TLS encrypted data flow with blockchain-ready integrity extension.",
  },
  {
    icon: IconFingerprint,
    title: "Biometric authentication",
    desc: "Face and fingerprint verification prevent unauthorized command access.",
  },
];

const stepItems = [
  "Wearables",
  "Secure Uplink",
  "AI Analysis",
  "Risk Alerts",
  "Command Action",
];

export default function SHMSLandingPage() {
  return (
    <div className="min-h-screen w-full bg-[#08111f] text-[#e8f2ff]">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        .font-jakarta { font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui; }
      `}</style>

      <div className="font-jakarta">
        <nav className="mx-auto flex w-full max-w-7xl items-center justify-between px-5 py-6">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-lg bg-[#123059] text-[#95e5ff]">
              <IconShieldCheck className="h-5 w-5" />
            </div>
            <span className="text-lg font-semibold tracking-wide">SHMS</span>
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
              Request Demo
            </a>
          </div>
        </nav>

        <section className="mx-auto grid w-full max-w-7xl grid-cols-1 gap-10 px-5 pb-14 pt-6 md:grid-cols-2">
          <div className="space-y-6">
            <p className="inline-flex rounded-full border border-[#2b4a70] bg-[#0d203a] px-3 py-1 text-xs tracking-wider text-[#95e5ff]">
              SOLDIER HEALTH MONITORING SYSTEM
            </p>
            <h1 className="text-4xl font-extrabold leading-tight md:text-6xl">
              Every heartbeat tracked.
              <br />
              Every mission protected.
            </h1>
            <p className="max-w-xl text-base text-[#b4c9e6] md:text-lg">
              AI + IoT + Cloud intelligence for real-time soldier safety, predictive
              alerts, and faster battlefield decisions.
            </p>
            <div className="flex items-center gap-3">
              <a
                href="#cta"
                className="inline-flex items-center rounded-full bg-[#1b4f7f] px-5 py-3 text-sm font-semibold text-white hover:bg-[#23639e]"
              >
                Secure Pilot Program <IconArrowUpRight className="ml-1 h-4 w-4" />
              </a>
            </div>
          </div>

          <div className="rounded-2xl border border-[#2b4a70] bg-gradient-to-b from-[#0f223f] to-[#0b1a31] p-6 shadow-2xl animate-[fadeInUp_0.65s_ease-out]">
            <h3 className="text-xl font-semibold">Live Battlefield Snapshot</h3>
            <div className="mt-5 grid grid-cols-2 gap-4">
              <div className="rounded-xl bg-[#102746] p-4">
                <p className="text-2xl font-bold text-[#95e5ff]">&lt;10s</p>
                <p className="text-xs text-[#a8bfdc]">alert latency</p>
              </div>
              <div className="rounded-xl bg-[#102746] p-4">
                <p className="text-2xl font-bold text-[#95e5ff]">24/7</p>
                <p className="text-xs text-[#a8bfdc]">telemetry feed</p>
              </div>
              <div className="rounded-xl bg-[#102746] p-4">
                <p className="text-2xl font-bold text-[#95e5ff]">GPS</p>
                <p className="text-xs text-[#a8bfdc]">position lock</p>
              </div>
              <div className="rounded-xl bg-[#102746] p-4">
                <p className="text-2xl font-bold text-[#95e5ff]">AI</p>
                <p className="text-xs text-[#a8bfdc]">risk prediction</p>
              </div>
            </div>
            <div className="mt-5 rounded-xl border border-[#27486d] bg-[#0c1d36] p-4">
              <p className="text-xs uppercase tracking-wider text-[#95e5ff]">
                Unit Readiness Mix
              </p>
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

        <section className="mx-auto w-full max-w-7xl px-5 py-8">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-[#2b4a70] bg-[#0e213c] p-5">
              <p className="text-xs uppercase tracking-wider text-[#95e5ff]">Problem</p>
              <p className="mt-2 text-sm text-[#c6d8ef]">Manual checks</p>
              <p className="text-sm text-[#c6d8ef]">Delayed reporting</p>
              <p className="text-sm text-[#c6d8ef]">Late medical response</p>
            </div>
            <div className="rounded-xl border border-[#2b4a70] bg-[#0e213c] p-5">
              <p className="text-xs uppercase tracking-wider text-[#95e5ff]">Risk</p>
              <p className="mt-2 text-sm text-[#c6d8ef]">Higher casualties</p>
              <p className="text-sm text-[#c6d8ef]">Mission uncertainty</p>
              <p className="text-sm text-[#c6d8ef]">Poor live visibility</p>
            </div>
            <div className="rounded-xl border border-[#2b4a70] bg-[#0e213c] p-5">
              <p className="text-xs uppercase tracking-wider text-[#95e5ff]">SHMS Fix</p>
              <p className="mt-2 text-sm text-[#c6d8ef]">Live health streams</p>
              <p className="text-sm text-[#c6d8ef]">Predictive alerts</p>
              <p className="text-sm text-[#c6d8ef]">Actionable command view</p>
            </div>
          </div>
        </section>

        <section className="mx-auto w-full max-w-7xl px-5 py-8">
          <h2 className="text-3xl font-bold">Core Capabilities</h2>
          <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {featureItems.map((item) => (
              <div
                key={item.title}
                className="rounded-xl border border-[#2b4a70] bg-[#0e213c] p-5"
              >
                <item.icon className="h-5 w-5 text-[#95e5ff]" />
                <h3 className="mt-3 font-semibold">{item.title}</h3>
                <p className="mt-2 text-sm text-[#b4c9e6]">{item.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mx-auto w-full max-w-7xl px-5 py-8">
          <h2 className="text-3xl font-bold">How It Works</h2>
          <ol className="mt-5 grid grid-cols-1 gap-3 md:grid-cols-5">
            {stepItems.map((step, idx) => (
              <li
                key={step}
                className="rounded-xl border border-[#2b4a70] bg-[#0e213c] p-4 text-center text-[#c3d6ef]"
              >
                <span className="block text-xl font-bold text-[#95e5ff]">0{idx + 1}</span>
                <span className="mt-1 block text-sm">{step}</span>
              </li>
            ))}
          </ol>
        </section>

        <section className="mx-auto w-full max-w-7xl px-5 py-8">
          <h2 className="text-3xl font-bold">Mission Impact</h2>
          <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-4">
            <div className="rounded-xl bg-[#102746] p-5">
              <h3 className="font-semibold text-[#95e5ff]">Fatality Risk</h3>
              <p className="mt-1 text-3xl font-bold text-white">↓</p>
              <p className="mt-2 text-xs text-[#b4c9e6]">Earlier intervention window</p>
            </div>
            <div className="rounded-xl bg-[#102746] p-5">
              <h3 className="font-semibold text-[#95e5ff]">Mission Success</h3>
              <p className="mt-1 text-3xl font-bold text-white">↑</p>
              <p className="mt-2 text-xs text-[#b4c9e6]">Readiness-led deployment</p>
            </div>
            <div className="rounded-xl bg-[#102746] p-5">
              <h3 className="font-semibold text-[#95e5ff]">Decision Speed</h3>
              <p className="mt-1 text-3xl font-bold text-white">⚡</p>
              <p className="mt-2 text-xs text-[#b4c9e6]">Live alert-driven command</p>
            </div>
            <div className="rounded-xl bg-[#102746] p-5">
              <h3 className="font-semibold text-[#95e5ff]">Data Integrity</h3>
              <p className="mt-1 text-3xl font-bold text-white">🔒</p>
              <p className="mt-2 text-xs text-[#b4c9e6]">Encrypted + biometric access</p>
            </div>
          </div>
        </section>

        <section className="mx-auto w-full max-w-7xl px-5 py-8">
          <h2 className="text-3xl font-bold">Future Vision</h2>
          <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-[#2b4a70] bg-[#0e213c] p-4 text-sm text-[#c6d8ef]">
              Digital twin-based health modeling
            </div>
            <div className="rounded-xl border border-[#2b4a70] bg-[#0e213c] p-4 text-sm text-[#c6d8ef]">
              Autonomous evacuation prioritization
            </div>
            <div className="rounded-xl border border-[#2b4a70] bg-[#0e213c] p-4 text-sm text-[#c6d8ef]">
              Unified cross-domain defence intelligence
            </div>
          </div>
        </section>

        <section id="cta" className="mx-auto w-full max-w-7xl px-5 pb-16 pt-8">
          <div className="rounded-2xl border border-[#2b4a70] bg-[#0f223f] p-8 text-center">
            <h2 className="text-3xl font-bold">Protect the force before risk becomes loss.</h2>
            <p className="mx-auto mt-3 max-w-3xl text-[#b4c9e6]">
              Built for defence organizations, government agencies, military R&D teams,
              and defence startups ready to modernize soldier survivability.
            </p>
            <a
              href="#"
              className="mt-6 inline-flex items-center rounded-full bg-[#1b4f7f] px-6 py-3 font-semibold text-white hover:bg-[#23639e]"
            >
              Book SHMS Strategic Demo <IconArrowUpRight className="ml-1 h-4 w-4" />
            </a>
          </div>
        </section>
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
