import { useContext, useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { AuthContext } from "../MainComponent";

const LoginPage = () => {
  const { userName, password, setAutenticated } = useContext(AuthContext);
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [animateIn, setAnimateIn] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimateIn(true);
    }, 300);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (error) setError("");
  }, [user, pass]);

  const handleLogin = (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    setTimeout(() => {
      if (user === userName && pass === password) {
        setAutenticated(true);
      } else {
        setError("Invalid username or password. Please try again.");
      }
      setLoading(false);
    }, 800);
  };

  const inputClass =
    "w-full rounded-xl border border-emerald-500/20 bg-[#0a1628]/90 py-3 pl-10 pr-3 text-[#e8f2ff] placeholder:text-slate-500 outline-none transition focus:border-emerald-400/45 focus:ring-2 focus:ring-emerald-500/25";

  return (
    <div className="ui-shell relative min-h-screen overflow-hidden">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        .font-jakarta { font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui; }
      `}</style>
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.35]"
        aria-hidden
        style={{
          backgroundImage: `radial-gradient(circle at 20% 20%, rgba(16,185,129,0.12), transparent 42%),
            radial-gradient(circle at 80% 10%, rgba(52,211,153,0.1), transparent 38%)`,
        }}
      />
      <div
        className={`font-jakarta relative z-10 flex min-h-screen flex-col items-center justify-center px-4 py-12 transition-opacity duration-700 ${animateIn ? "opacity-100" : "opacity-0"}`}
      >
        <div className="mb-10 text-center">
          <p className="mb-3 inline-flex rounded-full border border-emerald-500/25 bg-[#0d203a]/80 px-3 py-1 text-xs font-medium tracking-wider text-emerald-200/90">
            COMMAND ACCESS
          </p>
          <h1 className="bg-gradient-to-r from-white via-emerald-50 to-emerald-200/80 bg-clip-text text-5xl font-extrabold tracking-tight text-transparent drop-shadow-sm sm:text-6xl">
            Kavach
          </h1>
          <p className="mt-2 text-sm text-slate-400">Soldier health monitoring</p>
        </div>

        <div className="w-full max-w-[440px] overflow-hidden rounded-2xl border border-emerald-500/20 bg-gradient-to-b from-[#0f223f]/95 to-[#0a1628]/98 p-px shadow-[0_24px_60px_rgba(2,6,23,0.45)] backdrop-blur-md">
          <div className="rounded-2xl bg-[#0b1528]/90 p-8 sm:p-9">
            <h2 className="text-center text-xl font-bold text-white sm:text-2xl">
              Secure sign in
            </h2>
            <p className="mt-2 text-center text-sm text-slate-400">
              Use your issued credentials to open the operations dashboard.
            </p>

            <form onSubmit={handleLogin} className="mt-8 space-y-5">
              <div className="space-y-2">
                <label className="block text-left text-sm font-medium text-slate-300">
                  Username
                </label>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                    <svg
                      className="h-5 w-5 text-emerald-400/80"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                      />
                    </svg>
                  </div>
                  <input
                    type="text"
                    value={user}
                    onChange={(e) => setUser(e.target.value)}
                    placeholder="Enter username"
                    className={inputClass}
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="block text-left text-sm font-medium text-slate-300">
                  Password
                </label>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                    <svg
                      className="h-5 w-5 text-emerald-400/80"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                      />
                    </svg>
                  </div>
                  <input
                    type={showPassword ? "text" : "password"}
                    value={pass}
                    onChange={(e) => setPass(e.target.value)}
                    placeholder="Enter password"
                    className={`${inputClass} pr-11`}
                    required
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 flex items-center rounded-r-xl pr-3 text-slate-400 hover:text-emerald-200"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d={
                          showPassword
                            ? "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                            : "M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l18 18"
                        }
                      />
                    </svg>
                  </button>
                </div>
              </div>

              {error && (
                <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 px-3 py-2 text-center text-sm text-rose-200">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-full bg-gradient-to-r from-emerald-700 to-emerald-800 py-3.5 text-sm font-semibold text-white shadow-lg shadow-emerald-950/35 transition hover:from-emerald-600 hover:to-emerald-700 disabled:opacity-70"
              >
                {loading ? (
                  <svg
                    className="mx-auto h-5 w-5 animate-spin text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                ) : (
                  "Enter command center"
                )}
              </button>
            </form>

            <p className="mt-8 text-center text-xs text-slate-500">
              TLS-ready session · Biometric-ready integrations
            </p>
            <div className="mt-6 text-center">
              <Link
                to="/"
                className="text-sm font-medium text-emerald-300/90 underline-offset-4 transition hover:text-emerald-200 hover:underline"
              >
                ← Back to Kavach home
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
