import { useState, useEffect, useContext } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  LogOut,
  Map,
  Menu,
  User,
  X,
} from "lucide-react";
import { KavachMark } from "./KavachMark";
import { AuthContext } from "../MainComponent";
import { cn } from "../lib/utils";

const NAV_ITEMS = [
  {
    to: "/dashboard",
    match: (p) => p === "/dashboard",
    label: "Dashboard",
    Icon: LayoutDashboard,
  },
  {
    to: "/SingleSol/0",
    match: (p) => p.startsWith("/SingleSol"),
    label: "Soldier",
    Icon: User,
  },
  {
    to: "/tactics",
    match: (p) => p === "/tactics",
    label: "Formation",
    Icon: Map,
  },
];

export default function NavBar() {
  const { setAutenticated } = useContext(AuthContext);
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  const shell = cn(
    "fixed top-0 right-0 left-0 z-50 border-b transition-[background-color,border-color,backdrop-filter] duration-200",
    scrolled
      ? "border-[#2b4a70]/80 bg-[#050d18]/88 backdrop-blur-md"
      : "border-transparent bg-[#050d18]/72 backdrop-blur-sm"
  );

  const tabBase =
    "relative flex flex-1 items-center justify-center gap-2 rounded-full px-2 py-2 text-sm font-medium tracking-tight transition-all duration-200 sm:gap-2.5 sm:px-3";

  return (
    <nav className={shell} aria-label="Main">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4 sm:px-5">
        <Link
          to="/"
          className="group flex shrink-0 items-center gap-2.5 text-[#e8f2ff] sm:gap-3"
        >
          <KavachMark className="transition-transform duration-200 group-hover:scale-[1.02]" />
          <span className="text-base font-semibold tracking-wide sm:text-lg">
            Kavach
          </span>
        </Link>

        {/* Desktop: centered pill, equal-width tabs — matches dashboard heading polish */}
        <div className="hidden min-w-0 flex-1 justify-center md:flex md:px-2">
          <div
            className="flex w-full max-w-md items-stretch gap-0.5 rounded-full border border-[#2b4a70]/90 bg-[#0a1628]/95 p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]"
            role="tablist"
            aria-label="App sections"
          >
            {NAV_ITEMS.map(({ to, match, label, Icon }) => {
              const active = match(location.pathname);
              return (
                <Link
                  key={to}
                  to={to}
                  className={cn(
                    tabBase,
                    active
                      ? "bg-gradient-to-b from-emerald-500/30 to-emerald-600/15 text-white shadow-sm ring-1 ring-emerald-400/35"
                      : "text-[#8ea3b8] hover:bg-[#123059]/70 hover:text-[#e8f2ff]"
                  )}
                  aria-current={active ? "page" : undefined}
                >
                  <Icon
                    className={cn(
                      "size-4 shrink-0 sm:size-[1.05rem]",
                      active ? "text-emerald-200/95" : "opacity-85"
                    )}
                    strokeWidth={1.75}
                  />
                  <span
                    className={cn(
                      active &&
                        "bg-gradient-to-r from-white via-[#e8f2ff] to-emerald-100/90 bg-clip-text text-transparent"
                    )}
                  >
                    {label}
                  </span>
                  {active ? (
                    <span
                      className="absolute bottom-1 left-1/2 h-px w-8 -translate-x-1/2 bg-gradient-to-r from-transparent via-emerald-400/60 to-transparent sm:w-10"
                      aria-hidden
                    />
                  ) : null}
                </Link>
              );
            })}
          </div>
        </div>

        <div className="ml-auto flex shrink-0 items-center gap-2">
          <button
            type="button"
            className="flex size-9 items-center justify-center rounded-full border border-[#2b4a70] text-[#b8d7ff] transition hover:bg-[#123059] hover:text-white md:hidden"
            aria-expanded={menuOpen}
            aria-controls="mobile-nav"
            onClick={() => setMenuOpen((o) => !o)}
          >
            <span className="sr-only">{menuOpen ? "Close menu" : "Open menu"}</span>
            {menuOpen ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>

          <button
            type="button"
            className="inline-flex items-center gap-1.5 rounded-full border border-[#2b4a70] px-3 py-2 text-xs font-medium text-[#b8d7ff] transition hover:bg-[#123059] hover:text-[#e8f2ff] sm:text-sm"
            onClick={() => setAutenticated(false)}
          >
            <LogOut className="size-4 shrink-0 opacity-90" strokeWidth={1.75} />
            <span className="hidden sm:inline">Log out</span>
          </button>
        </div>
      </div>

      {/* Mobile panel */}
      <div
        id="mobile-nav"
        className={cn(
          "border-t border-[#2b4a70]/60 bg-[#050d18]/95 px-4 py-3 md:hidden",
          menuOpen ? "block" : "hidden"
        )}
      >
        <div className="mx-auto flex max-w-7xl flex-col gap-1 sm:px-1">
          {NAV_ITEMS.map(({ to, match, label, Icon }) => {
            const active = match(location.pathname);
            return (
              <Link
                key={to}
                to={to}
                className={cn(
                  "flex items-center justify-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium tracking-tight transition-all",
                  active
                    ? "bg-gradient-to-b from-emerald-500/25 to-emerald-600/12 text-white ring-1 ring-emerald-400/30"
                    : "text-[#8ea3b8] hover:bg-[#123059]/60 hover:text-[#e8f2ff]"
                )}
                aria-current={active ? "page" : undefined}
              >
                <Icon
                  className={cn("size-4 shrink-0", active && "text-emerald-200/90")}
                  strokeWidth={1.75}
                />
                <span
                  className={cn(
                    active &&
                      "bg-gradient-to-r from-white via-[#e8f2ff] to-emerald-100/90 bg-clip-text text-transparent"
                  )}
                >
                  {label}
                </span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
