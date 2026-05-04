import React from "react";
import { Link } from "react-router-dom";
import { Shield, Mail } from "lucide-react";
import { cn } from "@/lib/utils";

const IconRss = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
    <path d="M4 11a9 9 0 0 1 9 9M4 4a16 16 0 0 1 16 16" />
    <circle cx="5" cy="19" r="1" fill="currentColor" stroke="none" />
  </svg>
);
const IconLinkedin = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <path d="M6.5 8.5h-3V21h3V8.5zm-1.5-5a1.75 1.75 0 1 0 0 3.5A1.75 1.75 0 0 0 5 3.5zM21 21h-3v-5.5c0-1.3-.5-2.2-1.7-2.2-1 0-1.5.7-1.8 1.3-.1.2-.1.5-.1.8V21h-3V8.5h3v1.4h.1c.4-.7 1.4-1.6 3.4-1.6 2.5 0 4.1 1.6 4.1 4.8V21z" />
  </svg>
);
const IconGithub = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <path d="M12 2C6.48 2 2 6.58 2 12.26c0 4.52 2.87 8.35 6.84 9.7.5.1.68-.22.68-.48 0-.24-.01-.87-.01-1.7-2.78.62-3.37-1.37-3.37-1.37-.45-1.18-1.1-1.5-1.1-1.5-.9-.63.07-.62.07-.62 1 .07 1.53 1.05 1.53 1.05.9 1.52 2.35 1.08 2.92.83.09-.65.35-1.08.63-1.33-2.22-.26-4.55-1.14-4.55-5.07 0-1.12.38-2.03 1-2.75-.1-.25-.44-1.3.1-2.7 0 0 .84-.27 2.75 1.05A9.3 9.3 0 0 1 12 6.8c.85.004 1.71.12 2.51.35 1.9-1.32 2.75-1.05 2.75-1.05.54 1.4.2 2.45.1 2.7.63.72 1 1.63 1 2.75 0 3.94-2.34 4.8-4.57 5.06.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.8 0 .27.18.59.69.48A10.01 10.01 0 0 0 22 12.26C22 6.58 17.52 2 12 2z" />
  </svg>
);

export interface FooterLink {
  label: string;
  href: string;
  external?: boolean;
}

export interface SocialLink {
  icon: React.ReactNode;
  href: string;
  label: string;
}

export interface KavachFooterProps {
  brandName?: string;
  brandDescription?: string;
  socialLinks?: SocialLink[];
  navLinks?: FooterLink[];
  creatorName?: string;
  creatorUrl?: string;
  brandIcon?: React.ReactNode;
  className?: string;
}

const defaultSocial: SocialLink[] = [
  { icon: <IconRss className="h-5 w-5" />, href: "https://twitter.com", label: "Updates" },
  { icon: <IconLinkedin className="h-5 w-5" />, href: "https://linkedin.com", label: "LinkedIn" },
  { icon: <IconGithub className="h-5 w-5" />, href: "https://github.com", label: "GitHub" },
  { icon: <Mail className="h-5 w-5" />, href: "mailto:contact@kavach.defence", label: "Email" },
];

const defaultNav: FooterLink[] = [
  { label: "Home", href: "/" },
  { label: "Login", href: "/login" },
  { label: "Request demo", href: "#cta" },
];

export function KavachFooter({
  brandName = "Kavach",
  brandDescription = "AI + IoT + Cloud soldier health monitoring for defence command.",
  socialLinks = defaultSocial,
  navLinks = defaultNav,
  creatorName,
  creatorUrl,
  brandIcon,
  className,
}: KavachFooterProps) {
  return (
    <section className={cn("relative w-full overflow-hidden border-t border-[#1e3a5f] bg-[#050d18]", className)}>
      <footer className="relative mt-0">
        <div className="mx-auto flex min-h-[22rem] max-w-7xl flex-col justify-between px-4 py-12 sm:min-h-[26rem] md:min-h-[30rem]">
          <div className="mb-10 flex w-full flex-col md:mb-14">
            <div className="flex w-full flex-col items-center">
              <div className="flex flex-1 flex-col items-center space-y-2">
                <div className="flex items-center gap-2">
                  {brandIcon ?? (
                    <div className="grid h-10 w-10 place-items-center rounded-lg bg-emerald-900/80 text-emerald-200">
                      <Shield className="h-6 w-6" />
                    </div>
                  )}
                  <span className="text-2xl font-bold tracking-tight text-[#e8f2ff] md:text-3xl">
                    {brandName}
                  </span>
                </div>
                <p className="max-w-md px-4 text-center text-sm font-medium text-[#94a3b8] sm:px-0 sm:text-base">
                  {brandDescription}
                </p>
              </div>

              {socialLinks.length > 0 && (
                <div className="mb-8 mt-4 flex gap-4">
                  {socialLinks.map((link, index) => (
                    <a
                      key={index}
                      href={link.href}
                      className="text-[#94a3b8] transition-colors hover:text-emerald-300"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <span className="block hover:scale-110 duration-300">{link.icon}</span>
                      <span className="sr-only">{link.label}</span>
                    </a>
                  ))}
                </div>
              )}

              {navLinks.length > 0 && (
                <nav className="flex max-w-full flex-wrap justify-center gap-4 px-4 text-sm font-medium text-[#94a3b8]">
                  {navLinks.map((link, index) =>
                    link.external ? (
                      <a
                        key={index}
                        className="duration-300 hover:font-semibold hover:text-[#e8f2ff]"
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {link.label}
                      </a>
                    ) : link.href.startsWith("#") ? (
                      <a
                        key={index}
                        className="duration-300 hover:font-semibold hover:text-[#e8f2ff]"
                        href={link.href}
                      >
                        {link.label}
                      </a>
                    ) : (
                      <Link
                        key={index}
                        className="duration-300 hover:font-semibold hover:text-[#e8f2ff]"
                        to={link.href}
                      >
                        {link.label}
                      </Link>
                    )
                  )}
                </nav>
              )}
            </div>
          </div>

          <div className="mt-12 flex flex-col items-center justify-center gap-2 px-4 md:mt-16 md:flex-row md:items-center md:justify-between md:px-0">
            <p className="text-center text-sm text-[#94a3b8] md:text-left md:text-base">
              © {new Date().getFullYear()} {brandName}. All rights reserved.
            </p>
            {creatorName && creatorUrl && (
              <a
                href={creatorUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-[#94a3b8] transition-colors duration-300 hover:font-medium hover:text-[#e8f2ff] md:text-base"
              >
                Crafted by {creatorName}
              </a>
            )}
          </div>
        </div>

        <div
          className="pointer-events-none absolute bottom-28 left-1/2 -translate-x-1/2 select-none bg-gradient-to-b from-[#e8f2ff]/15 via-[#e8f2ff]/8 to-transparent bg-clip-text px-4 text-center font-extrabold tracking-tighter text-transparent md:bottom-24"
          style={{
            fontSize: "clamp(3rem, 12vw, 9rem)",
            maxWidth: "95vw",
          }}
        >
          {brandName.toUpperCase()}
        </div>

        <div className="absolute bottom-20 left-1/2 z-10 flex -translate-x-1/2 items-center justify-center rounded-3xl border-2 border-[#1e3a5f] bg-[#0a1628]/70 p-3 shadow-[0_0_24px_rgba(0,0,0,0.45)] backdrop-blur-sm duration-300 hover:border-emerald-500/40 md:bottom-16">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-700 to-emerald-900 shadow-lg sm:h-14 sm:w-14 md:h-20 md:w-20">
            {brandIcon ?? <Shield className="h-6 w-6 text-white sm:h-8 sm:w-8 md:h-10 md:w-10" />}
          </div>
        </div>

        <div className="absolute bottom-28 left-1/2 h-px w-full max-w-4xl -translate-x-1/2 bg-gradient-to-r from-transparent via-[#1e3a5f] to-transparent sm:bottom-32" />

        <div className="pointer-events-none absolute bottom-24 h-20 w-full bg-gradient-to-t from-[#050d18] via-[#050d18]/80 to-transparent blur-md md:bottom-20" />
      </footer>
    </section>
  );
}
