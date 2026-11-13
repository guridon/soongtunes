"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { CSSProperties, ReactNode } from "react";
import type { SourceName } from "@/lib/types";

export type StickyColor = "yellow" | "green" | "pink" | "blue" | "violet" | "orange" | "teal" | "gray";

export const STICKY: Record<StickyColor, { bg: string; ink: string }> = {
  yellow: { bg: "var(--color-yellow)", ink: "var(--color-yellow-ink)" },
  green: { bg: "var(--color-green)", ink: "var(--color-green-ink)" },
  pink: { bg: "var(--color-pink)", ink: "var(--color-pink-ink)" },
  blue: { bg: "var(--color-blue)", ink: "var(--color-blue-ink)" },
  violet: { bg: "var(--color-violet)", ink: "var(--color-violet-ink)" },
  orange: { bg: "var(--color-orange)", ink: "var(--color-orange-ink)" },
  teal: { bg: "var(--color-teal)", ink: "var(--color-teal-ink)" },
  gray: { bg: "var(--color-gray)", ink: "var(--color-ink-2)" },
};

/** Where a recommendation came from — shown as a FigJam-style stamp on each sticky. */
export const SOURCES: Record<SourceName, { label: string; short: string; color: StickyColor; help: string }> = {
  lookup: { label: "곡 찾기", short: "찾기", color: "yellow", help: "제목·아티스트로 특정 곡 조회" },
  graph: { label: "그래프 조건", short: "그래프", color: "green", help: "Neo4j: 태그·장르·아티스트 조건 + 제외 조건" },
  cooccur: { label: "함께 담긴 곡", short: "동시등장", color: "orange", help: "같은 플레이리스트에 자주 함께 담긴 곡" },
  bm25: { label: "키워드", short: "BM25", color: "blue", help: "OpenSearch BM25 (Nori 형태소)" },
  vector: { label: "의미", short: "벡터", color: "violet", help: "Harrier 임베딩 유사도 (pgvector)" },
  text2cypher: { label: "Text2Cypher", short: "v1", color: "gray", help: "v1: LLM이 작성한 Cypher" },
  tags: { label: "선호 태그", short: "취향", color: "green", help: "유저 태그 선호 상위 태그로 그래프 조건 검색" },
  popular: { label: "인기", short: "인기", color: "teal", help: "많이 담긴 곡 (인기도 인덱스)" },
};

export function topSource(sources?: Partial<Record<SourceName, number>>): SourceName | null {
  if (!sources) return null;
  const entries = Object.entries(sources) as [SourceName, number][];
  if (!entries.length) return null;
  const order: SourceName[] = ["lookup", "graph", "tags", "cooccur", "vector", "bm25", "popular", "text2cypher"];
  entries.sort((a, b) => a[1] - b[1] || order.indexOf(a[0]) - order.indexOf(b[0]));
  return entries[0][0];
}

export function Stamp({ source, rank }: { source: SourceName; rank?: number }) {
  const s = SOURCES[source];
  const c = STICKY[s.color];
  return (
    <span
      title={`${s.label}${rank ? ` · ${rank}위` : ""} — ${s.help}`}
      className="inline-flex items-center gap-1 rounded-full border border-black/10 px-1.5 py-[1px] text-[10px] font-semibold leading-4"
      style={{ background: c.bg, color: c.ink }}
    >
      {s.short}
      {rank ? <span className="opacity-60">#{rank}</span> : null}
    </span>
  );
}

export function Chip({ children, color = "gray" }: { children: ReactNode; color?: StickyColor }) {
  const c = STICKY[color];
  return (
    <span
      className="inline-flex max-w-full items-center truncate rounded-md px-1.5 py-[1px] text-[11px] font-medium"
      style={{ background: c.bg, color: c.ink }}
    >
      {children}
    </span>
  );
}

export function Sticky({
  color = "yellow",
  className = "",
  style,
  children,
  tilt = 0,
}: {
  color?: StickyColor;
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
  tilt?: number;
}) {
  return (
    <div
      className={`rounded-[3px] p-3 ${className}`}
      style={{
        background: STICKY[color].bg,
        boxShadow: "var(--shadow-sticky)",
        transform: tilt ? `rotate(${tilt}deg)` : undefined,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Panel({ className = "", children }: { className?: string; children: ReactNode }) {
  return (
    <div
      className={`rounded-2xl border border-[var(--color-line)] bg-[var(--color-paper)] ${className}`}
      style={{ boxShadow: "var(--shadow-float)" }}
    >
      {children}
    </div>
  );
}

/** Small vinyl mark used as the app logo. */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <circle cx="12" cy="12" r="11" fill="#1e1e1e" />
      <circle cx="12" cy="12" r="7.5" fill="none" stroke="#3a3a3a" strokeWidth="0.8" />
      <circle cx="12" cy="12" r="5" fill="none" stroke="#3a3a3a" strokeWidth="0.8" />
      <circle cx="12" cy="12" r="3.2" fill="var(--color-yellow)" />
      <circle cx="12" cy="12" r="0.9" fill="#1e1e1e" />
    </svg>
  );
}

const NAV = [
  { href: "/", label: "보드" },
  { href: "/feed", label: "홈 피드" },
  { href: "/compare", label: "전략 비교" },
  { href: "/system", label: "시스템" },
];

export function TopBar({ title, right }: { title: string; right?: ReactNode }) {
  const path = usePathname();
  return (
    <div className="pointer-events-none absolute inset-x-3 top-3 z-20 flex items-start justify-between gap-3">
      <Panel className="pointer-events-auto flex items-center gap-1 px-2 py-1.5">
        <Link href="/" className="flex items-center gap-2 rounded-lg px-1.5 py-1 hover:bg-black/5">
          <Logo />
          <span className="text-[15px] font-bold tracking-tight">SoongTunes</span>
        </Link>
        <span className="hidden px-1 text-[var(--color-ink-3)] sm:inline">/</span>
        <span className="hidden max-w-[200px] truncate px-1 text-sm text-[var(--color-ink-2)] sm:inline">{title}</span>
        <div className="mx-1 h-5 w-px bg-[var(--color-line)]" />
        <nav className="flex items-center gap-0.5">
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <Link
                key={n.href}
                href={n.href}
                className={`rounded-lg px-2.5 py-1 text-[13px] font-medium transition ${
                  active ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]" : "text-[var(--color-ink-2)] hover:bg-black/5"
                }`}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>
      </Panel>
      {right ? <div className="pointer-events-auto">{right}</div> : null}
    </div>
  );
}

export function Avatars() {
  return (
    <Panel className="flex items-center gap-2 px-2.5 py-1.5">
      <div className="flex -space-x-1.5">
        <span className="grid h-7 w-7 place-items-center rounded-full border-2 border-white bg-[var(--color-accent)] text-[11px] font-bold text-white">
          AI
        </span>
        <span className="grid h-7 w-7 place-items-center rounded-full border-2 border-white bg-[#ff7262] text-[11px] font-bold text-white">
          나
        </span>
      </div>
    </Panel>
  );
}

export function Icon({ name, size = 18 }: { name: "fit" | "plus" | "minus" | "reset" | "send" | "graph" | "spark"; size?: number }) {
  const p = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  switch (name) {
    case "fit":
      return (<svg {...p}><path d="M4 9V5a1 1 0 0 1 1-1h4M20 9V5a1 1 0 0 0-1-1h-4M4 15v4a1 1 0 0 0 1 1h4M20 15v4a1 1 0 0 1-1 1h-4" /></svg>);
    case "plus":
      return (<svg {...p}><path d="M12 5v14M5 12h14" /></svg>);
    case "minus":
      return (<svg {...p}><path d="M5 12h14" /></svg>);
    case "reset":
      return (<svg {...p}><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /></svg>);
    case "send":
      return (<svg {...p}><path d="M5 12h13M13 6l6 6-6 6" /></svg>);
    case "graph":
      return (<svg {...p}><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="8" r="2.5" /><circle cx="9" cy="18" r="2.5" /><path d="M8.2 7.3 15.6 8M7 8.4l1.4 7.2M10.8 16.4l5.6-6.5" /></svg>);
    case "spark":
      return (<svg {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.8 2.8M14.9 14.9l2.8 2.8M6.3 17.7l2.8-2.8M14.9 9.1l2.8-2.8" /></svg>);
  }
}
