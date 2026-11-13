"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import Link from "next/link";
import type { Pref, Song, SourceName } from "@/lib/types";
import { Chip, SOURCES, STICKY, Stamp, type StickyColor, topSource } from "./ui";

export const SONG_W = 184;
export const SONG_H = 172;

const hidden = { opacity: 0, width: 6, height: 6, border: 0 } as const;

function Handles() {
  return (
    <>
      <Handle type="target" position={Position.Left} style={hidden} isConnectable={false} />
      <Handle type="source" position={Position.Right} style={hidden} isConnectable={false} />
    </>
  );
}

// -------------------------------------------------------------- song sticky
export type SongData = { song: Song; rank?: number; tilt?: number; note?: string; color?: StickyColor };

export function SongNode({ data }: NodeProps<Node<SongData, "song">>) {
  const { song, rank, tilt = 0, note } = data;
  const top = topSource(song.sources);
  const color: StickyColor = data.color ?? (top ? SOURCES[top].color : "yellow");
  const c = STICKY[color];
  const sources = Object.entries(song.sources ?? {}).sort((a, b) => (a[1] ?? 0) - (b[1] ?? 0)) as [SourceName, number][];
  return (
    <div
      className="pop-in group relative flex flex-col gap-1.5 rounded-[3px] p-3 transition-transform hover:-translate-y-0.5"
      style={{ width: SONG_W, height: SONG_H, background: c.bg, boxShadow: "var(--shadow-sticky)", transform: `rotate(${tilt}deg)` }}
    >
      <Handles />
      {rank ? (
        <span className="absolute -left-2 -top-2 grid h-6 w-6 place-items-center rounded-full bg-[var(--color-ink)] text-[11px] font-bold text-white shadow">
          {rank}
        </span>
      ) : null}
      <Link
        href={`/song/${song.song_id}`}
        className="line-clamp-2 text-[14px] font-bold leading-[1.25] tracking-tight hover:underline"
        title={song.title}
      >
        {song.title}
      </Link>
      <div className="truncate text-[12px] font-medium" style={{ color: c.ink }} title={song.artists.join(", ")}>
        {song.artists.join(", ") || "알 수 없는 아티스트"}
      </div>
      <div className="flex flex-wrap gap-1 overflow-hidden" style={{ maxHeight: 40 }}>
        {song.genres.slice(0, 2).map((g) => (
          <Chip key={g} color="gray">{g}</Chip>
        ))}
        {song.tags.slice(0, 2).map((t) => (
          <Chip key={t} color="gray">#{t}</Chip>
        ))}
      </div>
      <div className="mt-auto flex items-end justify-between gap-1">
        <div className="flex flex-wrap gap-1">
          {sources.slice(0, 3).map(([s, r]) => (
            <Stamp key={s} source={s} rank={r} />
          ))}
          {note ? <span className="text-[10px] font-semibold text-[var(--color-ink-2)]">{note}</span> : null}
        </div>
        <span className="shrink-0 text-[10px] font-medium text-[var(--color-ink-3)]" title="이 곡이 담긴 플레이리스트 수">
          {song.year ? `${song.year} · ` : ""}♥{song.playlist_count.toLocaleString()}
        </span>
      </div>
    </div>
  );
}

// -------------------------------------------------------------- section
export type SectionData = { title: string; subtitle?: string; color?: StickyColor; width: number; height: number };

export function SectionNode({ data }: NodeProps<Node<SectionData, "section">>) {
  const c = STICKY[data.color ?? "gray"];
  return (
    <div
      className="rounded-xl border-2"
      style={{
        width: data.width,
        height: data.height,
        borderColor: c.bg,
        background: `color-mix(in srgb, ${c.bg} 22%, transparent)`,
      }}
    >
      <Handles />
      <div className="absolute -top-[30px] left-0 flex max-w-full items-center gap-2">
        <span className="truncate rounded-md px-2 py-0.5 text-[13px] font-bold" style={{ background: c.bg, color: c.ink }}>
          {data.title}
        </span>
        {data.subtitle ? <span className="truncate text-[12px] text-[var(--color-ink-3)]">{data.subtitle}</span> : null}
      </div>
    </div>
  );
}

// -------------------------------------------------------------- preference sticky
export type PrefData = { pref: Pref; positive: boolean };

const KIND: Record<Pref["kind"], string> = { artist: "아티스트", genre: "장르", tag: "태그", song: "곡" };

export function PrefNode({ data }: NodeProps<Node<PrefData, "pref">>) {
  const color: StickyColor = data.positive ? "green" : "pink";
  const c = STICKY[color];
  return (
    <div
      className="pop-in flex w-[150px] flex-col gap-0.5 rounded-[3px] px-3 py-2"
      style={{ background: c.bg, boxShadow: "var(--shadow-sticky)" }}
    >
      <Handles />
      <span className="text-[10px] font-bold uppercase tracking-wide" style={{ color: c.ink }}>
        {data.positive ? "좋아요" : "빼주세요"} · {KIND[data.pref.kind]}
      </span>
      <span className="truncate text-[14px] font-bold" title={data.pref.name}>
        {data.pref.kind === "tag" ? `#${data.pref.name}` : data.pref.name}
      </span>
    </div>
  );
}

// -------------------------------------------------------------- free-form note
export type NoteData = {
  kind: "prompt" | "reply" | "trace" | "empty";
  title?: string;
  body: React.ReactNode;
  width?: number;
};

export function NoteNode({ data }: NodeProps<Node<NoteData, "note">>) {
  const styles: Record<NoteData["kind"], { bg: string; border?: string }> = {
    prompt: { bg: "var(--color-yellow)" },
    reply: { bg: "var(--color-paper)", border: "var(--color-line)" },
    trace: { bg: "var(--color-blue)" },
    empty: { bg: "var(--color-paper)", border: "var(--color-line)" },
  };
  const s = styles[data.kind];
  return (
    <div
      className={`pop-in relative ${data.kind === "reply" || data.kind === "empty" ? "rounded-xl" : "rounded-[3px]"} p-3`}
      style={{
        width: data.width ?? 240,
        background: s.bg,
        border: s.border ? `1px solid ${s.border}` : undefined,
        boxShadow: "var(--shadow-sticky)",
      }}
    >
      <Handles />
      {data.kind === "prompt" ? (
        <span className="absolute -top-3 left-3 rounded-full bg-[#ff7262] px-2 py-0.5 text-[10px] font-bold text-white shadow">나</span>
      ) : null}
      {data.kind === "reply" ? (
        <span className="absolute -top-3 left-3 rounded-full bg-[var(--color-accent)] px-2 py-0.5 text-[10px] font-bold text-white shadow">
          AI 큐레이터
        </span>
      ) : null}
      {data.title ? <div className="mb-1 text-[11px] font-bold uppercase tracking-wide text-[var(--color-ink-2)]">{data.title}</div> : null}
      <div className="text-[13px] leading-relaxed">{data.body}</div>
    </div>
  );
}

// -------------------------------------------------------------- graph entity (artist/genre/tag/album)
export type EntityData = { kind: "artist" | "album" | "genre" | "tag"; label: string; sub?: string };

const ENTITY: Record<EntityData["kind"], { color: StickyColor; round: string; name: string }> = {
  artist: { color: "violet", round: "rounded-full", name: "아티스트" },
  album: { color: "teal", round: "rounded-lg", name: "앨범" },
  genre: { color: "green", round: "rounded-full", name: "장르" },
  tag: { color: "pink", round: "rounded-full", name: "태그" },
};

export function EntityNode({ data }: NodeProps<Node<EntityData, "entity">>) {
  const e = ENTITY[data.kind];
  const c = STICKY[e.color];
  return (
    <div
      className={`pop-in flex max-w-[220px] items-center gap-2 border border-black/5 px-3 py-1.5 ${e.round}`}
      style={{ background: c.bg, boxShadow: "var(--shadow-sticky)" }}
    >
      <Handles />
      <span className="text-[10px] font-bold" style={{ color: c.ink }}>{e.name}</span>
      <span className="truncate text-[13px] font-bold">{data.kind === "tag" ? `#${data.label}` : data.label}</span>
      {data.sub ? <span className="text-[11px] text-[var(--color-ink-2)]">{data.sub}</span> : null}
    </div>
  );
}

export const nodeTypes = { song: SongNode, section: SectionNode, pref: PrefNode, note: NoteNode, entity: EntityNode };
