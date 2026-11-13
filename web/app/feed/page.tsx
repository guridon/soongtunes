"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Avatars, Chip, Icon, Stamp, Sticky, type StickyColor, SOURCES, STICKY, TopBar, topSource } from "@/components/ui";
import { api } from "@/lib/api";
import type { ChatResponse, Song, SourceName } from "@/lib/types";

const SIGNAL_LABEL: Record<string, string> = {
  fusion: "후보 합의",
  tag_pref: "취향 태그",
  condition_tags: "조건 태그",
  popularity: "인기",
};

export default function Feed() {
  const [users, setUsers] = useState<string[]>([]);
  const [user, setUser] = useState("");
  const [recent, setRecent] = useState<Song[]>([]);
  const [prefs, setPrefs] = useState<[string, number][]>([]);
  const [feed, setFeed] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (id: string) => {
    if (!id) return;
    setLoading(true);
    setError(null);
    setUser(id);
    try {
      const [features, result] = await Promise.all([
        api.features([id], ["user_recent_plays", "user_tag_pref"]),
        api.feed(id, 20),
      ]);
      const f = features[id] ?? {};
      const plays = ((f.user_recent_plays as [string, number][] | null) ?? []).slice(0, 10);
      const tagPref = (f.user_tag_pref as Record<string, number> | null) ?? {};
      setPrefs(Object.entries(tagPref).sort((a, b) => b[1] - a[1]).slice(0, 10));
      setRecent(plays.length ? await api.songs(plays.map(([s]) => s)) : []);
      setFeed(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    api.sampleUsers(8).then((u) => {
      setUsers(u);
      if (u[0]) void load(u[0]);
    }).catch((e) => setError(String(e)));
  }, [load]);

  const trace = feed?.trace;

  return (
    <div className="dot-grid min-h-dvh w-full pb-16">
      <TopBar title="홈 피드" right={<Avatars />} />
      <div className="mx-auto max-w-[1400px] px-4 pt-24">
        <div className="mb-6 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-2xl font-extrabold tracking-tight">말하지 않아도 오늘의 추천</h1>
            <p className="mt-1 max-w-[720px] text-[13.5px] text-[var(--color-ink-2)]">
              발화 없이 피처 스토어의 최근 재생과 취향 태그만으로 만든 피드예요. 같은 에이전트가
              동시등장 · 취향 태그 · 유저 임베딩 · 인기 후보를 모아 랭커로 고르고, 아티스트가 겹치지 않게 섞어요.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {users.map((u) => (
              <button
                key={u}
                onClick={() => load(u)}
                className={`rounded-full border px-3 py-1 text-[12px] font-semibold ${u === user ? "border-transparent bg-[var(--color-accent)] text-white" : "border-[var(--color-line)] bg-[var(--color-paper)] text-[var(--color-ink-2)]"}`}
              >
                {u}
              </button>
            ))}
            <button
              onClick={() => api.sampleUsers(8).then(setUsers)}
              className="grid h-8 w-8 place-items-center rounded-full bg-[var(--color-paper)] text-[var(--color-ink-2)] shadow-sm"
              title="다른 유저"
              aria-label="다른 유저"
            >
              <Icon name="reset" size={15} />
            </button>
          </div>
        </div>

        {error ? <Sticky color="pink" className="mb-4 text-[13px]">{error}</Sticky> : null}

        <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
          <div className="space-y-6">
            <section>
              <SectionTitle color="yellow">최근 재생</SectionTitle>
              <div className="space-y-2">
                {recent.map((s, i) => (
                  <Sticky key={s.song_id} color="yellow" tilt={i % 2 ? 0.6 : -0.6} className="py-2">
                    <Link href={`/song/${s.song_id}`} className="block truncate text-[13px] font-bold hover:underline">{s.title}</Link>
                    <div className="truncate text-[11.5px] text-[var(--color-yellow-ink)]">{s.artists.join(", ")}</div>
                  </Sticky>
                ))}
                {!recent.length && !loading ? <p className="text-[12.5px] text-[var(--color-ink-3)]">재생 기록이 없어요</p> : null}
              </div>
            </section>
            <section>
              <SectionTitle color="green">취향 태그</SectionTitle>
              <div className="flex flex-wrap gap-1.5">
                {prefs.map(([t, w]) => (
                  <span key={t} className="rounded-full px-2.5 py-1 text-[12px] font-semibold" style={{ background: STICKY.green.bg, color: STICKY.green.ink, opacity: 0.55 + Math.min(0.45, w * 4) }}>
                    #{t} <span className="opacity-60">{(w * 100).toFixed(0)}</span>
                  </span>
                ))}
              </div>
            </section>
            {trace ? (
              <Sticky color="blue" className="text-[12px] leading-relaxed">
                <div className="mb-1 text-[11px] font-bold uppercase tracking-wide text-[var(--color-ink-2)]">어떻게 골랐나</div>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(trace.candidates ?? {}).map(([s, n]) =>
                    s in SOURCES ? (<span key={s} className="inline-flex items-center gap-0.5"><Stamp source={s as SourceName} /><span className="text-[10px]">{n}</span></span>) : null,
                  )}
                </div>
                {(trace as { timed_out?: string[] }).timed_out?.length ? (
                  <div className="mt-1 text-[var(--color-pink-ink)]">시간 초과로 제외: {(trace as { timed_out?: string[] }).timed_out!.join(", ")}</div>
                ) : null}
                <div className="mt-1 text-[var(--color-blue-ink)]">
                  {Object.entries(trace.timings_ms ?? {}).map(([k, v]) => `${k} ${Math.round(v)}ms`).join(" · ")}
                </div>
              </Sticky>
            ) : null}
          </div>

          <section>
            <SectionTitle color="violet">{loading ? "추천을 만드는 중…" : `오늘의 추천 ${feed?.songs.length ?? 0}곡`}</SectionTitle>
            <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))" }}>
              {(feed?.songs ?? []).map((s, i) => {
                const top = topSource(s.sources);
                const color: StickyColor = top ? SOURCES[top].color : "yellow";
                return (
                  <Sticky key={s.song_id} color={color} tilt={((i * 37) % 5) - 2} className="pop-in flex min-h-[180px] flex-col gap-1.5">
                    <div className="flex items-start justify-between gap-2">
                      <Link href={`/song/${s.song_id}`} className="line-clamp-2 text-[14px] font-bold leading-tight hover:underline">{s.title}</Link>
                      <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-[var(--color-ink)] text-[11px] font-bold text-white">{i + 1}</span>
                    </div>
                    <div className="truncate text-[12px] font-medium" style={{ color: STICKY[color].ink }}>{s.artists.join(", ")}</div>
                    <div className="flex flex-wrap gap-1">
                      {s.tags.slice(0, 3).map((t) => (<Chip key={t}>#{t}</Chip>))}
                    </div>
                    <div className="mt-auto space-y-0.5 pt-1">
                      {Object.entries(s.signals ?? {}).map(([k, v]) => (
                        <div key={k} className="flex items-center gap-1.5 text-[10px] text-[var(--color-ink-2)]">
                          <span className="w-12 shrink-0">{SIGNAL_LABEL[k] ?? k}</span>
                          <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-black/10">
                            <span className="block h-full rounded-full bg-[var(--color-ink)]" style={{ width: `${Math.min(100, Math.max(0, v * 100))}%` }} />
                          </span>
                        </div>
                      ))}
                      <div className="flex flex-wrap gap-1 pt-1">
                        {(Object.entries(s.sources ?? {}) as [SourceName, number][]).slice(0, 3).map(([src, r]) => (<Stamp key={src} source={src} rank={r} />))}
                      </div>
                    </div>
                  </Sticky>
                );
              })}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function SectionTitle({ color, children }: { color: StickyColor; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <span className="rounded-md px-2 py-0.5 text-[13px] font-bold" style={{ background: STICKY[color].bg, color: STICKY[color].ink }}>{children}</span>
    </div>
  );
}

