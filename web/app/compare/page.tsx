"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Avatars, Chip, Icon, Panel, STICKY, Stamp, type StickyColor, TopBar } from "@/components/ui";
import { api, newSessionId } from "@/lib/api";
import type { ChatResponse, SourceName, Strategy } from "@/lib/types";

const COLORS: Record<string, StickyColor> = {
  agent: "pink",
  hybrid_ctx: "violet",
  hybrid: "green",
  v1_contextual: "orange",
  v1_stateless: "yellow",
  bm25: "blue",
  vector: "teal",
};

type Cell = { loading: boolean; result?: ChatResponse; error?: string; ms?: number };
type Round = { message: string; cells: Record<string, Cell> };

export default function Compare() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selected, setSelected] = useState<string[]>(["agent", "hybrid_ctx", "hybrid", "v1_contextual", "bm25", "vector"]);
  const [sessionId, setSessionId] = useState(newSessionId);
  const [rounds, setRounds] = useState<Round[]>([]);
  const [input, setInput] = useState("");

  useEffect(() => {
    api.strategies().then(setStrategies).catch(() => setStrategies([]));
  }, []);

  const busy = rounds.some((r) => Object.values(r.cells).some((c) => c.loading));

  async function run(message: string) {
    const text = message.trim();
    if (!text || busy) return;
    setInput("");
    const index = rounds.length;
    setRounds((rs) => [...rs, { message: text, cells: Object.fromEntries(selected.map((s) => [s, { loading: true }])) }]);
    await Promise.all(
      selected.map(async (strategy) => {
        const t0 = performance.now();
        let cell: Cell;
        try {
          const result = await api.chat({ session_id: sessionId, message: text, strategy, explain: strategy.startsWith("hybrid") });
          cell = { loading: false, result, ms: performance.now() - t0 };
        } catch (e) {
          cell = { loading: false, error: String(e), ms: performance.now() - t0 };
        }
        setRounds((rs) => rs.map((r, i) => (i === index ? { ...r, cells: { ...r.cells, [strategy]: cell } } : r)));
      }),
    );
  }

  async function reset() {
    await api.reset(sessionId).catch(() => undefined);
    setRounds([]);
    setSessionId(newSessionId());
  }

  const current = rounds[rounds.length - 1];

  return (
    <div className="dot-grid relative min-h-dvh w-full pb-16">
      <TopBar title="전략 비교" right={<Avatars />} />
      <div className="mx-auto max-w-[1500px] px-4 pt-24">
        <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-2xl font-extrabold tracking-tight">같은 대화, 여러 검색 전략</h1>
            <p className="mt-1 max-w-[720px] text-[13.5px] text-[var(--color-ink-2)]">
              v1(LLM이 Cypher를 직접 작성)과 v2(LLM은 슬롯만 채우고, 그래프·키워드·의미 검색 결과를 코드로 합치고 검증)를
              나란히 봅니다. 대화는 전략마다 따로 이어져요.
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {(strategies.length ? strategies : selected.map((name) => ({ name, description: "" }))).map((s) => {
              const on = selected.includes(s.name);
              const c = STICKY[COLORS[s.name] ?? "gray"];
              return (
                <button
                  key={s.name}
                  title={s.description}
                  disabled={busy}
                  onClick={() => setSelected((xs) => (on ? xs.filter((x) => x !== s.name) : [...xs, s.name]))}
                  className="rounded-full border px-3 py-1 text-[12px] font-semibold transition"
                  style={{ background: on ? c.bg : "var(--color-paper)", color: on ? c.ink : "var(--color-ink-3)", borderColor: on ? "transparent" : "var(--color-line)" }}
                >
                  {s.name}
                </button>
              );
            })}
          </div>
        </div>

        <Panel className="mb-6 flex items-center gap-2 p-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.nativeEvent.isComposing) run(input);
            }}
            placeholder={rounds.length ? "이어서 말해보세요 (예: 이런 느낌으로 더, 발라드는 빼줘)" : "예: 마음을 울리는 곡을 듣고 싶어"}
            className="h-11 flex-1 rounded-xl bg-[var(--color-canvas)] px-3 text-[14px] outline-none"
          />
          <button onClick={() => run(input)} disabled={busy || !input.trim()} className="grid h-11 w-11 place-items-center rounded-xl bg-[var(--color-accent)] text-white disabled:opacity-40" aria-label="실행">
            <Icon name="send" />
          </button>
          <button onClick={reset} className="grid h-11 w-11 place-items-center rounded-xl text-[var(--color-ink-2)] hover:bg-black/5" aria-label="새 대화" title="새 대화">
            <Icon name="reset" />
          </button>
        </Panel>

        {rounds.length > 1 ? (
          <div className="mb-4 flex flex-wrap items-center gap-2 text-[12.5px]">
            <span className="font-semibold text-[var(--color-ink-3)]">이전 턴</span>
            {rounds.slice(0, -1).map((r, i) => (
              <Chip key={i} color="yellow">{i + 1}. {r.message}</Chip>
            ))}
          </div>
        ) : null}

        {!current ? (
          <div className="grid place-items-center py-24 text-[14px] text-[var(--color-ink-3)]">질문을 입력하면 전략별 결과가 섹션으로 나란히 펼쳐져요</div>
        ) : (
          <div className="grid gap-5" style={{ gridTemplateColumns: `repeat(auto-fit, minmax(260px, 1fr))` }}>
            {Object.entries(current.cells).map(([name, cell]) => (
              <Column key={name} name={name} cell={cell} description={strategies.find((s) => s.name === name)?.description} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Column({ name, cell, description }: { name: string; cell: Cell; description?: string }) {
  const color = COLORS[name] ?? "gray";
  const c = STICKY[color];
  return (
    <section className="relative rounded-xl border-2 p-3 pt-4" style={{ borderColor: c.bg, background: `color-mix(in srgb, ${c.bg} 20%, transparent)` }}>
      <div className="absolute -top-3.5 left-3 rounded-md px-2 py-0.5 text-[13px] font-bold" style={{ background: c.bg, color: c.ink }}>
        {name}
      </div>
      <p className="mb-2 mt-1 line-clamp-2 text-[11.5px] text-[var(--color-ink-2)]" title={description}>{description}</p>
      {cell.loading ? (
        <div className="typing py-8 text-center text-2xl font-bold text-[var(--color-ink-3)]"><span>•</span><span>•</span><span>•</span></div>
      ) : cell.error ? (
        <div className="rounded-lg bg-[var(--color-pink)] p-3 text-[12px] text-[var(--color-pink-ink)]">{cell.error}</div>
      ) : cell.result ? (
        <>
          <div className="mb-2 flex items-center justify-between text-[11.5px] font-semibold text-[var(--color-ink-2)]">
            <span>{cell.result.songs.length}곡</span>
            <span>{cell.ms ? `${(cell.ms / 1000).toFixed(2)}s` : ""}</span>
          </div>
          <div className="mb-3 rounded-lg border border-[var(--color-line)] bg-[var(--color-paper)] p-2.5 text-[12.5px] leading-relaxed">
            {cell.result.reply || <span className="text-[var(--color-ink-3)]">(답변 없음)</span>}
          </div>
          {cell.result.trace?.cypher ? (
            <details className="mb-3 text-[11px]">
              <summary className="cursor-pointer font-semibold text-[var(--color-ink-2)]">LLM이 쓴 Cypher</summary>
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap rounded-md bg-[var(--color-ink)] p-2 text-[10.5px] text-white">{cell.result.trace.cypher}</pre>
            </details>
          ) : null}
          <ol className="space-y-2">
            {cell.result.songs.map((s, i) => (
              <li key={s.song_id} className="rounded-[3px] px-2.5 py-2" style={{ background: "var(--color-paper)", boxShadow: "var(--shadow-sticky)" }}>
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 w-4 shrink-0 text-right text-[11px] font-bold text-[var(--color-ink-3)]">{i + 1}</span>
                  <div className="min-w-0 flex-1">
                    <Link href={`/song/${s.song_id}`} className="block truncate text-[13px] font-bold hover:underline">{s.title}</Link>
                    <div className="truncate text-[11.5px] text-[var(--color-ink-2)]">{s.artists.join(", ")}</div>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {(Object.entries(s.sources ?? {}) as [SourceName, number][]).slice(0, 3).map(([src, r]) => (
                        <Stamp key={src} source={src} rank={r} />
                      ))}
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </>
      ) : null}
    </section>
  );
}
