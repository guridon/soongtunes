"use client";

import {
  Background,
  BackgroundVariant,
  type Edge,
  MarkerType,
  type Node,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { nodeTypes, SONG_H, SONG_W } from "@/components/nodes";
import { Avatars, Icon, Panel, SOURCES, Stamp, TopBar } from "@/components/ui";
import { api, newSessionId } from "@/lib/api";
import type { ChatResponse, Linked, SourceName } from "@/lib/types";

type Turn = { id: string; message: string; result?: ChatResponse; error?: string };

const EXAMPLES = [
  "비 오는 날 듣기 좋은 잔잔한 노래",
  "아이유 노래 중에 신나는 거 5곡",
  "90년대 발라드 말고 댄스",
  "He Is We의 Our July In The Rain 찾아줘",
];

const COLS = 5;
const GAP = 22;
const PAD = 22;
const BAND_GAP = 120;
const LEFT_X = 250;
const RESULTS_X = 560;
const STORAGE_KEY = "stunes.board.v2";

function bandHeight(n: number) {
  const rows = Math.max(1, Math.ceil(n / COLS));
  return Math.max(rows * (SONG_H + GAP) + PAD * 2 - GAP, 430);
}

function linkedSummary(linked?: Record<string, Linked[] | string[]>) {
  if (!linked) return null;
  const names: Record<string, string> = {
    songs: "곡", artists: "아티스트", genres: "장르", tags: "태그",
    neg_artists: "제외 아티스트", neg_genres: "제외 장르", neg_tags: "제외 태그", unlinked: "못 찾음",
  };
  return Object.entries(linked).map(([k, items]) => (
    <div key={k} className="truncate">
      <b>{names[k] ?? k}</b>{" "}
      {(items as (Linked | string)[])
        .map((x) => (typeof x === "string" ? x : `${x.name}${x.via !== "exact" ? ` (${x.via})` : ""}`))
        .join(", ")}
    </div>
  ));
}

function buildBoard(turns: Turn[]) {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  let y = 0;
  const last = [...turns].reverse().find((t) => t.result);

  // conditions accumulated in the graph state (VirtualQuery)
  const prefs = last?.result
    ? [
        ...last.result.state.positive.map((p) => ({ p, positive: true })),
        ...last.result.state.negative.map((p) => ({ p, positive: false })),
      ]
    : [];
  const condH = Math.max(140, prefs.length * 70 + PAD * 2 + 10);
  nodes.push({
    id: "conditions",
    type: "section",
    position: { x: 0, y: 0 },
    data: { title: "대화 조건", subtitle: "VirtualQuery", color: "green", width: 194, height: condH },
    draggable: true,
    selectable: false,
  });
  if (!prefs.length) {
    nodes.push({
      id: "cond-empty",
      type: "note",
      parentId: "conditions",
      position: { x: 18, y: 22 },
      data: { kind: "empty", body: <span className="text-[12px] text-[var(--color-ink-3)]">말한 조건이 여기에 쌓여요</span>, width: 158 },
      draggable: false,
    });
  }
  prefs.forEach(({ p, positive }, i) => {
    nodes.push({
      id: `pref-${p.kind}-${p.key}`,
      type: "pref",
      parentId: "conditions",
      position: { x: 22, y: PAD + i * 70 },
      data: { pref: p, positive },
      draggable: false,
    });
  });

  turns.forEach((t, i) => {
    const songs = t.result?.songs ?? [];
    const h = bandHeight(songs.length);
    const latest = i === turns.length - 1;
    nodes.push({
      id: `prompt-${t.id}`,
      type: "note",
      position: { x: LEFT_X, y },
      data: { kind: "prompt", body: <span className="font-semibold">{t.message}</span>, width: 260 },
    });
    nodes.push({
      id: `reply-${t.id}`,
      type: "note",
      position: { x: LEFT_X, y: y + 112 },
      data: {
        kind: "reply",
        width: 260,
        body: t.error ? (
          <span className="text-[var(--color-pink-ink)]">요청 실패: {t.error}</span>
        ) : t.result ? (
          t.result.reply
        ) : (
          <span className="typing text-lg font-bold text-[var(--color-ink-3)]"><span>•</span><span>•</span><span>•</span></span>
        ),
      },
    });
    if (t.result?.trace) {
      const tr = t.result.trace;
      nodes.push({
        id: `trace-${t.id}`,
        type: "note",
        position: { x: LEFT_X, y: y + 250 },
        data: {
          kind: "trace",
          title: "어떻게 찾았나",
          width: 260,
          body: (
            <div className="space-y-1 text-[11.5px] leading-snug">
              {linkedSummary(tr.linked)}
              {tr.candidates ? (
                <div className="flex flex-wrap items-center gap-1 pt-1">
                  {Object.entries(tr.candidates).map(([s, n]) =>
                    s in SOURCES ? (
                      <span key={s} className="inline-flex items-center gap-0.5">
                        <Stamp source={s as SourceName} /> <span className="text-[10px]">{n}</span>
                      </span>
                    ) : null,
                  )}
                </div>
              ) : null}
              {tr.relaxed?.length ? <div>완화: {tr.relaxed.join(", ")}</div> : null}
              {tr.timings_ms ? (
                <div className="text-[var(--color-blue-ink)]">
                  {Object.entries(tr.timings_ms)
                    .map(([k, v]) => `${k} ${Math.round(v)}ms`)
                    .join(" · ")}
                </div>
              ) : null}
            </div>
          ),
        },
      });
    }

    const sid = `results-${t.id}`;
    const width = COLS * (SONG_W + GAP) + PAD * 2 - GAP;
    nodes.push({
      id: sid,
      type: "section",
      position: { x: RESULTS_X, y },
      data: {
        title: `Turn ${i + 1} · 추천 ${songs.length}곡`,
        subtitle: t.result?.trace?.mode ? `mode: ${t.result.trace.mode}` : undefined,
        color: latest ? "violet" : "gray",
        width,
        height: h,
      },
      width,
      height: h,
      selectable: false,
    });
    songs.forEach((song, j) => {
      nodes.push({
        id: `${sid}-${song.song_id}`,
        type: "song",
        parentId: sid,
        position: { x: PAD + (j % COLS) * (SONG_W + GAP), y: PAD + Math.floor(j / COLS) * (SONG_H + GAP) },
        data: { song, rank: j + 1, tilt: ((j * 37) % 5) - 2 },
        width: SONG_W,
        height: SONG_H,
      });
    });
    edges.push({
      id: `e-${t.id}`,
      source: `prompt-${t.id}`,
      target: sid,
      animated: latest && !t.result,
      markerEnd: { type: MarkerType.ArrowClosed, color: "#9a978e" },
      style: { strokeDasharray: latest ? undefined : "4 4" },
    });
    if (latest && t.result && prefs.length) {
      edges.push({
        id: `e-cond-${t.id}`,
        source: "conditions",
        target: `prompt-${t.id}`,
        style: { strokeDasharray: "6 5" },
        label: "누적 조건",
        labelStyle: { fontSize: 11, fontWeight: 600 },
      });
    }
    y += h + BAND_GAP;
  });
  return { nodes, edges };
}

function Board() {
  const flow = useReactFlow();
  const [sessionId, setSessionId] = useState<string>("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // `/?q=...` starts a fresh board with that question (shareable links)
    const q = new URLSearchParams(window.location.search).get("q");
    if (q) {
      setSessionId(newSessionId());
      setPendingQuery(q);
      window.history.replaceState(null, "", "/");
      return;
    }
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
      if (saved?.sessionId) {
        setSessionId(saved.sessionId);
        setTurns(saved.turns ?? []);
        return;
      }
    } catch {
      /* storage unavailable: start fresh */
    }
    setSessionId(newSessionId());
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ sessionId, turns: turns.filter((t) => t.result) }));
    } catch {
      /* ignore */
    }
  }, [sessionId, turns]);

  const { nodes, edges } = useMemo(() => buildBoard(turns), [turns]);

  const focusTurn = useCallback(
    (index: number) => {
      let y = 0;
      for (let i = 0; i < index; i++) y += bandHeight(turns[i]?.result?.songs.length ?? 0) + BAND_GAP;
      const h = bandHeight(turns[index]?.result?.songs.length ?? 0);
      flow.fitBounds({ x: 0, y: y - 40, width: RESULTS_X + COLS * (SONG_W + GAP) + 60, height: h + 80 }, { duration: 500, padding: 0.08 });
    },
    [flow, turns],
  );

  useEffect(() => {
    if (turns.length) {
      const t = setTimeout(() => focusTurn(turns.length - 1), 60);
      return () => clearTimeout(t);
    }
  }, [turns.length, focusTurn]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  async function send(message: string) {
    const text = message.trim();
    if (!text || busy || !sessionId) return;
    const id = crypto.randomUUID().slice(0, 8);
    setTurns((ts) => [...ts, { id, message: text }]);
    setInput("");
    setBusy(true);
    try {
      const result = await api.chat({ session_id: sessionId, message: text, strategy: "hybrid_ctx", explain: true });
      setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, result } : t)));
    } catch (e) {
      setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, error: String(e) } : t)));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (pendingQuery && sessionId) {
      setPendingQuery(null);
      void send(pendingQuery);
    }
    // send is recreated every render; running once per pending query is what matters
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingQuery, sessionId]);

  async function reset() {
    try {
      await api.reset(sessionId, "hybrid_ctx");
    } catch {
      /* a dead session is fine to drop */
    }
    setTurns([]);
    setSessionId(newSessionId());
    flow.setViewport({ x: 80, y: 120, zoom: 0.9 }, { duration: 300 });
  }

  return (
    <div className="relative h-dvh w-full overflow-hidden">
      <TopBar title="내 음악 보드" right={<Avatars />} />
      <div className="absolute inset-0 md:right-[380px]">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          defaultViewport={{ x: 80, y: 120, zoom: 0.9 }}
          minZoom={0.2}
          maxZoom={1.8}
          proOptions={{ hideAttribution: true }}
          nodesConnectable={false}
          panOnScroll
          selectionOnDrag={false}
        >
          <Background variant={BackgroundVariant.Dots} gap={22} size={1.6} color="#cfcdc6" bgColor="#f3f2ef" />
        </ReactFlow>
        {!turns.length ? (
          <div className="pointer-events-none absolute inset-0 grid place-items-center p-6">
            <div className="pointer-events-auto max-w-[520px] text-center">
              <div className="mb-2 text-[13px] font-semibold text-[var(--color-accent)]">대화로 찾는 음악 보드</div>
              <h1 className="text-3xl font-extrabold tracking-tight">어떤 노래가 듣고 싶어요?</h1>
              <p className="mt-2 text-[14px] text-[var(--color-ink-2)]">
                멜론 플레이리스트 11만 개로 만든 지식 그래프에서 찾아드려요. 조건은 대화하면서 계속 쌓여요.
              </p>
              <div className="mt-6 flex flex-wrap justify-center gap-3">
                {EXAMPLES.map((e, i) => (
                  <button
                    key={e}
                    onClick={() => send(e)}
                    className="rounded-[3px] px-3 py-2.5 text-left text-[13px] font-semibold transition hover:-translate-y-0.5"
                    style={{
                      background: ["var(--color-yellow)", "var(--color-green)", "var(--color-pink)", "var(--color-blue)"][i % 4],
                      boxShadow: "var(--shadow-sticky)",
                      transform: `rotate(${[-1.5, 1, -0.5, 1.5][i % 4]}deg)`,
                    }}
                  >
                    {e}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : null}

        <div className="absolute bottom-4 left-1/2 z-10 -translate-x-1/2">
          <Panel className="flex items-center gap-1 px-1.5 py-1.5">
            <ToolButton label="맞춤 보기" onClick={() => flow.fitView({ duration: 400, padding: 0.1 })}><Icon name="fit" /></ToolButton>
            <ToolButton label="확대" onClick={() => flow.zoomIn({ duration: 200 })}><Icon name="plus" /></ToolButton>
            <ToolButton label="축소" onClick={() => flow.zoomOut({ duration: 200 })}><Icon name="minus" /></ToolButton>
            <div className="mx-1 h-6 w-px bg-[var(--color-line)]" />
            <ToolButton label="새 대화" onClick={reset}><Icon name="reset" /></ToolButton>
            <div className="mx-1 hidden h-6 w-px bg-[var(--color-line)] sm:block" />
            <div className="hidden items-center gap-1 pr-1 sm:flex">
              {(["lookup", "graph", "cooccur", "bm25", "vector"] as SourceName[]).map((s) => (
                <Stamp key={s} source={s} />
              ))}
            </div>
          </Panel>
        </div>
      </div>

      <ChatDock
        turns={turns}
        input={input}
        setInput={setInput}
        busy={busy}
        onSend={send}
        onFocus={focusTurn}
        listRef={listRef}
      />
    </div>
  );
}

function ToolButton({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      title={label}
      aria-label={label}
      onClick={onClick}
      className="grid h-9 w-9 place-items-center rounded-xl text-[var(--color-ink-2)] transition hover:bg-black/5 hover:text-[var(--color-ink)]"
    >
      {children}
    </button>
  );
}

function ChatDock({
  turns, input, setInput, busy, onSend, onFocus, listRef,
}: {
  turns: Turn[];
  input: string;
  setInput: (s: string) => void;
  busy: boolean;
  onSend: (s: string) => void;
  onFocus: (i: number) => void;
  listRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <aside className="absolute inset-x-3 bottom-3 z-20 flex max-h-[46dvh] flex-col md:inset-x-auto md:bottom-3 md:right-3 md:top-[68px] md:max-h-none md:w-[364px]">
      <Panel className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="flex items-center justify-between border-b border-[var(--color-line)] px-4 py-3">
          <div>
            <div className="text-[14px] font-bold">AI 큐레이터</div>
            <div className="text-[11.5px] text-[var(--color-ink-3)]">그래프 + 키워드 + 의미 검색을 한 번에</div>
          </div>
          <Icon name="spark" />
        </div>
        <div ref={listRef} className="scrollbar-thin flex-1 space-y-3 overflow-y-auto px-4 py-3">
          {!turns.length ? (
            <p className="text-[13px] text-[var(--color-ink-3)]">
              예: “새벽에 듣기 좋은 인디”, “이런 느낌으로 더”, “발라드는 빼줘”. 이어지는 말은 앞 조건에 더해져요.
            </p>
          ) : null}
          {turns.map((t, i) => (
            <div key={t.id} className="space-y-1.5">
              <div className="ml-8 rounded-2xl rounded-br-md bg-[var(--color-ink)] px-3 py-2 text-[13px] text-white">{t.message}</div>
              <button
                onClick={() => onFocus(i)}
                className="block w-full rounded-2xl rounded-bl-md border border-[var(--color-line)] bg-[var(--color-canvas)] px-3 py-2 text-left text-[13px] transition hover:border-[var(--color-accent)]"
              >
                {t.error ? (
                  <span className="text-[var(--color-pink-ink)]">요청 실패 — {t.error}</span>
                ) : t.result ? (
                  <>
                    <span>{t.result.reply}</span>
                    <span className="mt-1 block text-[11px] font-semibold text-[var(--color-accent)]">
                      보드에서 {t.result.songs.length}곡 보기 →
                      {t.result.trace?.timings_ms?.total ? ` ${(t.result.trace.timings_ms.total / 1000).toFixed(1)}s` : ""}
                    </span>
                  </>
                ) : (
                  <span className="typing text-lg font-bold text-[var(--color-ink-3)]"><span>•</span><span>•</span><span>•</span></span>
                )}
              </button>
            </div>
          ))}
        </div>
        <form
          className="flex items-end gap-2 border-t border-[var(--color-line)] p-3"
          onSubmit={(e) => {
            e.preventDefault();
            onSend(input);
          }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                onSend(input);
              }
            }}
            rows={2}
            placeholder="듣고 싶은 노래를 말해보세요"
            className="min-h-[44px] flex-1 resize-none rounded-xl border border-[var(--color-line)] bg-[var(--color-canvas)] px-3 py-2 text-[14px] outline-none focus:border-[var(--color-accent)]"
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            aria-label="보내기"
            className="grid h-11 w-11 place-items-center rounded-xl bg-[var(--color-accent)] text-white transition disabled:opacity-40"
          >
            <Icon name="send" />
          </button>
        </form>
      </Panel>
    </aside>
  );
}

export default function Page() {
  return (
    <ReactFlowProvider>
      <Board />
    </ReactFlowProvider>
  );
}
