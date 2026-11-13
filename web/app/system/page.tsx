"use client";

import { useEffect, useState } from "react";
import { Avatars, Panel, Sticky, type StickyColor, TopBar } from "@/components/ui";
import { ExperimentResults } from "@/components/results";
import { api } from "@/lib/api";
import type { EvalRun, Health } from "@/lib/types";

const DEP_LABEL: Record<string, string> = {
  neo4j: "Neo4j · 그래프",
  opensearch: "OpenSearch · BM25",
  postgres: "Postgres · pgvector",
  llm: "LLM (vLLM)",
  embedder: "임베딩 서버",
};

const STAT_LABEL: Record<string, string> = {
  songs: "곡", artists: "아티스트", albums: "앨범", playlists: "플레이리스트", tags: "태그",
  includes: "플리→곡 관계", has_tag: "태그 관계",
};

const PIPELINE: { title: string; body: string; color: StickyColor }[] = [
  { title: "1. 슬롯 추출", body: "LLM이 발화를 JSON 스키마로 (곡·아티스트·장르·분위기·제외·연도)", color: "yellow" },
  { title: "2. 개체 링킹", body: "정확 일치 → BM25(Nori) → 임베딩. '이런 느낌'은 직전 곡의 태그 안에서", color: "orange" },
  { title: "3. 대화 상태", body: "Neo4j VirtualQuery에 선호/제외 누적 (워커 무상태)", color: "green" },
  { title: "4. 후보 생성 (병렬)", body: "그래프 조건 · 플레이리스트 동시등장 · BM25 · Harrier 벡터", color: "blue" },
  { title: "5. 융합 + 검증", body: "RRF로 순위 결합 → Cypher로 제외·필수 조건 검사 → 부족하면 1회 완화", color: "violet" },
  { title: "6. 답변", body: "LLM은 결과 요약 문장만 작성", color: "pink" },
];

export default function System() {
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<Record<string, number> | null>(null);
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setErr(String(e)));
    api.stats().then((s) => setStats(s.graph)).catch(() => undefined);
    api.evalLatest().then((r) => setRuns(r.runs)).catch(() => undefined);
    const t = setInterval(() => api.health().then(setHealth).catch(() => undefined), 15000);
    return () => clearInterval(t);
  }, []);


  return (
    <div className="dot-grid min-h-dvh w-full pb-16">
      <TopBar title="시스템" right={<Avatars />} />
      <div className="mx-auto max-w-[1300px] space-y-10 px-4 pt-24">
        <Section title="서비스 상태" subtitle="15초마다 갱신">
          {err ? <Sticky color="pink">API에 연결할 수 없어요: {err}</Sticky> : null}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
            {health
              ? Object.entries(health.deps).map(([k, d], i) => (
                  <Sticky key={k} color={d.ok ? "green" : "pink"} tilt={[-1, 1, -0.5, 0.8, -1.2][i % 5]}>
                    <div className="text-[12px] font-bold">{DEP_LABEL[k] ?? k}</div>
                    <div className="mt-2 text-2xl font-extrabold">{d.ok ? "정상" : "중단"}</div>
                    <div className="mt-1 truncate text-[11px] text-[var(--color-ink-2)]" title={d.error}>
                      {d.ok ? `${d.ms} ms` : d.error}
                    </div>
                  </Sticky>
                ))
              : Array.from({ length: 5 }, (_, i) => <Sticky key={i} color="gray" className="h-[104px] animate-pulse"> </Sticky>)}
          </div>
          {health ? (
            <div className="mt-4 flex flex-wrap gap-2 text-[12.5px]">
              {Object.entries(health.embedding_versions).map(([entity, v]) => (
                <span key={entity} className="rounded-full bg-[var(--color-paper)] px-3 py-1 font-semibold shadow-sm">
                  활성 임베딩 · {entity}: <span className="text-[var(--color-accent)]">{v ?? "없음"}</span>
                </span>
              ))}
            </div>
          ) : null}
        </Section>

        <Section title="지식 그래프" subtitle="Melon Playlist Dataset → Neo4j">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-7">
            {stats
              ? Object.entries(stats).map(([k, v], i) => (
                  <Sticky key={k} color={(["yellow", "blue", "orange", "violet", "pink", "teal", "green"] as StickyColor[])[i % 7]} tilt={i % 2 ? 0.8 : -0.8}>
                    <div className="text-[11.5px] font-bold text-[var(--color-ink-2)]">{STAT_LABEL[k] ?? k}</div>
                    <div className="mt-1 text-[20px] font-extrabold tabular-nums">{v.toLocaleString()}</div>
                  </Sticky>
                ))
              : null}
          </div>
        </Section>

        <Section title="v2 검색 파이프라인" subtitle="LLM은 계획과 설명만, 고르는 건 코드">
          <div className="flex flex-wrap items-stretch gap-3">
            {PIPELINE.map((p, i) => (
              <div key={p.title} className="flex items-center gap-3">
                <Sticky color={p.color} className="w-[180px]" tilt={i % 2 ? 0.6 : -0.6}>
                  <div className="text-[13px] font-extrabold">{p.title}</div>
                  <div className="mt-1 text-[12px] leading-snug text-[var(--color-ink-2)]">{p.body}</div>
                </Sticky>
                {i < PIPELINE.length - 1 ? <span className="text-xl text-[var(--color-ink-3)]">→</span> : null}
              </div>
            ))}
          </div>
        </Section>

        <Section title="실험 결과" subtitle="results/*/ 에서 읽음 — 실제로 돌린 것만">
          <ExperimentResults />
        </Section>

        <Section title="평가 (make eval)" subtitle={runs.length ? "최근 실행 — 실행마다 대화 수·동시성이 다르니 같은 표 안에서만 비교" : "아직 실행 기록이 없어요"}>
          {runs.length ? (
            <div className="space-y-6">
              {[...runs].reverse().slice(0, 3).map((run) => (
                <div key={run.run_id}>
                  <p className="mb-2 text-[12px] text-[var(--color-ink-3)]">
                    <b className="text-[var(--color-ink)]">{run.run_id}</b> · commit {run.commit} · {run.conversations}개 대화
                    {run.concurrency ? ` · 동시성 ${run.concurrency}` : ""} · k={run.k}
                    {run.note ? ` · ${run.note}` : ""}
                  </p>
                  <Panel className="overflow-x-auto p-0">
                    <table className="w-full min-w-[720px] text-[12.5px]">
                      <thead className="bg-[var(--color-canvas)] text-left text-[11.5px] text-[var(--color-ink-2)]">
                        <tr>
                          {["전략", "질의 유형", "n", "hit@k", "MRR", "nDCG@k", "P@k"].map((h) => (
                            <th key={h} className="px-3 py-2 font-bold">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(run.strategies).flatMap(([st, body]) =>
                          Object.entries(body.metrics).map(([qt, m]) => (
                            <tr key={`${st}-${qt}`} className={`border-t border-[var(--color-line)] ${qt === "ALL" ? "font-bold" : ""}`}>
                              <td className="px-3 py-1.5">{st}</td>
                              <td className="px-3 py-1.5">{qt}</td>
                              <td className="px-3 py-1.5 tabular-nums">{m.n}</td>
                              <td className="px-3 py-1.5 tabular-nums">{m.hit.toFixed(3)}</td>
                              <td className="px-3 py-1.5 tabular-nums">{m.mrr.toFixed(3)}</td>
                              <td className="px-3 py-1.5 tabular-nums">{m.ndcg.toFixed(3)}</td>
                              <td className="px-3 py-1.5 tabular-nums">{m.precision.toFixed(3)}</td>
                            </tr>
                          )),
                        )}
                      </tbody>
                    </table>
                    <div className="flex flex-wrap gap-4 border-t border-[var(--color-line)] px-3 py-2 text-[12px] text-[var(--color-ink-2)]">
                      {Object.entries(run.strategies).map(([st, b]) => (
                        <span key={st}>
                          <b>{st}</b> p50 {b.latency_ms.p50}ms · p95 {b.latency_ms.p95}ms · 오류 {b.errors}/{b.turns}
                          {b.degraded_turns ? ` · 소스 누락 턴 ${b.degraded_turns}` : ""}
                        </span>
                      ))}
                    </div>
                  </Panel>
                </div>
              ))}
            </div>
          ) : (
            <Sticky color="gray" className="max-w-[420px] text-[13px]">
              <code>make eval</code>을 실행하면 여기에 전략별·질의 유형별 지표가 표시돼요.
            </Sticky>
          )}
        </Section>
      </div>
    </div>
  );
}

function Section({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-3 flex flex-wrap items-baseline gap-2">
        <h2 className="rounded-md bg-[var(--color-ink)] px-2 py-0.5 text-[14px] font-bold text-white">{title}</h2>
        {subtitle ? <span className="text-[12.5px] text-[var(--color-ink-3)]">{subtitle}</span> : null}
      </div>
      {children}
    </section>
  );
}
