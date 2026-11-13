"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Panel, Sticky } from "./ui";

 
type Row = Record<string, any>;

function Table({ head, rows }: { head: string[]; rows: (string | number | null | undefined)[][] }) {
  return (
    <Panel className="overflow-x-auto p-0">
      <table className="w-full min-w-[640px] text-[12.5px]">
        <thead className="bg-[var(--color-canvas)] text-left text-[11.5px] text-[var(--color-ink-2)]">
          <tr>{head.map((h) => <th key={h} className="px-3 py-2 font-bold">{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-[var(--color-line)]">
              {r.map((c, j) => <td key={j} className="px-3 py-1.5 tabular-nums">{c ?? "—"}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

export function ExperimentResults() {
  const [data, setData] = useState<Record<string, Row[]> | null>(null);
  useEffect(() => {
    api.results().then(setData).catch(() => setData({}));
  }, []);
  if (!data) return null;

  const ingest = (data.ingest ?? []).filter((r) => r.config?.label !== "smoke");
  const tp = ingest.filter((r) => r.kind === "throughput");
  const chaos = ingest.filter((r) => r.kind === "chaos");
  const feats = data.features ?? [];
  // results/embedding also holds follow-up files (served-probes-*) without the variant table
  const emb = (data.embedding ?? []).filter((r) => r.variants).at(-1);
  const feed = (data.feed ?? []).at(-1);
  const tagVal = (data.llm_tags ?? []).filter((r) => r.thresholds && r.songs >= 100).at(-1);
  const tagApply = (data.llm_tags ?? []).filter((r) => r.stored != null).at(-1);

  return (
    <div className="space-y-8">
      <Block title="Phase 2 · 커밋 전략별 처리량" note="2만 건 백로그를 비우는 속도. 같은 머신·같은 배경 부하에서의 상대 비교">
        {tp.length ? (
          <Table
            head={["실험", "모드", "파티션×컨슈머", "batch", "msg/s", "중복", "유실", "효과 일치", "DLQ"]}
            rows={tp.map((r) => [r.config.label, r.config.mode, `${r.config.partitions}×${r.config.consumers}`, r.config.batch,
              r.result.throughput_msg_s, r.result.duplicates, r.result.lost, r.result.effects_match ? "✓" : "✗", r.result.dlq])}
          />
        ) : <Empty />}
      </Block>
      <Block title="Phase 2 · kill -9 장애 주입" note="라이브 발행 중 컨슈머를 5번 강제 종료 → 유실(보낸 것 − 처리된 것)과 중복">
        {chaos.length ? (
          <Table
            head={["모드", "rate", "kill", "보낸 이벤트", "유실", "중복", "e2e p95 ms", "효과 일치"]}
            rows={chaos.map((r) => [r.config.mode, r.config.rate, r.result.kills_at_s?.length, r.result.sent_total,
              r.result.lost, r.result.duplicates, r.result.e2e_ms?.p95, r.result.effects_match ? "✓" : "✗"])}
          />
        ) : <Empty />}
      </Block>
      <Block title="Phase 3 · 캐시 프로필" note="C0 캐시 없음 → C1 Redis → C2 +지터 → C3 +XFetch → C4 +네거티브 → C5 +L1">
        {feats.length ? (
          <Table
            head={["프로필", "실험", "rps", "p50 ms", "p99 ms", "L3 SQL/s 평균", "L3 SQL/s 최대", "L2 hit", "L1 hit", "negative hit", "오류"]}
            rows={feats.map((r) => [r.profile, r.label, r.rps, r.latency_ms?.p50, r.latency_ms?.p99, r.l3_qps?.mean, r.l3_qps?.max,
              pct(r.hit_ratio?.l2_hit), pct(r.hit_ratio?.l1_hit), pct(r.hit_ratio?.l2_negative_hit), r.errors])}
          />
        ) : <Empty />}
      </Block>
      <Block title="Phase 4 · 임베딩 크기 vs 리콜" note={emb ? `${emb.version} · ${emb.subset.toLocaleString()}곡 · 쿼리 ${emb.queries}개 · 정확 검색 대비 recall@${emb.k}` : ""}>
        {emb ? (
          <>
            <Table
              head={["변형", "recall", "p50 ms", "p95 ms", "bytes/vector", "Redis bytes/key", "index MB"]}
              rows={Object.entries(emb.variants as Record<string, Row>).map(([k, v]) => [k, v.recall_at_k, v.p50_ms, v.p95_ms, v.bytes_per_vector, v.redis_bytes_per_key, v.index_mb])}
            />
            {emb.served_index ? (
              <div className="mt-3">
                <p className="mb-1 text-[12px] text-[var(--color-ink-2)]">
                  전곡 서빙 인덱스 {emb.served_index.plan_index} ({emb.served_index.rows.toLocaleString()}곡, {emb.served_index.index_mb} MB) · 정확 검색 대비
                </p>
                <Table
                  head={["ivfflat.probes", "recall", "p50 ms", "p95 ms"]}
                  rows={Object.entries(emb.served_index.probes as Record<string, Row>).map(([k, v]) => [k, v.recall_at_k, v.p50_ms, v.p95_ms])}
                />
              </div>
            ) : null}
          </>
        ) : <Empty />}
      </Block>
      <Block title="Phase 4 · LLM 태그 피처" note={tagVal ? `${tagVal.model} · 태그를 가린 곡 ${tagVal.songs}개 · 태그 어휘 안에서만 선택 · 곡 10개/프롬프트` : ""}>
        {tagVal ? (
          <>
            <Table
              head={["신뢰도 ≥", "precision", "recall", "곡당 태그"]}
              rows={Object.entries(tagVal.thresholds as Record<string, Row>).map(([k, v]) => [k, v.precision, v.recall, v.tags_per_song])}
            />
            <p className="mt-2 text-[12px] text-[var(--color-ink-2)]">
              인기 태그 쏠림: 예측 상위 5개 비중 {pct(tagVal.skew?.pred_top5_share)} vs 실제 {pct(tagVal.skew?.truth_top5_share)}
              {tagApply ? ` · 롱테일 적용: 후보 ${tagApply.candidates}곡 중 ${tagApply.stored}곡 저장 (신뢰도 ≥ ${tagApply.threshold}, LLM 호출 ${tagApply.llm_calls}회)` : ""}
            </p>
          </>
        ) : <Empty />}
      </Block>
      <Block title="Phase 5 · 홈 피드 이어 듣기" note={feed ? `플레이리스트 ${feed.playlists}개, 앞 80% → 뒤 20% 맞히기, k=${feed.k}` : ""}>
        {feed ? (
          <Table
            head={["소스 구성", "Recall@k", "hit@k", "헤드곡 비중", "p95 ms (end-to-end)"]}
            rows={Object.entries(feed.results as Record<string, Row>).map(([k, v]) => [k, v.recall_at_k, v.hit_at_k, pct(v.head_share), v.p95_e2e_ms])}
          />
        ) : <Empty />}
      </Block>
    </div>
  );
}

function Block({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-baseline gap-2">
        <h3 className="text-[14px] font-extrabold">{title}</h3>
        {note ? <span className="text-[12px] text-[var(--color-ink-3)]">{note}</span> : null}
      </div>
      {children}
    </div>
  );
}

function Empty() {
  return <Sticky color="gray" className="max-w-[360px] text-[12.5px]">아직 실행 기록이 없어요</Sticky>;
}
