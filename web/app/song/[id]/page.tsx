"use client";

import { Background, BackgroundVariant, type Edge, type Node, ReactFlow, ReactFlowProvider } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { nodeTypes, SONG_H, SONG_W } from "@/components/nodes";
import { Avatars, Panel, TopBar } from "@/components/ui";
import { api } from "@/lib/api";
import type { Neighborhood } from "@/lib/types";

const RAD = Math.PI / 180;

function ring(n: number, from: number, to: number, r: number, cx = 0, cy = 0) {
  return Array.from({ length: n }, (_, i) => {
    const a = (n === 1 ? (from + to) / 2 : from + ((to - from) * i) / (n - 1)) * RAD;
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  });
}

function layout(g: Neighborhood) {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const s = g.song;
  const center = { x: -SONG_W / 2, y: -SONG_H / 2 };
  nodes.push({ id: "center", type: "song", position: center, data: { song: { ...s, artists: s.artists.map((a) => a.name) } } });

  const edge = (id: string, source: string, target: string, label?: string, dashed = false): Edge => ({
    id, source, target, label,
    labelStyle: { fontSize: 11, fontWeight: 600, fill: "#55534e" },
    style: dashed ? { strokeDasharray: "5 5" } : undefined,
  });

  s.artists.forEach((a, i) => {
    const id = `artist-${a.artist_id}`;
    nodes.push({ id, type: "entity", position: { x: -470, y: -40 + i * 60 }, data: { kind: "artist", label: a.name } });
    edges.push(edge(`e-${id}`, id, "center", "부름"));
  });
  if (s.album) {
    nodes.push({ id: "album", type: "entity", position: { x: -420, y: -230 }, data: { kind: "album", label: s.album.title } });
    edges.push(edge("e-album", "album", "center", "수록", true));
  }
  ring(s.genres.length, -120, -60, 300).forEach((p, i) => {
    const id = `genre-${i}`;
    nodes.push({ id, type: "entity", position: { x: p.x - 50, y: p.y - 40 }, data: { kind: "genre", label: s.genres[i] } });
    edges.push(edge(`e-${id}`, "center", id, undefined, true));
  });
  const tags = s.tags.slice(0, 8);
  ring(tags.length, -55, 35, 330).forEach((p, i) => {
    const id = `tag-${i}`;
    nodes.push({ id, type: "entity", position: { x: p.x + 40, y: p.y - 20 }, data: { kind: "tag", label: tags[i] } });
    edges.push(edge(`e-${id}`, "center", id, undefined, true));
  });

  // co-occurring songs fan out below the center on a shallow "smile" so stickies never overlap
  const n = g.cooccurring.length;
  g.cooccurring.forEach((song, i) => {
    const off = i - (n - 1) / 2;
    const id = `co-${song.song_id}`;
    nodes.push({
      id, type: "song", position: { x: off * (SONG_W + 28) - SONG_W / 2, y: 330 + off * off * 14 },
      data: { song: { ...song, sources: { cooccur: i + 1 } }, tilt: ((i * 37) % 5) - 2, note: `플리 ${song.shared_playlists}개 공유` },
    });
    edges.push(edge(`e-${id}`, "center", id, `함께 담김 ${song.shared_playlists}`));
  });

  const firstArtist = s.artists[0] ? `artist-${s.artists[0].artist_id}` : "center";
  g.same_artist.forEach((song, i) => {
    const id = `same-${song.song_id}`;
    nodes.push({
      id, type: "song",
      position: { x: -1180 + (i % 3) * (SONG_W + 24), y: -260 + Math.floor(i / 3) * (SONG_H + 30) },
      data: { song: { ...song, sources: {} }, color: "violet", tilt: ((i * 29) % 5) - 2, note: "같은 아티스트" },
    });
    edges.push(edge(`e-${id}`, firstArtist, id, undefined, true));
  });
  return { nodes, edges };
}

function SongGraph() {
  const { id } = useParams<{ id: string }>();
  // keyed by id: navigating to another song remounts with fresh state
  return <SongGraphView key={id} id={id} />;
}

function SongGraphView({ id }: { id: string }) {
  const [data, setData] = useState<Neighborhood | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.songGraph(id).then(setData).catch((e) => setError(String(e)));
  }, [id]);

  const { nodes, edges } = useMemo(() => (data ? layout(data) : { nodes: [], edges: [] }), [data]);

  return (
    <div className="relative h-dvh w-full">
      <TopBar title={data ? data.song.title : "곡 그래프"} right={<Avatars />} />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.15}
        proOptions={{ hideAttribution: true }}
        nodesConnectable={false}
        panOnScroll
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1.6} color="#cfcdc6" bgColor="#f3f2ef" />
      </ReactFlow>
      <div className="absolute bottom-4 left-4 z-10 max-w-[calc(100%-2rem)]">
        <Panel className="px-4 py-3 text-[13px]">
          {error ? (
            <span className="text-[var(--color-pink-ink)]">불러오지 못했어요: {error}</span>
          ) : !data ? (
            <span className="text-[var(--color-ink-3)]">그래프를 불러오는 중…</span>
          ) : (
            <div className="space-y-0.5">
              <div className="font-bold">{data.song.title}</div>
              <div className="text-[var(--color-ink-2)]">
                플레이리스트 {data.song.popularity.toLocaleString()}개에 담김 · 태그 {data.song.tag_count}개 · 곡을 누르면 그 곡의 그래프로 이동해요
              </div>
              <Link href="/" className="text-[12px] font-semibold text-[var(--color-accent)]">← 보드로</Link>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

export default function Page() {
  return (
    <ReactFlowProvider>
      <SongGraph />
    </ReactFlowProvider>
  );
}
