export type SourceName = "lookup" | "graph" | "tags" | "cooccur" | "bm25" | "vector" | "popular" | "text2cypher";

export type Song = {
  song_id: string;
  title: string;
  artists: string[];
  artist_ids?: string[];
  album?: string | null;
  genres: string[];
  tags: string[];
  playlist_count: number;
  year?: number | null;
  issue_date?: string | null;
  score?: number | null;
  sources?: Partial<Record<SourceName, number>>;
  signals?: Record<string, number>;
};

export type Pref = { kind: "artist" | "genre" | "tag" | "song"; key: string; name: string };

export type Linked = { key: string; name: string; via: string; score: number };

export type Trace = {
  mode?: string;
  slots?: Record<string, unknown>;
  linked?: Record<string, Linked[] | string[]>;
  seeds?: string[];
  plans?: Record<string, unknown>;
  candidates?: Record<string, number>;
  dropped_by_verifier?: Record<string, string[]>;
  relaxed?: string[];
  timings_ms?: Record<string, number>;
  error?: string;
  cypher?: string;
};

export type ChatResponse = {
  session_id: string;
  strategy: string;
  reply: string;
  songs: Song[];
  state: { positive: Pref[]; negative: Pref[] };
  trace: Trace;
};

export type Strategy = { name: string; description: string };

export type Neighborhood = {
  song: Omit<Song, "artists"> & {
    popularity: number;
    artists: { artist_id: string; name: string }[];
    album: { album_id: string; title: string } | null;
    tag_count: number;
  };
  cooccurring: (Song & { shared_playlists: number })[];
  same_artist: Song[];
};

export type Health = {
  ok: boolean;
  deps: Record<string, { ok: boolean; ms?: number; error?: string }>;
  embedding_versions: Record<string, string | null>;
};

export type EvalMetrics = { n: number; hit: number; mrr: number; ndcg: number; precision: number; recall: number };

export type EvalRun = {
  run_id: string;
  note: string;
  conversations: number;
  concurrency?: number;
  k: number;
  commit: string;
  strategies: Record<
    string,
    {
      metrics: Record<string, EvalMetrics>;
      latency_ms: { p50: number | null; p95: number | null; mean: number | null };
      errors: number;
      degraded_turns?: number;
      turns: number;
    }
  >;
};
