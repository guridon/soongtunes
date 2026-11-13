import type { ChatResponse, EvalRun, Health, Neighborhood, Song, Strategy } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${detail.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  chat: (body: { session_id: string; message: string; strategy?: string; limit?: number; explain?: boolean }) =>
    request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(body) }),
  reset: (sessionId: string, strategy?: string) =>
    request<{ ok: boolean }>(
      `/sessions/${encodeURIComponent(sessionId)}${strategy ? `?strategy=${strategy}` : ""}`,
      { method: "DELETE" },
    ),
  search: (q: string, mode = "hybrid", limit = 20) =>
    request<Song[]>(`/search?${new URLSearchParams({ q, mode, limit: String(limit) })}`),
  songGraph: (id: string) => request<Neighborhood>(`/songs/${encodeURIComponent(id)}/graph`),
  strategies: () => request<Strategy[]>("/strategies"),
  sampleUsers: (n = 8) => request<string[]>(`/users/sample?n=${n}`),
  songs: (ids: string[]) => request<Song[]>(`/songs?ids=${ids.map(encodeURIComponent).join(",")}`),
  feed: (userId: string, limit = 20) => request<ChatResponse>(`/feed/${encodeURIComponent(userId)}?limit=${limit}`),
  features: (entityIds: string[], groups: string[]) =>
    request<Record<string, Record<string, unknown>>>("/features", {
      method: "POST",
      body: JSON.stringify({ entity_ids: entityIds, groups, profile: "C5" }),
    }),
  health: () => request<Health>("/health"),
  stats: () => request<{ graph: Record<string, number>; genres: string[] }>("/stats"),
  evalLatest: () => request<{ runs: EvalRun[] }>("/eval/latest"),
   
  results: () => request<Record<string, Record<string, any>[]>>("/results"),
};

export function newSessionId(): string {
  return `s-${crypto.randomUUID().slice(0, 8)}`;
}
