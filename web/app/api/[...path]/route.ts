// Same-origin proxy to the FastAPI service. The backend URL is read at request time, so one
// image works on the host (localhost:8000) and in compose (http://api:8000).
const BACKEND = process.env.API_INTERNAL_URL ?? "http://localhost:8000";

async function forward(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const url = new URL(req.url);
  const target = `${BACKEND}/api/${path.map(encodeURIComponent).join("/")}${url.search}`;
  const init: RequestInit = {
    method: req.method,
    headers: { "content-type": req.headers.get("content-type") ?? "application/json" },
    body: req.method === "GET" || req.method === "HEAD" ? undefined : await req.text(),
    cache: "no-store",
    signal: AbortSignal.timeout(180_000),
  };
  try {
    const res = await fetch(target, init);
    return new Response(res.body, {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch (e) {
    return Response.json({ detail: `backend unreachable: ${String(e)}` }, { status: 502 });
  }
}

export const GET = forward;
export const POST = forward;
export const DELETE = forward;
export const dynamic = "force-dynamic";
