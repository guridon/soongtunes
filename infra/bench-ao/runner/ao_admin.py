"""Sign in as the admin, mint an API token for the eval (saved to /work/../ao_token, never
printed), and read or set instance settings.

usage: python ao_admin.py token | settings | enable-compilation
"""
import http.cookiejar, json, os, sys, urllib.request

BASE = os.environ.get("AO_URL", "http://auto-ontology-frontend:3000")
ORIGIN = os.environ.get("APP_URL", "http://localhost:13000")
jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Origin": ORIGIN})
    with opener.open(req, timeout=60) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


call("POST", "/api/auth/sign-in/email", {"email": os.environ["AUTO_ONTOLOGY_ADMIN_EMAIL"],
                                          "password": os.environ["AUTO_ONTOLOGY_ADMIN_PASSWORD"]})
cmd = sys.argv[1]
if cmd == "token":
    key = call("POST", "/api/api-tokens", {"name": "stunes-eval", "expires_in_days": 30})["token"]
    path = "/out/ao_token"
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(key)
    print("token saved to", path, "prefix", key[:14])
elif cmd == "settings":
    for k in ("semantic-compilation", "distinct-value-probing", "pii-detection",
              "sql-query-timeout", "visualization"):
        try:
            print(k, call("GET", f"/api/configurations/{k}"))
        except Exception as e:  # noqa: BLE001
            print(k, "ERR", e)
elif cmd == "enable-compilation":
    print(call("PUT", "/api/configurations/semantic-compilation", {"enabled": True}))
elif cmd == "disable-visualization":
    print(call("PUT", "/api/configurations/visualization", {"enabled": False}))
