"""Read-only dummy protocol checks. Internal nginx boundary, not an SSO UI gate.

Run via existing xops-service Python; never emits cookies/credentials or writes data.
SSO identity provenance/session continuity must also be checked through public Caddy.
"""
import argparse
import hashlib
import json
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://frontend")
parser.add_argument("--state", choices=("unconfigured", "configured", "unsupported"), required=True)
options = parser.parse_args()
step = 0


def get(path, headers=None):
    request = urllib.request.Request(options.base + path, headers=headers or {})
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    body = response.read()
    return response.status, response.headers, body.decode("utf-8"), hashlib.sha256(body).hexdigest()


def check(name, path, status, verify, headers=None):
    global step
    actual, received, text, digest = get(path, headers)
    assert actual == status, f"{name}: HTTP {actual}, expected {status}"
    verify(received, text)
    step += 1
    safe_headers = {key: received.get(key) for key in
                    ("X-Data-Mode", "X-Integration-State", "Content-Type", "Cache-Control", "Vary")}
    print(json.dumps({"step": step, "name": name, "boundary": "internal-nginx", "path": path,
                      "http_status": actual, "headers": safe_headers, "body_sha256": digest,
                      "result": "PASS"}, ensure_ascii=False))


def ready(headers, text):
    assert "준비 중" in text and "외부 모듈 연결 준비" in text
    assert '<div id="root">' not in text
    assert headers.get("X-Integration-State") in ("unconfigured", "unsupported_mode")
    assert headers.get("Cache-Control") == "no-store"


for module in ("dashboard", "reporting"):
    fe = f"/rnd/population/{module}/"
    api = f"/rnd/population/{module}-api/"
    if options.state == "unconfigured":
        for route in (fe, api):
            for mode in ("mock", "real"):
                check(f"{module}-unset-{mode}", route + "?mode=" + mode, 200, ready)
        continue
    if options.state == "unsupported":
        for route in (fe, api):
            check(f"{module}-unsupported-mock", route + "?mode=mock", 200, ready)
        continue
    for mode in ("mock", "real"):
        def html(headers, text):
            assert f"{module} 더미 모듈" in text
            assert f'<base href="{fe}">' in text and f'data-mode="{mode}"' in text
            assert headers.get("X-Data-Mode") == mode
            assert headers.get("X-Integration-State") == "configured"
        check(f"{module}-fe-{mode}", fe + "?mode=" + mode, 200, html)
        check(f"{module}-deep-{mode}", fe + "deep/refresh-check/?mode=" + mode, 200, html)
        def asset(headers, text):
            assert "JS 자산 로드 완료" in text
            assert headers.get("X-Data-Mode") == mode
        check(f"{module}-asset-{mode}", fe + "assets/probe.js", 200, asset,
              {"Cookie": f"population_{module}_mode={mode}"})
        path = api + f"probe?mode={mode}&label=alpha%20beta&flag=1"
        def echo(headers, text):
            data = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
            assert data["module"] == module and data["x_data_mode"] == mode
            assert data["request_uri"] == f"/{module}-backend/probe?mode={mode}&label=alpha%20beta&flag=1"
            assert data["x_forwarded_prefix"] == api and data["remote_user"] == "integration-probe"
            assert data["cookie_received"] == "no" and data["authorization_received"] == "no"
            assert all(headers.get(field) is None for field in
                       ("Set-Cookie", "Clear-Site-Data", "X-Accel-Redirect", "Refresh"))
            assert headers.get("Cache-Control") == "private, no-store"
        check(f"{module}-api-{mode}", path, 200, echo,
              {"X-Data-Mode": mode, "Remote-User": "integration-probe",
               "Cookie": "integration-probe=placeholder", "Authorization": "Bearer integration-probe-placeholder"})
    def encoded(headers, text):
        assert f"request_uri=/{module}-backend/encoded%20name%3Fsegment?mode=mock&label=a%26b" in text
    check(f"{module}-raw-uri", api + "encoded%20name%3Fsegment?mode=mock&label=a%26b", 200, encoded)
    def cookie_html(headers, text):
        assert 'data-mode="mock"' in text and headers.get("X-Data-Mode") == "mock"
    check(f"{module}-fe-cookie-mode", fe, 200, cookie_html,
          {"Cookie": f"population_{module}_mode=mock"})
    for label, query, header in (
        ("duplicate", "mode=mock&mode=real", "mock"),
        ("conflict", "mode=mock", "real"),
        ("uppercase-value", "mode=MOCK", ""),
        ("uppercase-header", "", "MOCK"),
        ("uppercase-key", "Mode=mock", "mock"),
        ("encoded-key", "%6dode=real", "mock"),
        ("unknown", "mode=other", ""),
        ("missing", "", ""),
    ):
        check(f"{module}-reject-{label}", api + "probe" + ("?" + query if query else ""),
              400, lambda headers, text: None, {"X-Data-Mode": header} if header else {})

print(json.dumps({"summary": True, "state": options.state, "checks": step, "result": "PASS"}))
