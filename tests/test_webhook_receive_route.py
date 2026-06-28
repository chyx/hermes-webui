"""Unit tests for POST /api/webhook/{name} — the WebUI-side webhook receiver
that turns a GitHub/GitLab/Svix event into a real WebUI session via
``new_session`` + ``start_session_turn`` (no external-run special case).
"""

import hashlib
import hmac
import io
import json
from email.message import Message
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

import api.routes as routes


def _make_case_insensitive_headers(values: dict) -> Message:
    """Build a header container that mirrors ``BaseHTTPRequestHandler.headers``:
    case-insensitive ``.get(name)`` lookups. Uses ``email.message.Message``,
    the real underlying type of HTTPMessage, so the handler's case-sensitive
    dict assumptions fail loud (matching production behavior).
    """
    msg = Message()
    for k, v in (values or {}).items():
        msg[k] = v
    return msg


class _FakeHandler:
    def __init__(self, body: bytes = b"", headers: dict | None = None):
        self.status = None
        self.response_headers = []
        self.wfile = io.BytesIO()
        self._body = body
        self.rfile = io.BytesIO(body)
        # Auto-populate Content-Length so the dispatcher's content-length
        # helpers resolve correctly. Real BaseHTTPRequestHandler reads this
        # header from the wire; the fake has to mirror that explicitly.
        merged = dict(headers or {})
        merged.setdefault("Content-Length", str(len(body)))
        # Case-insensitive, like BaseHTTPRequestHandler.headers (HTTPMessage)
        self.headers = _make_case_insensitive_headers(merged)

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.response_headers.append((key, value))

    def end_headers(self):
        self.response_headers.append(("__end__", ""))


def _json_body(handler: _FakeHandler) -> dict:
    raw = handler.wfile.getvalue().decode("utf-8")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


_SECRET = "test-shared-secret-do-not-use-in-prod"
_PAYLOAD = {
    "action": "opened",
    "issue": {"number": 42, "title": "hello from a webhook"},
    "repository": {"full_name": "chyx/test"},
}


def _make_signature(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _post_webhook(handler: _FakeHandler, name: str = "github-issues"):
    parsed = urlsplit(f"/api/webhook/{name}")
    return routes._handle_webhook_receive(handler, parsed)


@pytest.fixture
def _stub_subscription(tmp_path, monkeypatch):
    """Point the loader at a tmp webhook_subscriptions.json with one route.

    Profile is intentionally NOT set in the route config — the handler
    must default to ``dockerdev`` and pin HERMES_HOME to that profile's
    home (test stubs out both ``get_hermes_home_for_profile`` and
    ``cron_profile_context_for_home`` so the test runs without a real
    multi-profile install).
    """
    sub_file = tmp_path / "webhook_subscriptions.json"
    sub_file.write_text(
        json.dumps(
            {
                "github-issues": {
                    "secret": _SECRET,
                    "prompt": "Issue #{issue.number}: {issue.title}\n\n{__raw__}",
                    "chat_topic": "issue #{issue.number}: {issue.title}",
                    "events": ["issues"],
                    "actions": ["opened"],
                    "workspace": str(tmp_path),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(routes, "_webhook_subscriptions_path", lambda: sub_file)
    # Profile plumbing stubs: handler resolves the target profile's home and
    # enters a context that pins HERMES_HOME. Both are imported lazily inside
    # the handler, so we patch the source module, not ``routes``.
    import api.profiles as _profiles

    monkeypatch.setattr(
        _profiles, "get_hermes_home_for_profile", lambda name: tmp_path / "profiles" / name
    )
    from contextlib import contextmanager

    @contextmanager
    def _noop_profile_ctx(_home):
        yield

    monkeypatch.setattr(_profiles, "cron_profile_context_for_home", _noop_profile_ctx)
    # Clear the in-memory delivery cache between tests
    routes._WEBHOOK_DELIVERY_CACHE.clear()
    return sub_file


def test_webhook_happy_path_creates_session_and_starts_turn(_stub_subscription, monkeypatch):
    body = json.dumps(_PAYLOAD).encode("utf-8")
    sig = _make_signature(_SECRET, body)
    handler = _FakeHandler(
        body=body,
        headers={
            "X-GitHub-Event": "issues",
            "X-GitHub-Delivery": "delivery-1",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )

    captured = {}

    def fake_new_session(*, workspace, model, model_provider, profile, project_id, worktree_info):
        s = SimpleNamespace(
            session_id="sess-abc123",
            title="Untitled",
            profile=profile,
            workspace=workspace,
            model=model,
            model_provider=model_provider,
        )
        s.save_called = 0

        def _save():
            s.save_called += 1

        s.save = _save
        captured["new_session_kwargs"] = {
            "workspace": workspace,
            "model": model,
            "model_provider": model_provider,
            "profile": profile,
        }
        return s

    def fake_start_session_turn(session_id, message, *, source):
        captured["start_kwargs"] = {
            "session_id": session_id,
            "source": source,
            "message_excerpt": message[:200],
        }
        return {
            "_status": 200,
            "stream_id": "stream-xyz",
            "session_id": session_id,
        }

    monkeypatch.setattr(routes, "new_session", fake_new_session)
    monkeypatch.setattr(routes, "start_session_turn", fake_start_session_turn)

    _post_webhook(handler)

    assert handler.status == 202
    body_json = _json_body(handler)
    assert body_json["status"] == "accepted"
    assert body_json["session_id"] == "sess-abc123"
    assert body_json["stream_id"] == "stream-xyz"
    assert body_json["event"] == "issues"
    assert body_json["delivery_id"] == "delivery-1"
    assert body_json["redirect_url"] == "/session/sess-abc123"

    # The session was created with the route's workspace + the forced
    # profile (defaults to dockerdev when route config omits it).
    assert captured["new_session_kwargs"]["workspace"]
    assert captured["new_session_kwargs"]["profile"] == "dockerdev"

    # start_session_turn received the rendered prompt + webhook source label
    sk = captured["start_kwargs"]
    assert sk["session_id"] == "sess-abc123"
    assert sk["source"] == "webhook"
    assert "Issue #42" in sk["message_excerpt"]
    assert "hello from a webhook" in sk["message_excerpt"]


def test_webhook_rejects_invalid_signature(_stub_subscription, monkeypatch):
    body = json.dumps(_PAYLOAD).encode("utf-8")
    handler = _FakeHandler(
        body=body,
        headers={
            "X-GitHub-Event": "issues",
            "X-GitHub-Delivery": "delivery-bad-sig",
            "X-Hub-Signature-256": "sha256=" + "0" * 64,  # wrong
        },
    )
    started = {"count": 0}
    monkeypatch.setattr(
        routes,
        "start_session_turn",
        lambda *a, **kw: started.__setitem__("count", started["count"] + 1) or {"_status": 200},
    )
    _post_webhook(handler)
    assert handler.status == 401
    assert started["count"] == 0  # never reached start_session_turn


def test_webhook_rejects_missing_signature(_stub_subscription, monkeypatch):
    body = json.dumps(_PAYLOAD).encode("utf-8")
    handler = _FakeHandler(
        body=body,
        headers={"X-GitHub-Event": "issues", "X-GitHub-Delivery": "delivery-no-sig"},
    )
    _post_webhook(handler)
    assert handler.status == 401


def test_webhook_unknown_route_returns_404(tmp_path, monkeypatch):
    sub_file = tmp_path / "webhook_subscriptions.json"
    sub_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(routes, "_webhook_subscriptions_path", lambda: sub_file)
    routes._WEBHOOK_DELIVERY_CACHE.clear()

    body = json.dumps(_PAYLOAD).encode("utf-8")
    sig = _make_signature(_SECRET, body)
    handler = _FakeHandler(
        body=body,
        headers={
            "X-GitHub-Event": "issues",
            "X-Hub-Signature-256": sig,
        },
    )
    _post_webhook(handler, name="nope")
    assert handler.status == 404


def test_webhook_disabled_route_returns_403(_stub_subscription, monkeypatch):
    sub_file = routes._webhook_subscriptions_path()
    data = json.loads(sub_file.read_text(encoding="utf-8"))
    data["github-issues"]["enabled"] = False
    sub_file.write_text(json.dumps(data), encoding="utf-8")
    routes._WEBHOOK_DELIVERY_CACHE.clear()

    body = json.dumps(_PAYLOAD).encode("utf-8")
    sig = _make_signature(_SECRET, body)
    handler = _FakeHandler(
        body=body,
        headers={"X-GitHub-Event": "issues", "X-Hub-Signature-256": sig},
    )
    _post_webhook(handler)
    assert handler.status == 403


def test_webhook_ignores_filtered_event(_stub_subscription, monkeypatch):
    body = json.dumps(_PAYLOAD).encode("utf-8")
    sig = _make_signature(_SECRET, body)
    handler = _FakeHandler(
        body=body,
        headers={
            "X-GitHub-Event": "push",  # not in events=[issues]
            "X-Hub-Signature-256": sig,
        },
    )
    started = {"count": 0}
    monkeypatch.setattr(
        routes,
        "start_session_turn",
        lambda *a, **kw: started.__setitem__("count", started["count"] + 1) or {"_status": 200},
    )
    _post_webhook(handler)
    assert handler.status == 200
    assert _json_body(handler).get("status") == "ignored"
    assert started["count"] == 0


def test_webhook_ignores_filtered_action(_stub_subscription, monkeypatch):
    body = json.dumps(_PAYLOAD).encode("utf-8")
    sig = _make_signature(_SECRET, body)
    handler = _FakeHandler(
        body=body,
        headers={
            "X-GitHub-Event": "issues",
            "X-Hub-Signature-256": sig,
        },
    )
    started = {"count": 0}
    monkeypatch.setattr(
        routes,
        "start_session_turn",
        lambda *a, **kw: started.__setitem__("count", started["count"] + 1) or {"_status": 200},
    )
    # Payload has action="opened" but the route only allows ["reopened"] → must
    # be ignored with 200, not 202. Fix from earlier misnamed test that
    # accidentally asserted the happy path.
    sub_file = routes._webhook_subscriptions_path()
    data = json.loads(sub_file.read_text(encoding="utf-8"))
    data["github-issues"]["actions"] = ["reopened"]
    sub_file.write_text(json.dumps(data), encoding="utf-8")
    routes._WEBHOOK_DELIVERY_CACHE.clear()

    _post_webhook(handler)
    assert handler.status == 200
    body_json = _json_body(handler)
    assert body_json.get("status") == "ignored"
    assert body_json.get("action") == "opened"
    assert started["count"] == 0


def test_webhook_dedupes_duplicate_delivery_id(_stub_subscription, monkeypatch):
    body = json.dumps(_PAYLOAD).encode("utf-8")
    sig = _make_signature(_SECRET, body)
    started = {"count": 0}

    def fake_start(*a, **kw):
        started["count"] += 1
        return {"_status": 200, "stream_id": "stream-dup", "session_id": "sess-dup"}

    monkeypatch.setattr(routes, "new_session", lambda **kw: SimpleNamespace(
        session_id="sess-dup", title="t", profile=kw.get("profile"),
        workspace=kw.get("workspace"), save=lambda: None,
    ))
    monkeypatch.setattr(routes, "start_session_turn", fake_start)

    for i in range(3):
        handler = _FakeHandler(
            body=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": sig,
                "X-GitHub-Delivery": "delivery-dup",
            },
        )
        _post_webhook(handler)
    assert started["count"] == 1  # only the first call started a turn


def test_webhook_renders_prompt_template(_stub_subscription, monkeypatch):
    body = json.dumps(_PAYLOAD).encode("utf-8")
    sig = _make_signature(_SECRET, body)
    captured = {}
    monkeypatch.setattr(routes, "new_session", lambda **kw: SimpleNamespace(
        session_id="sess-render", title="t", profile=kw.get("profile"),
        workspace=kw.get("workspace"), save=lambda: None,
    ))

    def fake_start(session_id, message, *, source):
        captured["message"] = message
        captured["source"] = source
        return {"_status": 200, "stream_id": "s1", "session_id": session_id}

    monkeypatch.setattr(routes, "start_session_turn", fake_start)
    handler = _FakeHandler(
        body=body,
        headers={
            "X-GitHub-Event": "issues",
            "X-Hub-Signature-256": sig,
            "X-GitHub-Delivery": "delivery-render",
        },
    )
    _post_webhook(handler)
    assert "Issue #42" in captured["message"]
    assert "hello from a webhook" in captured["message"]
    assert "chyx/test" in captured["message"]


def test_webhook_returns_400_on_missing_route_name(_stub_subscription, monkeypatch):
    parsed = urlsplit("/api/webhook/")
    handler = _FakeHandler(body=b"", headers={})
    result = routes._handle_webhook_receive(handler, parsed)
    assert handler.status == 400


def test_webhook_csrf_exempt_path_includes_webhook_namespace():
    """`_csrf_exempt_path` must cover /api/webhook/* so HMAC is the real gate."""
    assert routes._csrf_exempt_path("/api/webhook/github-issues")
    assert routes._csrf_exempt_path("/api/webhook/anything")


def test_webhook_forces_dockerdev_when_route_omits_profile(
    _stub_subscription, monkeypatch
):
    """The user pinned webhook runs to dockerdev; verify the default kicks in.

    Route config has no ``profile`` field, but the handler must still
    resolve the run under ``dockerdev`` and create the session with that
    profile tag so the sidebar groups it correctly.
    """
    body = json.dumps(_PAYLOAD).encode("utf-8")
    sig = _make_signature(_SECRET, body)
    handler = _FakeHandler(
        body=body,
        headers={
            "X-GitHub-Event": "issues",
            "X-Hub-Signature-256": sig,
            "X-GitHub-Delivery": "delivery-default-dockerdev",
        },
    )
    captured = {}
    monkeypatch.setattr(
        routes,
        "new_session",
        lambda **kw: (captured.update(kw) or SimpleNamespace(
            session_id="sess-dockerdev",
            title="t",
            profile=kw["profile"],
            workspace=kw["workspace"],
            save=lambda: None,
        )),
    )
    monkeypatch.setattr(
        routes,
        "start_session_turn",
        lambda sid, msg, *, source: {"_status": 200, "stream_id": "s", "session_id": sid},
    )

    _post_webhook(handler)
    assert handler.status == 202
    assert captured["profile"] == "dockerdev"


def test_webhook_respects_explicit_profile_override(_stub_subscription, monkeypatch):
    """Operators can pin a non-dockerdev profile per route via the config.

    Documents that ``profile`` in webhook_subscriptions.json is the
    source of truth — handlers don't second-guess the operator.
    """
    sub_file = routes._webhook_subscriptions_path()
    data = json.loads(sub_file.read_text(encoding="utf-8"))
    data["github-issues"]["profile"] = "staging"
    sub_file.write_text(json.dumps(data), encoding="utf-8")
    routes._WEBHOOK_DELIVERY_CACHE.clear()

    body = json.dumps(_PAYLOAD).encode("utf-8")
    sig = _make_signature(_SECRET, body)
    handler = _FakeHandler(
        body=body,
        headers={
            "X-GitHub-Event": "issues",
            "X-Hub-Signature-256": sig,
            "X-GitHub-Delivery": "delivery-staging",
        },
    )
    captured = {}
    monkeypatch.setattr(
        routes,
        "new_session",
        lambda **kw: (captured.update(kw) or SimpleNamespace(
            session_id="sess-staging",
            title="t",
            profile=kw["profile"],
            workspace=kw["workspace"],
            save=lambda: None,
        )),
    )
    monkeypatch.setattr(
        routes,
        "start_session_turn",
        lambda sid, msg, *, source: {"_status": 200, "stream_id": "s", "session_id": sid},
    )

    _post_webhook(handler)
    assert handler.status == 202
    assert captured["profile"] == "staging"
