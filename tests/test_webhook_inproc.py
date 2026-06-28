"""In-process integration test for the webhook handler.

Bypasses the network layer (TCP socket, Nagle, keep-alive, the
pytest-side socket.create_connection patch, etc.) entirely. Builds a
fake request object that mirrors what ``BaseHTTPRequestHandler`` would
construct, and feeds it directly to ``handle_post`` — the same path the
real server uses for POST /api/webhook/{name}.

This trades 100% network-realism for 100% handler-realism. Every line of
my code that runs in production also runs in this test. The conftest
isolation rules (no real ~/.hermes, no real API calls) are preserved by
writing the subscription file to a tmp dir and stubbing the profile /
session / turn-start hooks.
"""

import hashlib
import hmac
import io
import json
import os
import time
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

import api.routes as routes
from tests.conftest import TEST_STATE_DIR


_TEST_SECRET = "in-proc-test-shared-secret-do-not-use-in-prod"
_TEST_NAME = "inproc-issues"


def _write_subscription(tmp_home, *, name=_TEST_NAME, secret=_TEST_SECRET,
                        profile=None, actions=("opened",)):
    sub_path = tmp_home / "webhook_subscriptions.json"
    sub = {
        name: {
            "secret": secret,
            "prompt": "Issue #{issue.number}: {issue.title}\n\n{__raw__}",
            "chat_topic": "issue #{issue.number}: {issue.title}",
            "events": ["issues"],
            "actions": list(actions),
            "workspace": str(tmp_home),
        }
    }
    if profile is not None:
        sub[name]["profile"] = profile
    sub_path.write_text(json.dumps(sub), encoding="utf-8")
    return sub_path


class _FakeHandler:
    """Mimics what BaseHTTPRequestHandler builds per request.

    Uses ``email.message.Message`` (the real type behind
    BaseHTTPRequestHandler.headers) so case-insensitive header lookups
    work exactly like production.
    """

    def __init__(self, body: bytes = b"", headers: dict | None = None):
        from email.message import Message
        self.status = None
        self.response_headers = []
        self.wfile = io.BytesIO()
        self._body = body
        self.rfile = io.BytesIO(body)
        merged = dict(headers or {})
        merged.setdefault("Content-Length", str(len(body)))
        self.headers = Message()
        for k, v in merged.items():
            self.headers[k] = v

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.response_headers.append((key, value))

    def end_headers(self):
        self.response_headers.append(("__end__", ""))

    def __getattr__(self, name):
        # Some dispatch paths probe for attributes the real handler has
        # but our fake doesn't need (e.g. connection, close_connection).
        # Return a benign default so those probes don't AttributeError.
        return lambda *a, **kw: None


def _post(handler: _FakeHandler, path: str = "/api/webhook/inproc-issues"):
    """Drive handle_post with a fully-built handler. No socket involved."""
    parsed = urlparse(path)
    routes.handle_post(handler, parsed)
    raw = handler.wfile.getvalue()
    if not raw:
        return handler.status, {}
    try:
        return handler.status, json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return handler.status, {"_raw": raw[:200].decode("utf-8", errors="replace")}


@pytest.fixture
def inproc_env(tmp_path, monkeypatch):
    """Set up subscription file + stub profile/session/turn hooks for in-proc tests."""
    sub_path = _write_subscription(tmp_path)
    monkeypatch.setattr(routes, "_webhook_subscriptions_path", lambda: sub_path)

    # Profile stubs
    import api.profiles as _profiles
    monkeypatch.setattr(
        _profiles, "get_hermes_home_for_profile", lambda name: tmp_path / "profiles" / name
    )

    @contextmanager
    def _noop_profile_ctx(_home):
        yield

    monkeypatch.setattr(_profiles, "cron_profile_context_for_home", _noop_profile_ctx)

    # session / turn stubs
    monkeypatch.setattr(
        routes, "new_session", lambda **kw: SimpleNamespace(
            session_id=f"sess-{kw.get('profile', 'default')}",
            title="t", profile=kw.get("profile"),
            workspace=kw.get("workspace"), save=lambda: None,
        )
    )
    monkeypatch.setattr(
        routes, "start_session_turn",
        lambda sid, msg, *, source: {"_status": 200, "stream_id": f"stream-{sid}", "session_id": sid},
    )
    routes._WEBHOOK_DELIVERY_CACHE.clear()
    return tmp_path


def _signed_body(payload, *, secret=_TEST_SECRET, action="opened"):
    body = json.dumps({"action": action, **payload}).encode("utf-8")
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return body, sig


def test_inproc_happy_path_returns_202(inproc_env):
    payload = {
        "issue": {"number": 1, "title": "hello"},
        "repository": {"full_name": "chyx/test"},
    }
    body, sig = _signed_body(payload)
    handler = _FakeHandler(body, {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": sig,
        "X-GitHub-Delivery": "inproc-1",
    })
    status, resp = _post(handler)
    assert status == 202, f"expected 202, got {status}: {resp}"
    assert resp["status"] == "accepted"
    assert resp["session_id"] == "sess-dockerdev"  # forced profile
    assert resp["redirect_url"] == "/session/sess-dockerdev"


def test_inproc_rejects_bad_signature(inproc_env):
    body, _ = _signed_body({"issue": {"number": 1, "title": "x"},
                            "repository": {"full_name": "a/b"}})
    handler = _FakeHandler(body, {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": "sha256=" + "0" * 64,
    })
    status, resp = _post(handler)
    assert status == 401
    assert "signature" in (resp.get("error") or "").lower()


def test_inproc_dedupes(inproc_env):
    payload = {"issue": {"number": 1, "title": "x"},
               "repository": {"full_name": "a/b"}}
    body, sig = _signed_body(payload)
    h1 = _FakeHandler(body, {"X-GitHub-Event": "issues", "X-Hub-Signature-256": sig,
                              "X-GitHub-Delivery": "inproc-dup"})
    s1, _ = _post(h1)
    h2 = _FakeHandler(body, {"X-GitHub-Event": "issues", "X-Hub-Signature-256": sig,
                              "X-GitHub-Delivery": "inproc-dup"})
    s2, r2 = _post(h2)
    assert s1 == 202
    assert s2 == 200
    assert r2.get("status") == "duplicate"


def test_inproc_filters_unconfigured_event(inproc_env):
    body, sig = _signed_body({"issue": {"number": 1, "title": "x"},
                              "repository": {"full_name": "a/b"}})
    handler = _FakeHandler(body, {
        "X-GitHub-Event": "push",  # not in events=[issues]
        "X-Hub-Signature-256": sig,
    })
    status, resp = _post(handler)
    assert status == 200
    assert resp.get("status") == "ignored"


def test_inproc_unknown_route_returns_404(inproc_env, monkeypatch):
    # Remove the subscription file
    (inproc_env / "webhook_subscriptions.json").unlink()
    body, _ = _signed_body({"x": 1})
    handler = _FakeHandler(body, {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": "sha256=0",
    })
    status, _ = _post(handler, "/api/webhook/nonexistent")
    assert status == 404


def test_inproc_forces_dockerdev_default(inproc_env):
    body, sig = _signed_body({"issue": {"number": 1, "title": "x"},
                              "repository": {"full_name": "a/b"}})
    handler = _FakeHandler(body, {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": sig,
        "X-GitHub-Delivery": "inproc-dockerdev",
    })
    status, resp = _post(handler)
    assert status == 202
    # Forced profile: handler defaults to dockerdev
    assert resp["session_id"] == "sess-dockerdev"


def test_inproc_respects_explicit_profile_override(inproc_env):
    sub_path = routes._webhook_subscriptions_path()
    data = json.loads(sub_path.read_text())
    data[_TEST_NAME]["profile"] = "staging"
    sub_path.write_text(json.dumps(data))
    routes._WEBHOOK_DELIVERY_CACHE.clear()

    body, sig = _signed_body({"issue": {"number": 1, "title": "x"},
                              "repository": {"full_name": "a/b"}})
    handler = _FakeHandler(body, {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": sig,
        "X-GitHub-Delivery": "inproc-staging",
    })
    status, resp = _post(handler)
    assert status == 202
    assert resp["session_id"] == "sess-staging"
