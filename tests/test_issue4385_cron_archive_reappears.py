"""Regression coverage for archived state.db-projected sessions reappearing."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch


def test_cron_rows_are_not_cli_even_with_stale_cli_flag():
    """A stale sidecar flag must not turn a cron row into an external CLI row."""
    from api.agent_sessions import is_cli_session_row

    row = {
        "session_id": "cron_job123_20260618",
        "title": "Cron Session",
        "source_tag": "cron",
        "raw_source": "cron",
        "session_source": "cron",
        "source_label": "Cron",
        "is_cli_session": True,
    }

    assert is_cli_session_row(row) is False


def test_materializing_cron_session_preserves_non_cli_identity(monkeypatch):
    """Cron materialization must not stamp the sidecar as CLI-imported."""
    import api.routes as routes

    sid = "cron_job123_20260618"
    cron_meta = {
        "session_id": sid,
        "title": "Cron Session",
        "model": "test-model",
        "source_tag": "cron",
        "raw_source": "cron",
        "session_source": "cron",
        "source_label": "Cron",
        "read_only": False,
        "profile": "default",
    }

    class FakeSession:
        def __init__(self):
            self.session_id = sid
            self.title = "Cron Session"
            self.profile = "default"
            self.model = "test-model"
            self.archived = False
            self.is_cli_session = False
            self.source_tag = None
            self.raw_source = None
            self.session_source = None
            self.source_label = None
            self.read_only = False

        def save(self, *args, **kwargs):
            pass

    def fake_import_cli_session(*args, **kwargs):
        return FakeSession()

    with (
        patch.object(routes, "get_session", side_effect=KeyError(sid)),
        patch.object(routes, "_lookup_cli_session_metadata", return_value=cron_meta),
        patch.object(
            routes,
            "get_cli_session_messages",
            return_value=[
                {"role": "user", "content": "run"},
                {"role": "assistant", "content": "done"},
            ],
        ),
        patch.object(routes, "import_cli_session", side_effect=fake_import_cli_session),
    ):
        session = routes._get_or_materialize_session(sid)

    assert session.session_source == "cron"
    assert session.source_tag == "cron"
    assert session.is_cli_session is False


def test_cron_state_projection_preserves_archived_sidecar(monkeypatch, tmp_path):
    """A hidden archived sidecar must still mark the state.db cron projection archived."""
    import api.models as models

    sid = "cron_job123_20260618"
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                model TEXT,
                message_count INTEGER,
                started_at REAL,
                source TEXT,
                session_source TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (
                id, title, model, message_count, started_at, source, session_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (sid, "Cron Session", "test-model", 1, 20, "cron", "cron"),
        )

    class ArchivedSidecar:
        title = "Cron Session"
        archived = True

    monkeypatch.setattr(
        models.Session,
        "load_metadata_only",
        staticmethod(lambda candidate: ArchivedSidecar() if candidate == sid else None),
    )
    monkeypatch.setattr(models, "ensure_cron_project", lambda: "cron-project")

    rows = models._load_cli_sessions_uncached(
        tmp_path,
        db_path,
        "default",
        source_filter="cron",
        include_claude_code=False,
    )

    assert len(rows) == 1
    assert rows[0]["session_id"] == sid
    assert rows[0]["archived"] is True


def test_webhook_state_projection_preserves_archived_sidecar(monkeypatch, tmp_path):
    """Archived webhook sidecars must not reappear as unarchived state.db rows."""
    import api.models as models

    sid = "webhook_archive_20260618"
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                model TEXT,
                message_count INTEGER,
                started_at REAL,
                source TEXT,
                user_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (
                id, title, model, message_count, started_at, source, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                None,
                "test-model",
                2,
                20,
                "webhook",
                "webhook:read-later",
            ),
        )
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, 'user', 'payload', 21)
            """,
            (sid,),
        )
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, 'assistant', 'done', 22)
            """,
            (sid,),
        )

    class ArchivedWebhookSidecar:
        title = "Webhook Session"
        archived = True

    monkeypatch.setattr(
        models.Session,
        "load_metadata_only",
        staticmethod(lambda candidate: ArchivedWebhookSidecar() if candidate == sid else None),
    )

    rows = models._load_cli_sessions_uncached(
        tmp_path,
        db_path,
        "default",
        include_claude_code=False,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == sid
    assert row["title"] == "Webhook Session"
    assert row["source_tag"] == "webhook"
    assert row["raw_source"] == "webhook"
    assert row["session_source"] == "other"
    assert row["source_label"] == "Webhook"
    assert row["is_cli_session"] is False
    assert row["archived"] is True


def test_webhook_state_db_title_wins_over_long_platform_display_name(monkeypatch, tmp_path):
    """Concise generated state.db titles should not be overwritten by chat_topic."""
    import api.models as models

    sid = "webhook_title_20260628"
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                model TEXT,
                message_count INTEGER,
                started_at REAL,
                source TEXT,
                user_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (
                id, title, model, message_count, started_at, source, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                "修复子模块数量显示为直接子节点数",
                "test-model",
                2,
                20,
                "webhook",
                "webhook:github-issues",
            ),
        )
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, 'user', 'payload', 21)
            """,
            (sid,),
        )

    class LongWebhookSidecar:
        title = "chyx/hermes-companion-tools #119: [MC-Command] 顶上的bar显示的应该是直接儿子的数量，不应该递归求和吧，不然ui上看着很奇怪"
        archived = False

    monkeypatch.setattr(
        models.Session,
        "load_metadata_only",
        staticmethod(lambda candidate: LongWebhookSidecar() if candidate == sid else None),
    )

    rows = models._load_cli_sessions_uncached(
        tmp_path,
        db_path,
        "default",
        include_claude_code=False,
    )

    assert len(rows) == 1
    assert rows[0]["session_id"] == sid
    assert rows[0]["title"] == "修复子模块数量显示为直接子节点数"
    assert rows[0]["archived"] is False


def test_session_detail_uses_state_title_unless_manually_renamed():
    """Direct /session/<sid> loads should not keep stale webhook chat_topic titles."""
    import api.routes as routes

    class AutoSidecar:
        manual_title = False

    class ManualSidecar:
        manual_title = True

    stale_payload = {"title": "repo #128: [MC-Command] very long webhook title"}
    state_meta = {"title": "修复文档计数显示矛盾"}

    auto = routes._apply_state_title_to_session_detail(
        stale_payload,
        state_meta,
        AutoSidecar(),
    )
    manual = routes._apply_state_title_to_session_detail(
        stale_payload,
        state_meta,
        ManualSidecar(),
    )

    assert auto["title"] == "修复文档计数显示矛盾"
    assert manual["title"] == stale_payload["title"]


def test_archived_webhook_projection_reaches_sidebar_payload(monkeypatch):
    """The sidebar payload must preserve archived state for webhook projections."""
    import api.routes as routes

    sid = "webhook_archive_20260618"
    raw_webhook_row = {
        "session_id": sid,
        "title": "Webhook Session",
        "profile": "default",
        "updated_at": 22,
        "last_message_at": 22,
        "message_count": 2,
        "user_message_count": 1,
        "archived": True,
        "source_tag": "webhook",
        "raw_source": "webhook",
        "session_source": "other",
        "source_label": "Webhook",
        "is_cli_session": False,
    }

    monkeypatch.setattr(routes, "all_sessions", lambda diag=None: [])
    monkeypatch.setattr(routes, "get_cli_sessions", lambda source_filter=None, all_profiles=False: [raw_webhook_row])
    monkeypatch.setattr(routes, "_reconcile_stale_stream_state_for_session_rows", lambda _sessions: False)

    payload = routes._build_session_list_cache_payload(
        active_profile="default",
        all_profiles=False,
        show_cli_sessions=True,
        show_previous_messaging_sessions=False,
        show_cron_sessions=False,
    )

    rows = payload["sessions"]
    matching = [row for row in rows if row["session_id"] == sid]
    assert len(matching) == 1
    assert matching[0]["archived"] is True
    assert matching[0]["source_tag"] == "webhook"
    assert matching[0]["is_cli_session"] is False


def test_archived_cron_sidecar_suppresses_raw_unarchived_cron_row(monkeypatch):
    """An archived cron sidecar should keep the raw state.db cron row hidden."""
    import api.routes as routes

    sid = "cron_job123_20260618"
    archived_sidecar = {
        "session_id": sid,
        "title": "Cron Session",
        "profile": "default",
        "updated_at": 20,
        "last_message_at": 20,
        "message_count": 1,
        "user_message_count": 1,
        "archived": True,
        "source_tag": "cron",
        "raw_source": "cron",
        "session_source": "cron",
        "source_label": "Cron",
        "is_cli_session": True,
    }
    raw_cron_row = {
        "session_id": sid,
        "title": "Cron Session",
        "profile": "default",
        "updated_at": 20,
        "last_message_at": 20,
        "message_count": 1,
        "user_message_count": 1,
        "archived": False,
        "project_id": "cron-project",
        "source_tag": "cron",
        "raw_source": "cron",
        "session_source": "cron",
        "source_label": "Cron",
        "is_cli_session": False,
    }

    monkeypatch.setattr(routes, "all_sessions", lambda diag=None: [archived_sidecar])
    monkeypatch.setattr(routes, "get_cli_sessions", lambda source_filter=None, all_profiles=False: [raw_cron_row])
    monkeypatch.setattr(routes, "_reconcile_stale_stream_state_for_session_rows", lambda _sessions: False)

    payload = routes._build_session_list_cache_payload(
        active_profile="default",
        all_profiles=False,
        show_cli_sessions=True,
        show_previous_messaging_sessions=False,
        show_cron_sessions=True,
    )

    rows = payload["sessions"]
    matching = [row for row in rows if row["session_id"] == sid]
    assert len(matching) == 1
    assert matching[0]["archived"] is True
    assert matching[0]["is_cli_session"] is False
