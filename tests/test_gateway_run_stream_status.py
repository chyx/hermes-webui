from api import routes


def test_gateway_run_stream_status_reports_active_external_run(monkeypatch):
    run_id = "run_issue133"
    monkeypatch.setattr(routes, "find_run_summary", lambda _sid: None)
    monkeypatch.setattr(
        routes,
        "_resolve_gateway_run_status",
        lambda sid: {
            "run_id": sid,
            "status": "running",
            "session_id": "20260628_211341_b3457286",
            "source": "webhook",
            "last_event": "tool.started",
            "created_at": 1.0,
            "updated_at": 2.0,
        },
    )

    payload = routes._chat_stream_status_payload(run_id)

    assert payload["active"] is True
    assert payload["gateway_run"] is True
    assert payload["status"] == "running"
    assert payload["session_id"] == "20260628_211341_b3457286"
    assert payload["last_event"] == "tool.started"


def test_gateway_run_stream_status_reports_completed_external_run_inactive(monkeypatch):
    run_id = "run_done"
    monkeypatch.setattr(routes, "find_run_summary", lambda _sid: None)
    monkeypatch.setattr(
        routes,
        "_resolve_gateway_run_status",
        lambda sid: {"run_id": sid, "status": "completed", "source": "webhook"},
    )

    payload = routes._chat_stream_status_payload(run_id)

    assert payload["active"] is False
    assert payload["gateway_run"] is True
    assert payload["status"] == "completed"
