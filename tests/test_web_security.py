"""C2/C3: server-side live-trading gate, Host allowlist, Origin checks on writes and the WS feed."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from kalshi_poc.config import ConfigManager
from kalshi_poc.telemetry import TelemetryManager
from kalshi_poc.web_server import create_app

LOCAL = "http://127.0.0.1:8000"


def _cfg(tmp_path: Path, allow_live: bool = False) -> ConfigManager:
    return ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
        allow_live=allow_live,
    )


def test_live_mode_rejected_without_confirm_live(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="confirm-live"):
        cfg.update_and_persist({"execution_mode": "live"})
    client = TestClient(create_app(config_manager=cfg), base_url=LOCAL)
    resp = client.post("/api/config", json={"execution_mode": "live", "kalshi_environment": "production"})
    assert resp.status_code == 400
    assert "confirm-live" in resp.json()["detail"]
    assert cfg.settings.execution_mode.value == "paused"


def test_live_mode_allowed_with_confirm_live(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, allow_live=True)
    s = cfg.update_and_persist({"execution_mode": "live", "strategy_provider": "gemini"})
    assert s.execution_mode.value == "live"


def test_live_mode_blocked_with_stub_strategy(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, allow_live=True)
    with pytest.raises(ValueError, match="stub_momentum"):
        cfg.update_and_persist({"execution_mode": "live"})  # default provider is the stub
    assert cfg.settings.execution_mode.value == "paused"
    # Switching to the stub while live is blocked too
    cfg.update_and_persist({"execution_mode": "live", "strategy_provider": "gemini"})
    with pytest.raises(ValueError, match="stub_momentum"):
        cfg.update_and_persist({"strategy_provider": "stub_momentum"})
    assert cfg.settings.strategy_provider.value == "gemini"


def test_foreign_host_header_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(config_manager=_cfg(tmp_path)), base_url="http://rebind.attacker.example:8000")
    assert client.get("/api/config").status_code == 403
    local = TestClient(create_app(config_manager=_cfg(tmp_path)), base_url="http://localhost:8000")
    assert local.get("/api/config").status_code == 200


def test_cross_origin_writes_rejected_same_origin_and_no_origin_allowed(tmp_path: Path) -> None:
    client = TestClient(create_app(config_manager=_cfg(tmp_path)), base_url=LOCAL)
    evil = client.post("/api/engine/start", headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    null_origin = client.post("/api/engine/kill", headers={"Origin": "null"})
    assert null_origin.status_code == 403
    same = client.post("/api/engine/pause", headers={"Origin": "http://127.0.0.1:8000"})
    assert same.status_code == 200
    cli = client.post("/api/engine/pause")  # curl / scripts send no Origin
    assert cli.status_code == 200


def test_extra_allowed_hosts_from_settings(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.update_and_persist({"web_allowed_hosts": ["Trader.Internal"]})
    client = TestClient(create_app(config_manager=cfg), base_url="http://trader.internal:8000")
    assert client.get("/api/config").status_code == 200


async def test_ws_telemetry_rejects_cross_origin(tmp_path: Path) -> None:
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    try:
        client = TestClient(create_app(config_manager=_cfg(tmp_path), telemetry=telem), base_url=LOCAL)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("ws://127.0.0.1:8000/ws/telemetry", headers={"Origin": "https://evil.example"}) as ws:
                ws.receive_json()
        with client.websocket_connect("ws://127.0.0.1:8000/ws/telemetry", headers={"Origin": LOCAL}) as ws:
            assert ws.receive_json()["type"] == "STATUS"
    finally:
        await telem.close()
