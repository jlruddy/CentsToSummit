"""FastAPI REST + WebSocket backend for setup, inline API key testing, and live UI feed."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from kalshi_poc.auth import KalshiSigner, load_private_key, load_private_key_from_path
from kalshi_poc.client import KalshiApiError, KalshiRestClient
from kalshi_poc.config import ConfigManager
from kalshi_poc.models import Environment, ExecutionMode
from kalshi_poc.telemetry import TelemetryManager, export_parquet_zip

STATIC_DIR = Path(__file__).parent / "static"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _hostname(value: str) -> str | None:
    """Extract a lowercase hostname from a Host header or Origin URL (None if unparsable)."""
    value = value.strip()
    if not value or value == "null":
        return None
    try:
        parts = urlsplit(value if "://" in value else f"//{value}")
        return parts.hostname.lower() if parts.hostname else None
    except ValueError:
        return None


def is_allowed_host(host_header: str, extra_hosts: list[str]) -> bool:
    name = _hostname(host_header)
    return name is not None and (name in LOOPBACK_HOSTS or name in extra_hosts)


def is_allowed_origin(origin: str | None, extra_hosts: list[str]) -> bool:
    """Missing Origin (curl, scripts) is allowed; any present Origin must be an allowed host."""
    if origin is None:
        return True
    name = _hostname(origin)
    return name is not None and (name in LOOPBACK_HOSTS or name in extra_hosts)


class EngineController(Protocol):
    """Interface exposed by TradingEngine to the FastAPI layer."""

    def get_status(self) -> dict[str, Any]: ...
    async def start_engine(self) -> None: ...
    async def pause_engine(self) -> None: ...
    async def eval_now(self) -> dict[str, Any]: ...
    async def reset_shadow(self) -> dict[str, Any]: ...
    async def emergency_kill(self) -> dict[str, Any]: ...


class CredentialTestRequest(BaseModel):
    environment: Environment = Environment.DEMO
    api_key_id: str = ""
    private_key_pem: str = ""


class CredentialSaveRequest(BaseModel):
    environment: Environment = Environment.DEMO
    api_key_id: str
    private_key_pem: str = ""


def create_app(
    config_manager: ConfigManager | None = None,
    telemetry: TelemetryManager | None = None,
    engine: EngineController | None = None,
) -> FastAPI:
    """Create the FastAPI application with injected state."""
    app = FastAPI(title="Kalshi AI POC Control Panel", version="0.1.0")
    cfg_mgr = config_manager or ConfigManager()
    telem = telemetry or TelemetryManager(cfg_mgr.settings.db_path)

    app.state.config_manager = cfg_mgr
    app.state.telemetry = telem
    app.state.engine = engine
    app.state.last_cred_test_monotonic = 0.0
    app.state.cred_test_lock = asyncio.Lock()

    @app.middleware("http")
    async def host_and_origin_guard(request: Request, call_next: Any) -> Any:
        """Block DNS-rebinding (Host) and cross-site requests (Origin) to the control panel."""
        extra = app.state.config_manager.settings.web_allowed_hosts
        if not is_allowed_host(request.headers.get("host", ""), extra):
            return JSONResponse({"detail": "Host not allowed"}, status_code=403)
        if request.method not in SAFE_METHODS and not is_allowed_origin(
            request.headers.get("origin"), extra
        ):
            return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def serve_index() -> HTMLResponse:
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return app.state.config_manager.get_masked_config()

    @app.post("/api/config")
    async def update_config(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            new_s = app.state.config_manager.update_and_persist(payload)
            status = await get_status()
            await app.state.telemetry.broadcast("STATUS", status)
            warning = None
            if (
                new_s.execution_mode == ExecutionMode.DEMO
                and new_s.kalshi_environment == Environment.PRODUCTION
            ):
                warning = "Environment mismatch: 'demo' execution mode cannot run with 'production' environment."
            return {
                "ok": True,
                "warning": warning,
                "config": app.state.config_manager.get_masked_config(),
                "status": status,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/credentials/save")
    async def save_credentials(req: CredentialSaveRequest) -> dict[str, Any]:
        try:
            updates: dict[str, Any] = {
                "kalshi_environment": req.environment.value,
                "kalshi_api_key_id": req.api_key_id.strip(),
            }
            algo = None
            if req.private_key_pem.strip() and "••••" not in req.private_key_pem:
                key_path, algo = app.state.config_manager.save_private_key_pem(
                    req.private_key_pem, req.environment
                )
                updates["kalshi_private_key_path"] = str(key_path)
            app.state.config_manager.update_and_persist(updates)
            return {
                "ok": True,
                "algorithm": algo,
                "config": app.state.config_manager.get_masked_config(),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/credentials/test")
    async def test_credentials(req: CredentialTestRequest) -> dict[str, Any]:
        """Test unsaved or saved Kalshi credentials against /portfolio/balance (1 req/sec debounced)."""
        async with app.state.cred_test_lock:
            now = time.monotonic()
            elapsed = now - app.state.last_cred_test_monotonic
            if elapsed < 1.05:
                await asyncio.sleep(1.05 - elapsed)
            app.state.last_cred_test_monotonic = time.monotonic()

            s = app.state.config_manager.settings
            key_id = req.api_key_id.strip() or s.kalshi_api_key_id.strip()
            if not key_id:
                return {"ok": False, "algorithm": None, "error": "Kalshi API Key ID is required."}

            algo: str | None = None
            try:
                pem_input = req.private_key_pem.strip()
                if pem_input and "••••" not in pem_input:
                    priv_key, algo = load_private_key(pem_input)
                else:
                    priv_key, algo = load_private_key_from_path(s.resolved_private_key_path)
            except Exception as exc:
                return {"ok": False, "algorithm": None, "error": str(exc)}

            base_url = (
                "https://external-api.kalshi.com/trade-api/v2"
                if req.environment == Environment.PRODUCTION
                else "https://external-api.demo.kalshi.co/trade-api/v2"
            )
            signer = KalshiSigner(api_key_id=key_id, private_key=priv_key, algorithm=algo)
            client = KalshiRestClient(base_url=base_url, signer=signer, timeout_s=8.0)
            try:
                balance, latency_ms = await client.get_balance()
                return {
                    "ok": True,
                    "environment": req.environment.value,
                    "algorithm": algo,
                    "latency_ms": round(latency_ms, 1),
                    "balance_dollars": str(balance.balance_dollars),
                    "portfolio_value_dollars": str(balance.portfolio_value_dollars),
                }
            except KalshiApiError as exc:
                return {
                    "ok": False,
                    "environment": req.environment.value,
                    "algorithm": algo,
                    "error": f"HTTP {exc.status_code}: {exc.message}",
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "environment": req.environment.value,
                    "algorithm": algo,
                    "error": str(exc),
                }
            finally:
                await client.close()

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        if app.state.engine is not None:
            return app.state.engine.get_status()
        s = app.state.config_manager.settings
        budget = max(Decimal("0.00"), s.initial_capital - s.equity_floor)
        return {
            "engine_state": "PAUSED",
            "environment": s.kalshi_environment.value,
            "execution_mode": s.execution_mode.value,
            "kill_switch_tripped": False,
            "ws_health": {"connected": False, "reconnect_count": 0, "subscribed_tickers": []},
            "cash_balance_dollars": str(s.initial_capital),
            "current_equity_dollars": str(s.initial_capital),
            "equity_floor_dollars": str(s.equity_floor),
            "deployable_budget_dollars": str(budget),
            "session_net_pnl_dollars": "0.00",
            "estimated_llm_spend_usd": "0.0000",
            "orderbooks": {},
            "equity_history": [],
        }

    @app.post("/api/engine/start")
    async def start_engine() -> dict[str, Any]:
        if app.state.engine is not None:
            try:
                await app.state.engine.start_engine()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"ok": True, "status": app.state.engine.get_status()}
        app.state.config_manager.update_and_persist({"execution_mode": ExecutionMode.SHADOW.value})
        return {"ok": True}

    @app.post("/api/engine/pause")
    async def pause_engine() -> dict[str, Any]:
        if app.state.engine is not None:
            await app.state.engine.pause_engine()
            return {"ok": True, "status": app.state.engine.get_status()}
        app.state.config_manager.update_and_persist({"execution_mode": ExecutionMode.PAUSED.value})
        return {"ok": True}

    @app.post("/api/engine/eval-now")
    async def eval_now() -> dict[str, Any]:
        if app.state.engine is not None:
            res = await app.state.engine.eval_now()
            return {"ok": True, **res, "status": app.state.engine.get_status()}
        return {"ok": False, "error": "Engine not attached"}

    @app.post("/api/engine/reset-shadow")
    async def reset_shadow() -> dict[str, Any]:
        if app.state.engine is not None:
            res = await app.state.engine.reset_shadow()
            return {"ok": True, **res, "status": app.state.engine.get_status()}
        return {"ok": False, "error": "Engine not attached"}

    @app.post("/api/engine/kill")
    async def kill_engine() -> dict[str, Any]:
        if app.state.engine is not None:
            res = await app.state.engine.emergency_kill()
            return {"ok": True, **res}
        app.state.config_manager.update_and_persist({"execution_mode": ExecutionMode.PAUSED.value})
        return {"ok": True, "canceled_orders": 0}

    @app.get("/api/export/parquet")
    async def download_parquet() -> FileResponse:
        s = app.state.config_manager.settings
        export_dir = Path(s.db_path).parent / "exports"
        zip_path = await app.state.telemetry.export_parquet_zip_async(export_dir)
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename="kalshi_telemetry_parquet.zip",
        )

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(websocket: WebSocket) -> None:
        extra = app.state.config_manager.settings.web_allowed_hosts
        if not is_allowed_host(websocket.headers.get("host", ""), extra) or not is_allowed_origin(
            websocket.headers.get("origin"), extra
        ):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        q = app.state.telemetry.subscribe()
        try:
            # Send initial status snapshot and recent 50 activity events on connect
            status = await get_status()
            await websocket.send_json({"type": "STATUS", "data": status})
            recent = await app.state.telemetry.get_recent_activity(limit=50)
            for item in recent:
                await websocket.send_json(item)
            while True:
                event = await q.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            app.state.telemetry.unsubscribe(q)

    return app


app = create_app()
