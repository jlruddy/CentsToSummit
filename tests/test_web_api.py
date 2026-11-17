"""Test config persistence, masked secret handling, and inline API key test endpoint."""

import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi.testclient import TestClient

from kalshi_poc.config import ConfigManager
from kalshi_poc.models import BalanceResponse
from kalshi_poc.telemetry import TelemetryManager
from kalshi_poc.web_server import create_app


def _ed25519_pem() -> str:
    priv = ed25519.Ed25519PrivateKey.generate()
    return priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def test_config_persistence_and_masked_secrets(tmp_path: Path) -> None:
    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg_mgr.update_and_persist({"strategy_provider": "stub_momentum"})
    telem = TelemetryManager(tmp_path / "telem.db")
    app = create_app(config_manager=cfg_mgr, telemetry=telem)
    client = TestClient(app)

    pem = _ed25519_pem()
    save_resp = client.post(
        "/api/credentials/save",
        json={
            "environment": "demo",
            "api_key_id": "demo-key-123",
            "private_key_pem": pem,
        },
    )
    assert save_resp.status_code == 200
    assert save_resp.json()["algorithm"] == "Ed25519"

    saved_key_file = tmp_path / "keys" / "demo_private_key.pem"
    assert saved_key_file.exists()
    assert (os.stat(saved_key_file).st_mode & 0o777) == 0o600

    # GET /api/config must never return raw private key or raw LLM key
    client.post("/api/config", json={"llm_api_key": "sk-secret-value", "equity_floor": "1500.00"})
    cfg_resp = client.get("/api/config").json()
    assert cfg_resp["private_key_configured"] is True
    assert cfg_resp["private_key_masked"] == "••••configured"
    assert cfg_resp["llm_api_key"] == "••••configured"
    assert cfg_resp["equity_floor"] == "1500.00"
    assert "BEGIN PRIVATE KEY" not in str(cfg_resp)


def test_inline_credentials_test_endpoint(tmp_path: Path) -> None:
    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg_mgr.update_and_persist({"strategy_provider": "stub_momentum"})
    telem = TelemetryManager(tmp_path / "telem.db")
    app = create_app(config_manager=cfg_mgr, telemetry=telem)
    client = TestClient(app)

    pem = _ed25519_pem()
    mock_bal = BalanceResponse(
        balance_dollars=Decimal("5000.00"),
        portfolio_value_dollars=Decimal("5125.50"),
    )

    with patch(
        "kalshi_poc.web_server.KalshiRestClient.get_balance",
        new=AsyncMock(return_value=(mock_bal, 42.5)),
    ):
        resp = client.post(
            "/api/credentials/test",
            json={
                "environment": "demo",
                "api_key_id": "unsaved-key-id",
                "private_key_pem": pem,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["algorithm"] == "Ed25519"
    assert body["latency_ms"] == 42.5
    assert body["balance_dollars"] == "5000.00"
    assert body["portfolio_value_dollars"] == "5125.50"

    # Invalid PEM returns diagnostic error
    bad_resp = client.post(
        "/api/credentials/test",
        json={
            "environment": "demo",
            "api_key_id": "unsaved-key-id",
            "private_key_pem": "not-a-valid-pem",
        },
    )
    assert bad_resp.status_code == 200
    assert bad_resp.json()["ok"] is False
    assert "Invalid PEM" in bad_resp.json()["error"]


def test_hot_reload_initial_capital_updates_cash_and_budget(tmp_path: Path) -> None:
    from kalshi_poc.main import TradingEngine

    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg_mgr.update_and_persist({"strategy_provider": "stub_momentum"})
    telem = TelemetryManager(tmp_path / "telem.db")
    engine = TradingEngine(cfg_mgr, telem)
    app = create_app(config_manager=cfg_mgr, telemetry=telem, engine=engine)
    client = TestClient(app)

    resp = client.post(
        "/api/config",
        json={
            "execution_mode": "shadow",
            "initial_capital": "12.00",
            "equity_floor": "1.00",
        },
    )
    assert resp.status_code == 200
    status = resp.json()["status"]
    assert status["cash_balance_dollars"] == "12.00"
    assert status["deployable_budget_dollars"] == "11.00"
    assert engine.portfolio.cash_balance_dollars == Decimal("12.00")


async def test_ws_client_parses_fp_snapshot_delta_and_ticker() -> None:
    import json
    from kalshi_poc.client import KalshiWebSocketClient

    ws = KalshiWebSocketClient(ws_url="wss://example.invalid")
    await ws._handle_message(
        json.dumps(
            {
                "type": "orderbook_snapshot",
                "msg": {
                    "market_ticker": "KXBTC-26NOV-T90000",
                    "yes_dollars_fp": [["0.4800", "10.00"]],
                    "no_dollars_fp": [["0.5000", "15.00"]],
                },
            }
        )
    )
    snap = ws.orderbooks["KXBTC-26NOV-T90000"]
    assert snap.best_yes_bid == Decimal("0.4800")
    assert snap.best_yes_ask == Decimal("0.5000")
    assert snap.yes_spread == Decimal("0.0200")

    # Delta on a new ticker without prior snapshot should initialize and apply cleanly
    await ws._handle_message(
        json.dumps(
            {
                "type": "orderbook_delta",
                "msg": {
                    "market_ticker": "KXETH-26NOV-T3000",
                    "side": "yes",
                    "price_dollars": "0.4200",
                    "delta_fp": "5.00",
                },
            }
        )
    )
    assert ws.orderbooks["KXETH-26NOV-T3000"].best_yes_bid == Decimal("0.4200")


async def test_discover_active_markets_prefers_nearest_1_to_3h_near_money() -> None:
    from datetime import datetime, timedelta, timezone
    from kalshi_poc.client import KalshiRestClient
    from kalshi_poc.models import MarketTicker

    now = datetime(2026, 11, 16, 20, 0, 0, tzinfo=timezone.utc)
    markets = [
        # 24h out stale penny book
        MarketTicker(
            ticker="KXBTC-26NOV1717-T95000",
            series_ticker="KXBTC",
            yes_bid_dollars=Decimal("0.01"),
            yes_ask_dollars=Decimal("0.99"),
            close_time=now + timedelta(hours=24),
        ),
        # 2h out deep-OTM penny book
        MarketTicker(
            ticker="KXBTC-26NOV1622-T99000",
            series_ticker="KXBTC",
            yes_bid_dollars=Decimal("0.01"),
            yes_ask_dollars=Decimal("0.02"),
            close_time=now + timedelta(hours=2),
        ),
        # 2h out near-the-money active book
        MarketTicker(
            ticker="KXBTC-26NOV1622-T91000",
            series_ticker="KXBTC",
            yes_bid_dollars=Decimal("0.48"),
            yes_ask_dollars=Decimal("0.52"),
            close_time=now + timedelta(hours=2),
        ),
    ]
    client = KalshiRestClient(base_url="https://external-api.demo.kalshi.co/trade-api/v2")
    with patch.object(client, "get_markets", new=AsyncMock(return_value=markets)):
        selected = await client.discover_active_markets(
            ["KXBTC"], max_tickers=2, environment="production", now_utc=now
        )
    await client.close()

    assert [m.ticker for m in selected] == ["KXBTC-26NOV1622-T91000"]


async def test_discover_active_markets_demo_fallback_by_volume() -> None:
    from datetime import datetime, timedelta, timezone
    from kalshi_poc.client import KalshiRestClient
    from kalshi_poc.models import MarketTicker

    now = datetime(2026, 11, 16, 20, 0, 0, tzinfo=timezone.utc)
    # 0 two-sided markets in allowed prefix, but active two-sided markets in general open feed
    open_markets = [
        MarketTicker(
            ticker="DEMO-LOW-VOL",
            yes_bid_dollars=Decimal("0.40"),
            yes_ask_dollars=Decimal("0.44"),
            volume_fp=Decimal("50"),
            close_time=now + timedelta(hours=5),
        ),
        MarketTicker(
            ticker="DEMO-HIGH-VOL",
            yes_bid_dollars=Decimal("0.48"),
            yes_ask_dollars=Decimal("0.52"),
            volume_fp=Decimal("5000"),
            close_time=now + timedelta(hours=5),
        ),
        MarketTicker(
            ticker="DEMO-ONE-SIDED",
            yes_bid_dollars=Decimal("0.00"),
            yes_ask_dollars=Decimal("1.00"),
            volume_fp=Decimal("99999"),
            close_time=now + timedelta(hours=5),
        ),
    ]

    get_mock = AsyncMock(return_value=open_markets)
    client = KalshiRestClient(base_url="https://external-api.demo.kalshi.co/trade-api/v2")
    with patch.object(client, "get_markets", get_mock):
        demo_selected = await client.discover_active_markets(
            ["KXBTC"], max_tickers=6, environment="demo", now_utc=now, max_per_series=2
        )
        assert get_mock.call_count >= 1
        assert get_mock.call_args_list[0].kwargs["series_ticker"] == "KXBTC15M"
        assert get_mock.call_args_list[0].kwargs["limit"] == 200
        prod_selected = await client.discover_active_markets(
            ["KXBTC"], max_tickers=6, environment="production", now_utc=now
        )
    await client.close()

    assert [m.ticker for m in demo_selected] == ["DEMO-HIGH-VOL", "DEMO-LOW-VOL"]
    assert prod_selected == []


def test_eval_now_emits_hold_ai_decision_and_derived_yes_asks(tmp_path: Path) -> None:
    from kalshi_poc.client import KalshiWebSocketClient
    from kalshi_poc.main import TradingEngine
    from kalshi_poc.models import OrderbookSnapshot, PriceLevel

    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg_mgr.update_and_persist({"strategy_provider": "stub_momentum"})
    telem = TelemetryManager(tmp_path / "telem.db")
    engine = TradingEngine(cfg_mgr, telem)
    engine.ws_client = KalshiWebSocketClient(ws_url="wss://example.invalid")

    # NO bids at 0.01 (10) and 0.02 (20) -> Derived YES asks: 0.9800 (20), 0.9900 (10)
    snap = OrderbookSnapshot(
        ticker="KXBTC-26NOV1622-T91000",
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("30"))],
        no_bids=[
            PriceLevel(price=Decimal("0.01"), count=Decimal("10")),
            PriceLevel(price=Decimal("0.02"), count=Decimal("20")),
        ],
    )
    asks = snap.effective_yes_asks
    assert [a.price for a in asks] == [Decimal("0.9800"), Decimal("0.9900")]

    engine.ws_client.orderbooks[snap.ticker] = snap
    q = telem.subscribe()

    app = create_app(config_manager=cfg_mgr, telemetry=telem, engine=engine)
    client = TestClient(app)
    resp = client.post("/api/engine/eval-now")
    assert resp.status_code == 200
    assert resp.json()["evaluated"] == 1

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    ai_events = [e for e in events if e["type"] == "AI_DECISION"]
    assert len(ai_events) == 1
    assert ai_events[0]["data"]["action"] == "HOLD"
    assert ai_events[0]["data"]["ticker"] == "KXBTC-26NOV1622-T91000"


async def test_ws_per_ticker_isolation_and_sid_routing() -> None:
    import json
    from kalshi_poc.client import KalshiWebSocketClient

    ws = KalshiWebSocketClient(ws_url="wss://example.invalid")
    await ws._handle_message(
        json.dumps(
            {
                "type": "orderbook_snapshot",
                "sid": 1,
                "msg": {
                    "market_ticker": "TICKER-A",
                    "yes_dollars": [["0.6800", "25"]],
                    "no_dollars": [["0.2900", "40"]],
                },
            }
        )
    )
    await ws._handle_message(
        json.dumps(
            {
                "type": "orderbook_snapshot",
                "sid": 2,
                "msg": {
                    "market_ticker": "TICKER-B",
                    "yes_dollars": [["0.4000", "10"]],
                    "no_dollars": [["0.5500", "15"]],
                },
            }
        )
    )
    # Delta routed via sid=1 without market_ticker
    await ws._handle_message(
        json.dumps(
            {
                "type": "orderbook_delta",
                "sid": 1,
                "msg": {"side": "yes", "price_dollars": "0.6900", "delta_fp": "12"},
            }
        )
    )

    book_a = ws.orderbooks["TICKER-A"]
    book_b = ws.orderbooks["TICKER-B"]
    assert book_a.best_yes_bid == Decimal("0.6900")
    assert book_a.best_yes_ask == Decimal("0.7100")  # 1.00 - 0.29
    assert book_a.yes_spread == Decimal("0.0200")

    assert book_b.best_yes_bid == Decimal("0.4000")
    assert book_b.best_yes_ask == Decimal("0.4500")  # 1.00 - 0.55
    assert book_b.yes_spread == Decimal("0.0500")


async def test_reset_shadow_clears_kill_switch_flattens_and_rotates_session(tmp_path: Path) -> None:
    from kalshi_poc.main import TradingEngine
    from kalshi_poc.models import Position

    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg_mgr.update_and_persist({"initial_capital": "12.40", "equity_floor": "1.00"})
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()

    engine = TradingEngine(cfg_mgr, telem)
    engine.portfolio.kill_switch_tripped = True
    engine.portfolio.cash_balance_dollars = Decimal("6.51")
    engine.portfolio.positions["KXBTC-TEST"] = Position(
        ticker="KXBTC-TEST",
        net_contracts=Decimal("5"),
        avg_entry_price=Decimal("0.50"),
        cost_basis_dollars=Decimal("2.50"),
    )
    old_session = telem.session_id

    app = create_app(config_manager=cfg_mgr, telemetry=telem, engine=engine)
    client = TestClient(app)
    try:
        resp = client.post("/api/engine/reset-shadow")

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["session_id"] != old_session
        assert body["status"]["engine_state"] == "PAUSED"
        assert body["status"]["kill_switch_tripped"] is False
        assert body["status"]["cash_balance_dollars"] == "12.40"
        assert body["status"]["session_net_pnl_dollars"] == "0.00"
        assert engine.risk_gate.kill_switch_tripped is False
        assert engine.portfolio.positions == {}

        # Subsequent heartbeat / SQLite equity query stays clean
        subsequent_status = engine.get_status()
        assert subsequent_status["cash_balance_dollars"] == "12.40"
        assert subsequent_status["session_net_pnl_dollars"] == "0.00"
        assert subsequent_status["kill_switch_tripped"] is False
        hist = await telem.get_session_equity_history()
        assert len(hist) >= 1
        assert all(
            pt["timestamp_ms"] >= telem.session_start_ts and pt["equity"] == "12.40"
            for pt in hist
        )
    finally:
        await telem.close()


async def test_execution_error_surfaced_and_demo_prod_mismatch_blocked(tmp_path: Path) -> None:
    from kalshi_poc.client import KalshiApiError, KalshiRestClient, KalshiWebSocketClient
    from kalshi_poc.main import TradingEngine
    from kalshi_poc.models import OrderbookSnapshot, PriceLevel

    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg_mgr.update_and_persist(
        {
            "kalshi_environment": "demo",
            "execution_mode": "demo",
            "initial_capital": "100.00",
            "equity_floor": "10.00",
            "strategy_provider": "stub_momentum",
        }
    )
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()

    engine = TradingEngine(cfg_mgr, telem)
    engine.ws_client = KalshiWebSocketClient(ws_url="wss://example.invalid")
    engine.rest_client = KalshiRestClient(
        base_url="https://external-api.demo.kalshi.co/trade-api/v2",
        signer=AsyncMock(),
    )

    snap = OrderbookSnapshot(
        ticker="KXBTC-26NOV1622-T91000",
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.48"), count=Decimal("50"))],
        no_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("10"))],
    )
    engine.ws_client.set_snapshot(snap)
    q = telem.subscribe()

    try:
        with patch.object(
            engine.rest_client,
            "create_order",
            new=AsyncMock(side_effect=KalshiApiError(401, "Invalid API key for demo")),
        ):
            await engine._process_snapshot(snap, cfg_mgr.settings)

        events = []
        while not q.empty():
            events.append(q.get_nowait())
        exec_events = [e for e in events if e["type"] == "EXECUTION"]
        assert len(exec_events) == 1
        assert exec_events[0]["data"]["status"] == "REJECTED_BY_EXCHANGE"
        assert "401" in exec_events[0]["data"]["error_message"]

        # Environment mismatch: demo mode + production environment blocked on start_engine
        cfg_mgr.update_and_persist({"kalshi_environment": "production", "execution_mode": "demo"})
        app = create_app(config_manager=cfg_mgr, telemetry=telem, engine=engine)
        client = TestClient(app)
        resp = client.post("/api/engine/start")
        assert resp.status_code == 400
        assert "demo" in resp.json()["detail"]
    finally:
        await engine.rest_client.close()
        await telem.close()


async def test_discover_active_markets_filters_atm_tight_spread_and_earliest_expiry() -> None:
    from datetime import datetime, timedelta, timezone
    from kalshi_poc.client import KalshiRestClient
    from kalshi_poc.models import MarketTicker

    now = datetime(2026, 11, 16, 23, 0, 0, tzinfo=timezone.utc)
    markets = [
        # Deep OTM ($0.05/$0.06) -> excluded
        MarketTicker(
            ticker="KXWTI-DEEP-OTM",
            yes_bid_dollars=Decimal("0.05"),
            yes_ask_dollars=Decimal("0.06"),
            volume_fp=Decimal("10000"),
            close_time=now + timedelta(minutes=10),
        ),
        # Wide spread ($0.05/$0.19) -> excluded
        MarketTicker(
            ticker="KXINX-WIDE",
            yes_bid_dollars=Decimal("0.05"),
            yes_ask_dollars=Decimal("0.19"),
            volume_fp=Decimal("10000"),
            close_time=now + timedelta(minutes=10),
        ),
        # ATM tight spread expiring in 2h
        MarketTicker(
            ticker="KXBTCD-2H",
            yes_bid_dollars=Decimal("0.48"),
            yes_ask_dollars=Decimal("0.52"),
            volume_fp=Decimal("500"),
            close_time=now + timedelta(hours=2),
        ),
        # ATM tight spread expiring in 15m -> should be first!
        MarketTicker(
            ticker="KXBTC15M-15M",
            yes_bid_dollars=Decimal("0.46"),
            yes_ask_dollars=Decimal("0.50"),
            volume_fp=Decimal("200"),
            close_time=now + timedelta(minutes=15),
        ),
    ]

    client = KalshiRestClient(base_url="https://external-api.demo.kalshi.co/trade-api/v2")
    with patch.object(client, "get_markets", new=AsyncMock(return_value=markets)):
        selected = await client.discover_active_markets(
            ["KXBTC15M", "KXBTCD"], max_tickers=6, environment="production", now_utc=now
        )
    await client.close()

    # Both share underlying BTC (max_per_series=1), and 15m expiry bucket ranks first
    assert [m.ticker for m in selected] == ["KXBTC15M-15M"]


async def test_ws_telemetry_hydrates_recent_history_on_connect(tmp_path: Path) -> None:
    from kalshi_poc.models import ExecutionMode, ExecutionRecord, OrderSide

    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    try:
        await telem.log_execution(
            ExecutionRecord(
                order_id="ord-1",
                client_order_id="cli-1",
                intent_id="int-1",
                ticker="KXBTC15M-TEST",
                mode=ExecutionMode.SHADOW,
                side=OrderSide.BID,
                count=Decimal("2"),
                requested_price=Decimal("0.48"),
                fill_price=Decimal("0.48"),
                slippage_dollars=Decimal("0.00"),
                fee_dollars=Decimal("0.04"),
                status="SHADOW_SETTLEMENT",
                timestamp_ms=1700000000000,
            )
        )
        app = create_app(config_manager=cfg_mgr, telemetry=telem)
        client = TestClient(app)
        with client.websocket_connect("/ws/telemetry") as ws:
            msg1 = ws.receive_json()
            msg2 = ws.receive_json()
        assert msg1["type"] == "STATUS"
        assert msg2["type"] == "EXECUTION"
        assert msg2["data"]["status"] == "SHADOW_SETTLEMENT"
    finally:
        await telem.close()


async def test_shard_transfer_retry_and_tier_upgrade(tmp_path: Path) -> None:
    from kalshi_poc.client import KalshiApiError, KalshiRestClient, KalshiWebSocketClient
    from kalshi_poc.main import TradingEngine
    from kalshi_poc.models import OrderbookSnapshot, PriceLevel

    client = KalshiRestClient(
        base_url="https://external-api.demo.kalshi.co/trade-api/v2",
        signer=AsyncMock(),
    )
    client.market_shards["KXBTCD-SHARD2"] = 2

    req_mock = AsyncMock(return_value=({}, 10.0))
    with patch.object(client, "_request", req_mock):
        await client.fund_shard(2, Decimal("11.40"))
        req_mock.assert_awaited_once()
        _, ep = req_mock.call_args.args
        assert ep == "/portfolio/intra_exchange_instance_transfer"
        assert req_mock.call_args.kwargs["json_body"] == {
            "source": "event_contract",
            "destination": "event_contract",
            "source_exchange_shard": 0,
            "destination_exchange_shard": 2,
            "source_subaccount": 0,
            "destination_subaccount": 0,
            "amount": 114000,
        }

        # get_market_exchange_index fetches GET /markets/{ticker} when not cached
        req_mock.reset_mock()
        req_mock.return_value = ({"market": {"ticker": "KXETH15M-1", "exchange_index": 3}}, 5.0)
        shard = await client.get_market_exchange_index("KXETH15M-1")
        assert shard == 3
        assert client.market_shards["KXETH15M-1"] == 3

        # BalanceResponse parses balance_breakdown per exchange_index
        bal = BalanceResponse.from_api_payload(
            {
                "balance_dollars": "12.40",
                "portfolio_value_dollars": "12.40",
                "balance_breakdown": [
                    {"exchange_index": 0, "balance_dollars": "10.00"},
                    {"exchange_index": 3, "balance_dollars": "2.40"},
                ],
            }
        )
        assert bal.balance_breakdown == {0: Decimal("10.00"), 3: Decimal("2.40")}

        # upgrade_api_tier_once ignores 403 and only fires once
        req_mock.reset_mock()
        req_mock.side_effect = KalshiApiError(403, "Already upgraded")
        await client.upgrade_api_tier_once()
        await client.upgrade_api_tier_once()
        assert req_mock.call_count == 1

    # Test insufficient_shard_balance triggers fund_shard + retry once in TradingEngine
    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg_mgr.update_and_persist(
        {
            "kalshi_environment": "demo",
            "execution_mode": "demo",
            "initial_capital": "12.40",
            "equity_floor": "1.00",
            "strategy_provider": "stub_momentum",
        }
    )
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    try:
        engine = TradingEngine(cfg_mgr, telem)
        engine.ws_client = KalshiWebSocketClient(ws_url="wss://example.invalid")
        engine.rest_client = client
        snap = OrderbookSnapshot(
            ticker="KXBTCD-SHARD2",
            timestamp_ms=1700000000000,
            yes_bids=[PriceLevel(price=Decimal("0.48"), count=Decimal("50"))],
            no_bids=[PriceLevel(price=Decimal("0.50"), count=Decimal("10"))],
        )
        engine.ws_client.set_snapshot(snap)

        create_mock = AsyncMock(
            side_effect=[
                KalshiApiError(404, "insufficient_shard_balance"),
                {"order": {"order_id": "ord-retry-ok", "status": "executed"}},
            ]
        )
        fund_mock = AsyncMock(return_value={})
        upgrade_mock = AsyncMock(return_value=None)
        with (
            patch.object(client, "create_order", create_mock),
            patch.object(client, "fund_shard", fund_mock),
            patch.object(client, "upgrade_api_tier_once", upgrade_mock),
        ):
            await engine._process_snapshot(snap, cfg_mgr.settings)

        assert create_mock.call_count == 2
        assert fund_mock.await_count == 2
        upgrade_mock.assert_awaited_once()
    finally:
        await client.close()
        await telem.close()


async def test_v2_order_endpoints_cancellation_and_fill_reconciliation(tmp_path: Path) -> None:
    from kalshi_poc.client import KalshiRestClient, KalshiWebSocketClient
    from kalshi_poc.main import TradingEngine
    from kalshi_poc.models import OrderSide, OrderbookSnapshot, PriceLevel, TradeIntent, V2CreateOrderRequest

    client = KalshiRestClient(
        base_url="https://external-api.demo.kalshi.co/trade-api/v2",
        signer=AsyncMock(),
    )
    req_mock = AsyncMock(return_value=({}, 5.0))
    with patch.object(client, "_request", req_mock):
        v2_req = V2CreateOrderRequest(
            ticker="KXBTCD-TEST",
            side="bid",
            count="4",
            price="0.4800",
        )
        await client.create_order(v2_req)
        assert req_mock.call_args_list[0].args == ("POST", "/portfolio/events/orders")

        await client.cancel_order("ord-123", market_ticker="KXBTCD-TEST")
        assert req_mock.call_args_list[1].args == ("DELETE", "/portfolio/events/orders/ord-123")
        assert req_mock.call_args_list[1].kwargs["params"] == {"market_ticker": "KXBTCD-TEST"}

        await client.cancel_all_orders()
        assert req_mock.call_args_list[2].args == ("DELETE", "/portfolio/events/orders")

    # Verify V2 fill response reconciliation updates cash_balance and positions
    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg_mgr.update_and_persist(
        {
            "kalshi_environment": "demo",
            "execution_mode": "demo",
            "initial_capital": "12.40",
            "equity_floor": "1.00",
            "strategy_provider": "stub_momentum",
        }
    )
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    try:
        engine = TradingEngine(cfg_mgr, telem)
        engine.ws_client = KalshiWebSocketClient(ws_url="wss://example.invalid")
        engine.rest_client = client
        snap = OrderbookSnapshot(
            ticker="KXBTCD-TEST",
            timestamp_ms=1700000000000,
            yes_bids=[PriceLevel(price=Decimal("0.47"), count=Decimal("50"))],
            no_bids=[PriceLevel(price=Decimal("0.52"), count=Decimal("50"))],
        )
        intent = TradeIntent(
            timestamp_ms=1700000000000,
            ticker="KXBTCD-TEST",
            side=OrderSide.BID,
            count=Decimal("4"),
            limit_price=Decimal("0.48"),
            expected_edge_dollars=Decimal("0.20"),
            expected_edge_per_contract=Decimal("0.05"),
            hold_to_expiry=True,
            confidence=Decimal("0.80"),
            inference_latency_ms=5,
            rationale="V2 fill test",
            provider="stub_momentum",
        )
        v2_fill_resp = {
            "order_id": "v2-ord-999",
            "fill_count": "4",
            "remaining_count": "0",
            "average_fill_price": "0.4800",
            "average_fee_paid": "0.07",
            "ts_ms": 1700000001000,
        }
        with (
            patch.object(client, "create_order", AsyncMock(return_value=v2_fill_resp)),
            patch.object(client, "upgrade_api_tier_once", AsyncMock(return_value=None)),
        ):
            await engine._evaluate_and_execute_intent(intent, snap, cfg_mgr.settings)

        # Cash reduced by 4 * 0.48 + 0.07 = 1.99 -> 12.40 - 1.99 = 10.41
        assert engine.portfolio.cash_balance_dollars == Decimal("10.4100")
        assert engine.portfolio.positions["KXBTCD-TEST"].net_contracts == Decimal("4")
    finally:
        await client.close()
        await telem.close()


async def test_discover_active_markets_strict_two_sided_and_per_series_cap() -> None:
    from datetime import datetime, timedelta, timezone
    from kalshi_poc.client import KalshiRestClient
    from kalshi_poc.models import MarketTicker

    now = datetime(2026, 11, 17, 10, 0, 0, tzinfo=timezone.utc)
    markets = [
        # Zero-bid KXWTIH -> must be rejected
        *[
            MarketTicker(
                ticker=f"KXWTIH-DEAD-{i}",
                series_ticker="KXWTIH",
                yes_bid_dollars=Decimal("0.00"),
                no_bid_dollars=Decimal("0.22"),
                volume_fp=Decimal("50000"),
                close_time=now + timedelta(minutes=30),
            )
            for i in range(6)
        ],
        # 4 liquid KXBTC15M contracts -> capped at 2!
        *[
            MarketTicker(
                ticker=f"KXBTC15M-{i}",
                series_ticker="KXBTC15M",
                yes_bid_dollars=Decimal("0.48"),
                yes_ask_dollars=Decimal("0.51"),  # 3c spread
                volume_fp=Decimal(str(1000 + i)),
                close_time=now + timedelta(minutes=15),
            )
            for i in range(4)
        ],
        # 2 liquid KXETH15M contracts with tighter 2c spread -> ranked first!
        MarketTicker(
            ticker="KXETH15M-0",
            series_ticker="KXETH15M",
            yes_bid_dollars=Decimal("0.49"),
            yes_ask_dollars=Decimal("0.51"),  # 2c spread
            volume_fp=Decimal("500"),
            close_time=now + timedelta(minutes=15),
        ),
        MarketTicker(
            ticker="KXETH15M-1",
            series_ticker="KXETH15M",
            yes_bid_dollars=Decimal("0.49"),
            yes_ask_dollars=Decimal("0.51"),  # 2c spread
            volume_fp=Decimal("400"),
            close_time=now + timedelta(minutes=15),
        ),
    ]

    client = KalshiRestClient(base_url="https://external-api.demo.kalshi.co/trade-api/v2")
    with patch.object(client, "get_markets", AsyncMock(return_value=markets)):
        selected = await client.discover_active_markets(
            ["KXBTC15M", "KXETH15M"], max_tickers=6, environment="production", now_utc=now
        )
    await client.close()

    tickers = [m.ticker for m in selected]
    assert not any(t.startswith("KXWTIH") for t in tickers)
    assert set(tickers) == {"KXETH15M-0", "KXBTC15M-3"}


async def test_discover_active_markets_deduplicates_underlying_assets() -> None:
    from datetime import datetime, timedelta, timezone
    from kalshi_poc.client import KalshiRestClient
    from kalshi_poc.models import MarketTicker

    now = datetime(2026, 11, 17, 10, 30, 0, tzinfo=timezone.utc)
    markets = [
        MarketTicker(
            ticker="KXINX-1",
            series_ticker="KXINX",
            yes_bid_dollars=Decimal("0.49"),
            yes_ask_dollars=Decimal("0.51"),
            volume_fp=Decimal("1000"),
            close_time=now + timedelta(hours=1),
        ),
        MarketTicker(
            ticker="KXINXU-1",
            series_ticker="KXINXU",
            yes_bid_dollars=Decimal("0.49"),
            yes_ask_dollars=Decimal("0.51"),
            volume_fp=Decimal("900"),
            close_time=now + timedelta(hours=1),
        ),
        MarketTicker(
            ticker="KXINXU-2",
            series_ticker="KXINXU",
            yes_bid_dollars=Decimal("0.49"),
            yes_ask_dollars=Decimal("0.51"),
            volume_fp=Decimal("800"),
            close_time=now + timedelta(hours=1),
        ),
        MarketTicker(
            ticker="KXBTC15M-1",
            series_ticker="KXBTC15M",
            yes_bid_dollars=Decimal("0.48"),
            yes_ask_dollars=Decimal("0.52"),
            volume_fp=Decimal("500"),
            close_time=now + timedelta(minutes=15),
        ),
    ]
    client = KalshiRestClient(base_url="https://external-api.demo.kalshi.co/trade-api/v2")
    with patch.object(client, "get_markets", AsyncMock(return_value=markets)):
        selected = await client.discover_active_markets(
            ["KXINX", "KXINXU", "KXBTC15M"],
            max_tickers=6,
            environment="production",
            now_utc=now,
            max_per_series=2,
        )
    await client.close()

    tickers = [m.ticker for m in selected]
    # KXINX + KXINXU share the "INX" cap of 2 -> KXBTC15M-1 is included!
    assert sum(1 for t in tickers if t.startswith("KXINX")) == 2
    assert "KXBTC15M-1" in tickers


async def test_evict_widened_off_band_and_expiring_tickers(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone
    from kalshi_poc.client import KalshiWebSocketClient
    from kalshi_poc.main import TradingEngine
    from kalshi_poc.models import OrderbookSnapshot, PriceLevel

    cfg_mgr = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    telem = TelemetryManager(tmp_path / "telem.db")
    engine = TradingEngine(cfg_mgr, telem)
    engine.ws_client = KalshiWebSocketClient(ws_url="wss://example.invalid")

    now = datetime(2026, 11, 17, 13, 0, 0, tzinfo=timezone.utc)
    # KXNASDAQDUD widened to $0.03 / $0.19 ($0.16 spread > $0.05)
    bad_spread_snap = OrderbookSnapshot(
        ticker="KXNASDAQDUD-WIDE",
        timestamp_ms=int(now.timestamp() * 1000),
        yes_bids=[PriceLevel(price=Decimal("0.03"), count=Decimal("10"))],
        no_bids=[PriceLevel(price=Decimal("0.81"), count=Decimal("10"))],  # ask=0.19
    )
    # Healthy KXBTC15M ($0.48 / $0.51 -> $0.03 spread, mid 0.495)
    good_snap = OrderbookSnapshot(
        ticker="KXBTC15M-GOOD",
        timestamp_ms=int(now.timestamp() * 1000),
        yes_bids=[PriceLevel(price=Decimal("0.48"), count=Decimal("10"))],
        no_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("10"))],
    )
    # Empty-book (Spread: --) after 10s grace period -> must be evicted!
    empty_snap = OrderbookSnapshot(
        ticker="KXSOL15M-EMPTY",
        timestamp_ms=int(now.timestamp() * 1000),
        yes_bids=[],
        no_bids=[],
    )
    engine.ws_client.tickers = ["KXNASDAQDUD-WIDE", "KXBTC15M-GOOD", "KXSOL15M-EMPTY"]
    engine.ws_client.set_snapshot(bad_spread_snap)
    engine.ws_client.set_snapshot(good_snap)
    engine.ws_client.set_snapshot(empty_snap)
    engine._market_expiries["KXNASDAQDUD-WIDE"] = now + timedelta(minutes=30)
    engine._market_expiries["KXBTC15M-GOOD"] = now + timedelta(minutes=15)
    engine._market_expiries["KXSOL15M-EMPTY"] = now + timedelta(minutes=10)
    # All subscribed 20s ago (past 10s grace period)
    engine._subscribed_at = {
        "KXNASDAQDUD-WIDE": 100.0,
        "KXBTC15M-GOOD": 100.0,
        "KXSOL15M-EMPTY": 100.0,
    }

    evicted = await engine._evict_unhealthy_subscriptions(cfg_mgr.settings, now, now_mono=120.0)
    assert evicted == ["KXNASDAQDUD-WIDE", "KXSOL15M-EMPTY"]
    assert engine.ws_client.tickers == ["KXBTC15M-GOOD"]
    assert "KXSOL15M-EMPTY" not in engine.ws_client.orderbooks


async def test_discovery_skips_under_125s_and_falls_back_to_hourly_crypto_and_nasdaqdud() -> None:
    from datetime import datetime, timedelta, timezone
    from kalshi_poc.client import KalshiRestClient
    from kalshi_poc.models import MarketTicker

    # 13:28:30 ET (during US market hours, 90s before 13:30:00 rollover)
    now = datetime(2026, 11, 17, 18, 28, 30, tzinfo=timezone.utc)

    async def fake_get_markets(series_ticker=None, status="open", limit=200, **kwargs):
        assert limit == 200
        if series_ticker == "KXBTC15M":
            return [
                MarketTicker(
                    ticker="KXBTC15M-DYING-90S",
                    series_ticker="KXBTC15M",
                    yes_bid_dollars=Decimal("0.49"),
                    yes_ask_dollars=Decimal("0.51"),
                    close_time=now + timedelta(seconds=90),  # < 125s -> skipped!
                )
            ]
        if series_ticker == "KXBTCD":
            return [
                # Deep ITM tail -> strictly rejected
                MarketTicker(
                    ticker="KXBTCD-TAIL-ITM",
                    series_ticker="KXBTCD",
                    yes_bid_dollars=Decimal("0.97"),
                    yes_ask_dollars=Decimal("0.99"),
                    close_time=now + timedelta(minutes=31),
                ),
                # Deep OTM tail -> strictly rejected
                MarketTicker(
                    ticker="KXBTCD-TAIL-OTM",
                    series_ticker="KXBTCD",
                    yes_bid_dollars=Decimal("0.00"),
                    yes_ask_dollars=Decimal("0.01"),
                    close_time=now + timedelta(minutes=31),
                ),
                # ATM hourly strike -> selected via fallback!
                MarketTicker(
                    ticker="KXBTCD-ATM-HOURLY",
                    series_ticker="KXBTCD",
                    yes_bid_dollars=Decimal("0.49"),
                    yes_ask_dollars=Decimal("0.52"),
                    close_time=now + timedelta(minutes=31),
                ),
            ]
        if series_ticker == "KXSOL15M":
            return [
                MarketTicker(
                    ticker="KXSOL15M-OK-130S",
                    series_ticker="KXSOL15M",
                    yes_bid_dollars=Decimal("0.48"),
                    yes_ask_dollars=Decimal("0.52"),
                    close_time=now + timedelta(seconds=130),  # >= 125s -> eligible!
                ),
            ]
        if series_ticker == "KXINXU":
            # Only extreme tails -> must select NONE (no blind fallback!)
            return [
                MarketTicker(
                    ticker="KXINXU-TAIL-HIGH",
                    series_ticker="KXINXU",
                    yes_bid_dollars=Decimal("0.97"),
                    yes_ask_dollars=Decimal("0.99"),
                    close_time=now + timedelta(hours=1),
                ),
                MarketTicker(
                    ticker="KXINXU-TAIL-LOW",
                    series_ticker="KXINXU",
                    yes_bid_dollars=Decimal("0.00"),
                    yes_ask_dollars=Decimal("0.01"),
                    close_time=now + timedelta(hours=1),
                ),
            ]
        if series_ticker == "KXNASDAQ100U":
            return []
        if series_ticker == "KXNASDAQDUD":
            return [
                MarketTicker(
                    ticker="KXNASDAQDUD-ACTIVE",
                    series_ticker="KXNASDAQDUD",
                    yes_bid_dollars=Decimal("0.47"),
                    yes_ask_dollars=Decimal("0.50"),
                    close_time=now + timedelta(hours=2),
                )
            ]
        return []

    client = KalshiRestClient(base_url="https://external-api.demo.kalshi.co/trade-api/v2")
    with (
        patch.object(client, "get_markets", side_effect=fake_get_markets),
        patch("kalshi_poc.client.asyncio.sleep", AsyncMock(return_value=None)),
    ):
        selected = await client.discover_active_markets(
            ["KXBTC15M", "KXSOL15M", "KXINXU", "KXNASDAQ100U"],
            max_tickers=6,
            environment="production",
            now_utc=now,
        )
    await client.close()

    assert [m.ticker for m in selected] == [
        "KXBTCD-ATM-HOURLY",
        "KXSOL15M-OK-130S",
        "KXNASDAQDUD-ACTIVE",
    ]
