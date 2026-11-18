"""H8/H9/M2: single-application fill accounting, fill side/action mapping, reaper scoping."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

from kalshi_poc.client import KalshiRestClient, KalshiWebSocketClient
from kalshi_poc.config import ConfigManager
from kalshi_poc.main import TradingEngine
from kalshi_poc.models import (
    CLIENT_ORDER_ID_PREFIX,
    OrderSide,
    Position,
    TradeIntent,
    V2CreateOrderRequest,
)
from kalshi_poc.telemetry import TelemetryManager


async def _engine(tmp_path: Path, mode: str = "demo") -> tuple[TradingEngine, TelemetryManager]:
    cfg = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg.update_and_persist(
        {
            "kalshi_environment": "demo",
            "execution_mode": mode,
            "initial_capital": "100.00",
            "equity_floor": "10.00",
        }
    )
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    engine = TradingEngine(cfg, telem)
    engine.ws_client = KalshiWebSocketClient(ws_url="wss://example.invalid")
    engine.rest_client = KalshiRestClient(
        base_url="https://external-api.demo.kalshi.co/trade-api/v2", signer=AsyncMock()
    )
    return engine, telem


def _fill_msg(trade_id: str, order_id: str, count: str, side: str = "yes", action: str = "buy",
              yes_price: str = "0.40") -> dict:
    return {
        "trade_id": trade_id,
        "order_id": order_id,
        "market_ticker": "KXBTCD-T",
        "side": side,
        "action": action,
        "count_fp": count,
        "yes_price_dollars": yes_price,
        "is_taker": True,
    }


def test_client_order_ids_carry_engine_prefix() -> None:
    intent = TradeIntent(
        timestamp_ms=0, ticker="KXBTCD-T", side=OrderSide.BID, count=Decimal("1"),
        limit_price=Decimal("0.40"), expected_edge_dollars=Decimal("0.05"),
        confidence=Decimal("0.5"), inference_latency_ms=0, rationale="x", provider="t",
    )
    assert V2CreateOrderRequest.from_intent(intent).client_order_id.startswith(CLIENT_ORDER_ID_PREFIX)
    assert V2CreateOrderRequest(ticker="X", side="bid", count="1", price="0.5").client_order_id.startswith(
        CLIENT_ORDER_ID_PREFIX
    )


async def test_rest_then_ws_fill_for_same_order_applies_once(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    try:
        with patch.object(engine, "sync_balance_from_api", AsyncMock()), patch.object(
            engine.rest_client, "upgrade_api_tier_once", AsyncMock()
        ):
            applied = engine._apply_order_fill(
                "ord-1", "KXBTCD-T", OrderSide.BID, Decimal("10"), Decimal("0.40"),
                source="rest", fee_total=Decimal("0.17"), ts_ms=0,
            )
            assert applied == Decimal("10")
            await engine._on_fill(_fill_msg("tr-1", "ord-1", "6"))
            await engine._on_fill(_fill_msg("tr-2", "ord-1", "4"))
        pos = engine.portfolio.positions["KXBTCD-T"]
        assert pos.net_contracts == Decimal("10")
        assert engine.portfolio.cash_balance_dollars == Decimal("95.83")  # 100 - 4.00 - 0.17
    finally:
        await engine.rest_client.close()
        await telem.close()


async def test_ws_then_rest_fill_and_duplicate_trade_ids_apply_once(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    try:
        with patch.object(engine, "sync_balance_from_api", AsyncMock()), patch.object(
            engine.rest_client, "upgrade_api_tier_once", AsyncMock()
        ):
            await engine._on_fill(_fill_msg("tr-1", "ord-2", "4"))
            await engine._on_fill(_fill_msg("tr-1", "ord-2", "4"))  # duplicate delivery
            assert engine.portfolio.positions["KXBTCD-T"].net_contracts == Decimal("4")
            # REST response arrives later reporting cumulative fill of 10 -> only 6 more applied
            applied = engine._apply_order_fill(
                "ord-2", "KXBTCD-T", OrderSide.BID, Decimal("10"), Decimal("0.40"),
                source="rest", fee_total=None, ts_ms=0,
            )
            assert applied == Decimal("6")
        assert engine.portfolio.positions["KXBTCD-T"].net_contracts == Decimal("10")
    finally:
        await engine.rest_client.close()
        await telem.close()


async def test_fill_side_and_action_mapping(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    engine.portfolio.positions["KXBTCD-T"] = Position(
        ticker="KXBTCD-T", net_contracts=Decimal("5"),
        avg_entry_price=Decimal("0.40"), cost_basis_dollars=Decimal("2.00"),
    )
    try:
        with patch.object(engine, "sync_balance_from_api", AsyncMock()), patch.object(
            engine.rest_client, "upgrade_api_tier_once", AsyncMock()
        ):
            # Selling YES reduces a long YES position (previously mis-mapped to a buy)
            await engine._on_fill(_fill_msg("tr-s", "ord-s", "2", side="yes", action="sell", yes_price="0.50"))
            assert engine.portfolio.positions["KXBTCD-T"].net_contracts == Decimal("3")
            # Buying NO is equivalent to selling YES
            await engine._on_fill(_fill_msg("tr-n", "ord-n", "1", side="no", action="buy", yes_price="0.50"))
            assert engine.portfolio.positions["KXBTCD-T"].net_contracts == Decimal("2")
            # Selling NO is equivalent to buying YES
            await engine._on_fill(_fill_msg("tr-ns", "ord-ns", "1", side="no", action="sell", yes_price="0.50"))
            assert engine.portfolio.positions["KXBTCD-T"].net_contracts == Decimal("3")
    finally:
        await engine.rest_client.close()
        await telem.close()


async def test_shadow_mode_ignores_exchange_fills(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path, mode="shadow")
    try:
        with patch.object(engine, "sync_balance_from_api", AsyncMock()) as sync_mock:
            await engine._on_fill(_fill_msg("tr-x", "ord-x", "5"))
        assert engine.portfolio.positions == {}
        assert engine.portfolio.cash_balance_dollars == Decimal("100.00")
        sync_mock.assert_not_awaited()
    finally:
        await engine.rest_client.close()
        await telem.close()


async def test_order_response_without_fill_count_is_not_recorded_as_filled(tmp_path: Path) -> None:
    from kalshi_poc.models import OrderbookSnapshot, PriceLevel

    engine, telem = await _engine(tmp_path)
    snap = OrderbookSnapshot(
        ticker="KXBTCD-T", timestamp_ms=0,
        yes_bids=[PriceLevel(price=Decimal("0.47"), count=Decimal("50"))],
        no_bids=[PriceLevel(price=Decimal("0.52"), count=Decimal("50"))],
    )
    intent = TradeIntent(
        timestamp_ms=0, ticker="KXBTCD-T", side=OrderSide.BID, count=Decimal("4"),
        limit_price=Decimal("0.48"), expected_edge_dollars=Decimal("0.40"),
        expected_edge_per_contract=Decimal("0.10"), confidence=Decimal("0.8"),
        inference_latency_ms=1, rationale="x", provider="t",
    )
    q = telem.subscribe()
    try:
        with (
            patch.object(engine.rest_client, "create_order",
                         AsyncMock(return_value={"order": {"order_id": "o-unk", "status": "canceled"}})),
            patch.object(engine.rest_client, "upgrade_api_tier_once", AsyncMock()),
            patch.object(engine, "_ensure_shard_funded", AsyncMock()),
        ):
            await engine._evaluate_and_execute_intent(intent, snap, engine.config_manager.settings)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        rec = [e for e in events if e["type"] == "EXECUTION"][-1]["data"]
        assert rec["status"] == "canceled"
        assert rec["count"] == "0"
        assert "KXBTCD-T" not in engine.portfolio.positions
        assert "o-unk" in engine._own_order_ids
    finally:
        await engine.rest_client.close()
        await telem.close()


async def test_reaper_only_cancels_engine_orders(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    now = datetime.now(timezone.utc)
    old = (now - timedelta(seconds=60)).isoformat()
    engine._own_order_ids.add("own-tracked")
    resting = [
        {"order_id": "own-prefixed", "client_order_id": f"{CLIENT_ORDER_ID_PREFIX}abc",
         "ticker": "KXBTCD-A", "remaining_count_fp": "1", "price_dollars": "0.40", "created_time": old},
        {"order_id": "own-tracked", "client_order_id": "legacy-uuid",
         "ticker": "KXBTCD-B", "remaining_count_fp": "1", "price_dollars": "0.40", "created_time": old},
        {"order_id": "manual-1", "client_order_id": "user-placed",
         "ticker": "KXBTCD-C", "remaining_count_fp": "2", "price_dollars": "0.45", "created_time": old},
    ]
    try:
        with (
            patch.object(engine.rest_client, "get_resting_orders", AsyncMock(return_value=resting)),
            patch.object(engine.rest_client, "cancel_order", AsyncMock(return_value={})) as cancel_mock,
        ):
            await engine._reap_stale_resting_orders(now_utc=now)
        canceled = sorted(c.args[0] for c in cancel_mock.await_args_list)
        assert canceled == ["own-prefixed", "own-tracked"]
        # Manual order is untouched but still counted as resting exposure
        assert engine.portfolio.resting_order_tickers == {"KXBTCD-C"}
        assert engine.portfolio.resting_order_collateral_dollars == Decimal("0.90")
    finally:
        await engine.rest_client.close()
        await telem.close()
