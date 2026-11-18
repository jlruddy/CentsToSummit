"""C4/H6: exits keep running after kill/pause, Emergency Flatten closes positions, persisted state."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from kalshi_poc.client import KalshiWebSocketClient
from kalshi_poc.config import ConfigManager
from kalshi_poc.main import TradingEngine
from kalshi_poc.models import ExecutionMode, OrderbookSnapshot, Position, PriceLevel
from kalshi_poc.telemetry import TelemetryManager

TICKER = "KXBTC15M-26NOV171730-30"


async def _shadow_engine(tmp_path: Path) -> tuple[TradingEngine, TelemetryManager]:
    cfg = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg.update_and_persist(
        {
            "execution_mode": "shadow",
            "initial_capital": "100.00",
            "equity_floor": "10.00",
            "allowed_series_prefixes": ["KXBTC15M"],
        }
    )
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    engine = TradingEngine(cfg, telem)
    engine.ws_client = KalshiWebSocketClient(ws_url="wss://example.invalid")
    return engine, telem


def _hold_long(engine: TradingEngine, bid: str = "0.50", no_bid: str = "0.48") -> None:
    engine.portfolio.positions[TICKER] = Position(
        ticker=TICKER,
        net_contracts=Decimal("4"),
        avg_entry_price=Decimal("0.55"),
        cost_basis_dollars=Decimal("2.20"),
    )
    engine.portfolio.cash_balance_dollars = Decimal("97.80")
    engine.ws_client.set_snapshot(
        OrderbookSnapshot(
            ticker=TICKER,
            timestamp_ms=0,
            yes_bids=[PriceLevel(price=Decimal(bid), count=Decimal("20"))],
            no_bids=[PriceLevel(price=Decimal(no_bid), count=Decimal("20"))],
        )
    )


async def test_emergency_kill_flattens_shadow_positions(tmp_path: Path) -> None:
    engine, telem = await _shadow_engine(tmp_path)
    _hold_long(engine)
    try:
        res = await engine.emergency_kill()
        assert engine.portfolio.kill_switch_tripped is True
        assert engine.config_manager.settings.execution_mode == ExecutionMode.PAUSED
        assert TICKER not in engine.portfolio.positions
        assert res["flatten"][0]["ticker"] == TICKER
        assert res["flatten"][0]["remaining_contracts"] == "0"
        # 97.80 + 4 * 0.50 - taker fee ceil(0.07*4*0.25)=0.07
        assert engine.portfolio.cash_balance_dollars == Decimal("99.73")
    finally:
        await telem.close()


async def test_exit_management_runs_after_kill_switch_and_pause(tmp_path: Path) -> None:
    engine, telem = await _shadow_engine(tmp_path)
    now = datetime(2026, 11, 17, 22, 10, 0, tzinfo=timezone.utc)
    engine._market_expiries[TICKER] = now + timedelta(seconds=600)
    try:
        await engine.pause_engine()
        engine.portfolio.kill_switch_tripped = True
        # Hard stop-loss: bid 0.32 is $0.23 below the 0.55 entry with > 180s left
        _hold_long(engine, bid="0.32", no_bid="0.66")
        submitted = await engine._run_exit_management(engine.config_manager.settings, now)
        assert len(submitted) == 1
        assert TICKER not in engine.portfolio.positions
    finally:
        await telem.close()


async def test_exit_management_inactive_when_paused_at_boot(tmp_path: Path) -> None:
    engine, telem = await _shadow_engine(tmp_path)
    engine.config_manager.update_and_persist({"execution_mode": "paused"})
    engine._exit_mode_when_paused = None
    now = datetime(2026, 11, 17, 22, 10, 0, tzinfo=timezone.utc)
    _hold_long(engine, bid="0.32", no_bid="0.66")
    try:
        assert await engine._run_exit_management(engine.config_manager.settings, now) == []
        assert TICKER in engine.portfolio.positions
    finally:
        await telem.close()


async def test_deterministic_exit_retries_are_throttled_per_ticker(tmp_path: Path) -> None:
    engine, telem = await _shadow_engine(tmp_path)
    now = datetime(2026, 11, 17, 22, 10, 0, tzinfo=timezone.utc)
    engine._market_expiries[TICKER] = now + timedelta(seconds=600)
    # Stop-loss triggers but the YES bid has no depth at the exit price -> IOC cannot fill
    _hold_long(engine, bid="0.32", no_bid="0.66")
    engine.ws_client.set_snapshot(
        OrderbookSnapshot(
            ticker=TICKER,
            timestamp_ms=0,
            yes_bids=[PriceLevel(price=Decimal("0.32"), count=Decimal("1"))],
            no_bids=[PriceLevel(price=Decimal("0.66"), count=Decimal("20"))],
        )
    )
    try:
        first = await engine._evaluate_deterministic_exits(engine.config_manager.settings, now)
        second = await engine._evaluate_deterministic_exits(engine.config_manager.settings, now)
        assert len(first) == 1
        assert second == []
    finally:
        await telem.close()


async def test_kill_switch_state_persists_across_engine_restart(tmp_path: Path) -> None:
    engine, telem = await _shadow_engine(tmp_path)
    try:
        await engine.emergency_kill()
        fresh = TradingEngine(engine.config_manager, telem)
        await fresh._load_persisted_state()
        assert fresh.portfolio.kill_switch_tripped is True
        assert fresh.risk_gate.kill_switch_tripped is True

        # Explicit operator Start clears it (and the cleared state is persisted)
        await fresh.start_engine()
        again = TradingEngine(engine.config_manager, telem)
        await again._load_persisted_state()
        assert again.portfolio.kill_switch_tripped is False
    finally:
        if fresh._task is not None:
            await fresh.stop()
        await telem.close()


async def test_daily_baseline_survives_restart_and_rolls_at_utc_midnight(tmp_path: Path) -> None:
    engine, telem = await _shadow_engine(tmp_path)
    day1 = datetime(2026, 11, 17, 22, 0, 0, tzinfo=timezone.utc)
    try:
        await engine._ensure_daily_baseline(engine.config_manager.settings, day1)
        assert engine.portfolio.daily_start_equity_dollars == Decimal("100.00")

        # Restart same UTC day after a $30 loss: baseline must stay at $100 so the loss still counts
        fresh = TradingEngine(engine.config_manager, telem)
        fresh.portfolio.cash_balance_dollars = Decimal("70.00")
        await fresh._ensure_daily_baseline(engine.config_manager.settings, day1 + timedelta(hours=1))
        assert fresh.portfolio.daily_start_equity_dollars == Decimal("100.00")
        assert fresh.portfolio.daily_pnl_dollars == Decimal("-30.00")

        # Next UTC day: baseline rolls to current equity
        await fresh._ensure_daily_baseline(engine.config_manager.settings, day1 + timedelta(hours=3))
        assert fresh.portfolio.daily_start_equity_dollars == Decimal("70.00")
    finally:
        await telem.close()


# --- M3/M2: strategy state survives config saves; shadow isolated from the exchange account ---


async def test_config_save_keeps_strategy_state_unless_provider_changes(tmp_path: Path) -> None:
    engine, telem = await _shadow_engine(tmp_path)
    try:
        strat = engine.strategy
        strat._mid_history["KXBTC15M-X"] = "sentinel"  # type: ignore[assignment]
        engine.config_manager.update_and_persist({"max_slippage_cents": "2"})
        await engine.pause_engine()
        assert engine.strategy is strat
        assert engine.strategy.settings.max_slippage_cents == Decimal("2")
        assert engine.strategy._mid_history["KXBTC15M-X"] == "sentinel"

        engine.config_manager.update_and_persist({"strategy_provider": "ollama_local"})
        assert engine.strategy is not strat
        assert engine.strategy.provider_name == "ollama_local"
    finally:
        await telem.close()


async def test_shadow_scope_never_syncs_exchange_balance_or_positions(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, patch

    from kalshi_poc.client import KalshiRestClient

    engine, telem = await _shadow_engine(tmp_path)
    engine.rest_client = KalshiRestClient(
        base_url="https://external-api.demo.kalshi.co/trade-api/v2", signer=AsyncMock()
    )
    try:
        with (
            patch.object(engine.rest_client, "get_balance", AsyncMock()) as bal_mock,
            patch.object(engine.rest_client, "get_open_positions", AsyncMock(return_value=[])),
        ):
            await engine.sync_balance_from_api()
            reset = await engine.reset_shadow()
        bal_mock.assert_not_awaited()
        assert engine.portfolio.cash_balance_dollars == Decimal("100.00")
        assert reset["cash_balance_dollars"] == "100.00"
    finally:
        await engine.rest_client.close()
        await telem.close()


async def test_first_exchange_sync_sets_baseline_from_real_equity(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, patch

    from kalshi_poc.client import KalshiRestClient
    from kalshi_poc.models import BalanceResponse

    engine, telem = await _shadow_engine(tmp_path)
    engine.config_manager.update_and_persist({"execution_mode": "demo"})
    engine.rest_client = KalshiRestClient(
        base_url="https://external-api.demo.kalshi.co/trade-api/v2", signer=AsyncMock()
    )
    try:
        with (
            patch.object(engine.rest_client, "get_balance",
                         AsyncMock(return_value=(BalanceResponse(balance_dollars=Decimal("40.00")), 1.0))),
            patch.object(engine.rest_client, "get_open_positions", AsyncMock(return_value=[])),
            patch.object(engine.rest_client, "get_resting_orders", AsyncMock(return_value=[])),
        ):
            await engine.sync_balance_from_api()
        # Real balance $40 vs configured $100: no phantom -$60 "daily loss"
        assert engine.portfolio.daily_start_equity_dollars == Decimal("40.00")
        assert engine.portfolio.daily_pnl_dollars == Decimal("0.00")
    finally:
        await engine.rest_client.close()
        await telem.close()
