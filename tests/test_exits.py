"""Engine-side trailing stop, configurable TP/SL, exit slippage limit, reduce_only exits."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from kalshi_poc.client import KalshiWebSocketClient
from kalshi_poc.config import AppSettings, ConfigManager
from kalshi_poc.main import TradingEngine
from kalshi_poc.models import (
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    Position,
    PriceLevel,
    TradeIntent,
    V2CreateOrderRequest,
)
from kalshi_poc.risk_gate import RiskGate
from kalshi_poc.telemetry import TelemetryManager

TICKER = "KXBTC15M-26NOV171730-30"
NOW = datetime(2026, 11, 17, 22, 10, 0, tzinfo=timezone.utc)


async def _engine(tmp_path: Path, **overrides) -> tuple[TradingEngine, TelemetryManager]:
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
            **overrides,
        }
    )
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    engine = TradingEngine(cfg, telem)
    engine.ws_client = KalshiWebSocketClient(ws_url="wss://example.invalid")
    engine._market_expiries[TICKER] = NOW + timedelta(seconds=600)
    return engine, telem


def _book(engine: TradingEngine, bid: str, ask: str, depth: str = "20") -> None:
    engine.ws_client.set_snapshot(
        OrderbookSnapshot(
            ticker=TICKER,
            timestamp_ms=0,
            yes_bids=[PriceLevel(price=Decimal(bid), count=Decimal(depth))],
            no_bids=[PriceLevel(price=Decimal("1.00") - Decimal(ask), count=Decimal(depth))],
        )
    )


def _long(engine: TradingEngine, entry: str = "0.50", qty: str = "4") -> None:
    engine.portfolio.positions[TICKER] = Position(
        ticker=TICKER,
        net_contracts=Decimal(qty),
        avg_entry_price=Decimal(entry),
        cost_basis_dollars=(Decimal(qty) * Decimal(entry)),
    )


def _short(engine: TradingEngine, entry_yes: str = "0.50", qty: str = "4") -> None:
    engine.portfolio.positions[TICKER] = Position(
        ticker=TICKER,
        net_contracts=-Decimal(qty),
        avg_entry_price=Decimal(entry_yes),
        cost_basis_dollars=(Decimal(qty) * (Decimal("1") - Decimal(entry_yes))),
    )


async def _tick(engine: TradingEngine) -> list[TradeIntent]:
    engine._last_exit_attempt_mono.clear()  # isolate trigger logic from the retry throttle
    return await engine._evaluate_deterministic_exits(engine.config_manager.settings, NOW)


async def test_trailing_stop_locks_in_spike_after_pullback(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)  # defaults: activate +$0.08, trail $0.06
    _long(engine, entry="0.50")
    try:
        for bid, ask in (("0.55", "0.56"), ("0.62", "0.63"), ("0.66", "0.67"), ("0.61", "0.62")):
            _book(engine, bid, ask)
            assert await _tick(engine) == [], bid  # 0.61 is only $0.05 off the 0.66 peak
        _book(engine, "0.60", "0.61")  # $0.06 below peak -> trailing stop
        exits = await _tick(engine)
        assert len(exits) == 1 and exits[0].side == OrderSide.ASK
        assert "TRAILING_STOP" in exits[0].rationale
        assert exits[0].reduce_only is True
        assert TICKER not in engine.portfolio.positions
    finally:
        await telem.close()


async def test_trailing_stop_inactive_until_activation_gain(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    _long(engine, entry="0.50")
    try:
        for bid in ("0.57", "0.51", "0.46"):  # peak +$0.07 < $0.08 activation; drop < stop-loss
            _book(engine, bid, str(Decimal(bid) + Decimal("0.01")))
            assert await _tick(engine) == [], bid
        assert TICKER in engine.portfolio.positions
    finally:
        await telem.close()


async def test_trailing_stop_on_no_position_mirrors_long(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    _short(engine, entry_yes="0.50")  # long NO bought at $0.50
    try:
        _book(engine, "0.39", "0.40")  # NO bid = 0.60 -> +$0.10, trailing active, peak 0.60
        assert await _tick(engine) == []
        _book(engine, "0.45", "0.46")  # NO bid = 0.54 -> $0.06 off peak
        exits = await _tick(engine)
        assert len(exits) == 1 and exits[0].side == OrderSide.BID
        assert "TRAILING_STOP" in exits[0].rationale
    finally:
        await telem.close()


async def test_trailing_peak_resets_for_a_new_position(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    _long(engine, entry="0.50")
    try:
        _book(engine, "0.66", "0.67")
        await _tick(engine)
        engine.portfolio.positions.pop(TICKER)
        await _tick(engine)  # flat: peak state is discarded
        _long(engine, entry="0.62")
        _book(engine, "0.61", "0.62")  # would be $0.05 off the old 0.66 peak; new position is not active
        assert await _tick(engine) == []
    finally:
        await telem.close()


async def test_trailing_stop_can_be_disabled(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path, trailing_stop_enabled=False)
    _long(engine, entry="0.50")
    try:
        _book(engine, "0.66", "0.67")
        await _tick(engine)
        _book(engine, "0.58", "0.59")
        assert await _tick(engine) == []
    finally:
        await telem.close()


async def test_take_profit_and_stop_loss_thresholds_are_configurable(tmp_path: Path) -> None:
    engine, telem = await _engine(
        tmp_path,
        take_profit_min_gain="0.10",
        take_profit_min_price="0.60",
        trailing_stop_enabled=False,
    )
    _long(engine, entry="0.50")
    try:
        _book(engine, "0.60", "0.61")
        exits = await _tick(engine)
        assert len(exits) == 1 and "TAKE_PROFIT_LOCK" in exits[0].rationale
    finally:
        await telem.close()

    engine2, telem2 = await _engine(tmp_path / "b", stop_loss_drop="0.10", trailing_stop_enabled=False)
    _long(engine2, entry="0.50")
    try:
        _book(engine2, "0.40", "0.41")
        exits = await _tick(engine2)
        assert len(exits) == 1 and "HARD_STOP_LOSS" in exits[0].rationale
    finally:
        await telem2.close()


async def test_wide_spread_no_longer_blocks_exits(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    _long(engine, entry="0.50")
    try:
        _book(engine, "0.66", "0.67")
        await _tick(engine)
        _book(engine, "0.60", "0.70")  # 10c spread: 5c from mid > 3c entry limit, < 6c exit limit
        exits = await _tick(engine)
        assert len(exits) == 1
        assert TICKER not in engine.portfolio.positions
    finally:
        await telem.close()


def test_gate_uses_exit_slippage_limit_only_for_reducing_exits() -> None:
    gate = RiskGate(AppSettings(_env_file=None, allowed_series_prefixes=["KXBTC15M"]))
    p = PortfolioState(
        cash_balance_dollars=Decimal("100"),
        starting_capital_dollars=Decimal("100"),
        daily_start_equity_dollars=Decimal("100"),
    )
    p.positions[TICKER] = Position(
        ticker=TICKER, net_contracts=Decimal("4"),
        avg_entry_price=Decimal("0.50"), cost_basis_dollars=Decimal("2.00"),
    )
    wide = OrderbookSnapshot(
        ticker=TICKER, timestamp_ms=0,
        yes_bids=[PriceLevel(price=Decimal("0.60"), count=Decimal("20"))],
        no_bids=[PriceLevel(price=Decimal("0.30"), count=Decimal("20"))],  # ask 0.70
    )
    common = dict(timestamp_ms=0, ticker=TICKER, confidence=Decimal("0.9"),
                  inference_latency_ms=0, rationale="x", provider="t")
    exit_intent = TradeIntent(side=OrderSide.ASK, count=Decimal("4"), limit_price=Decimal("0.60"),
                              expected_edge_dollars=Decimal("0"), **common)
    assert gate.evaluate(exit_intent, wide, p).approved is True

    # 14c spread: selling at the bid is 7c below mid, beyond the 6c exit limit -> rejected
    wider = wide.model_copy(update={"yes_bids": [PriceLevel(price=Decimal("0.56"), count=Decimal("20"))]})
    assert gate.evaluate(exit_intent.model_copy(update={"limit_price": Decimal("0.56")}), wider, p).approved is False


def test_exit_orders_are_reduce_only_entries_are_not() -> None:
    common = dict(timestamp_ms=0, ticker=TICKER, count=Decimal("2"), limit_price=Decimal("0.50"),
                  expected_edge_dollars=Decimal("0.10"), confidence=Decimal("0.5"),
                  inference_latency_ms=0, rationale="x", provider="t", side=OrderSide.ASK)
    exit_payload = V2CreateOrderRequest.from_intent(TradeIntent(reduce_only=True, **common)).model_dump(
        mode="json", exclude_none=True
    )
    entry_payload = V2CreateOrderRequest.from_intent(TradeIntent(**common)).model_dump(
        mode="json", exclude_none=True
    )
    assert exit_payload["reduce_only"] is True
    assert exit_payload["time_in_force"] == "immediate_or_cancel"
    assert "reduce_only" not in entry_payload
