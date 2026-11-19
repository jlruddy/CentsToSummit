"""Execution guards: slippage cap vs signal reference, in-flight duplicate guard, fee parsing,
post-inference edge re-check, and session-scoped activity feed."""

import asyncio
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from kalshi_poc.client import KalshiWebSocketClient
from kalshi_poc.config import AppSettings, ConfigManager
from kalshi_poc.main import TradingEngine, order_fee_total
from kalshi_poc.models import (
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    PriceLevel,
    RiskEvaluation,
    TradeIntent,
    V2CreateOrderRequest,
)
from kalshi_poc.risk_gate import RiskGate
from kalshi_poc.telemetry import TelemetryManager

TICKER = "KXBTC15M-26NOV181715-15"
MAX_SLIP = Decimal("0.015")


def _settings(**kw) -> AppSettings:
    base = dict(
        _env_file=None,
        max_slippage_cents=Decimal("1.5"),
        min_net_edge_dollars=Decimal("0.015"),
        max_position_dollars_per_market=Decimal("10"),
        allowed_series_prefixes=["KXBTC15M"],
        initial_capital=Decimal("100"),
        equity_floor=Decimal("10"),
        dynamic_sizing_enabled=False,
    )
    base.update(kw)
    return AppSettings(**base)


def _book(bid: str, asks: list[tuple[str, int]]) -> OrderbookSnapshot:
    """YES bid at `bid`; YES asks given as (price, size), derived from NO bids."""
    no_bids = sorted(
        (PriceLevel(price=Decimal("1.00") - Decimal(p), count=Decimal(c)) for p, c in asks),
        key=lambda lvl: lvl.price,
    )
    return OrderbookSnapshot(
        ticker=TICKER,
        timestamp_ms=1,
        yes_bids=[PriceLevel(price=Decimal(bid), count=Decimal("100"))],
        no_bids=no_bids,
    )


def _intent(limit: str, ref: str | None, edge_per_c: str = "0.20", count: int = 10, **kw) -> TradeIntent:
    c = Decimal(count)
    e = Decimal(edge_per_c)
    base = dict(
        timestamp_ms=1,
        ticker=TICKER,
        side=OrderSide.BID,
        count=c,
        limit_price=Decimal(limit),
        reference_price=Decimal(ref) if ref is not None else None,
        expected_edge_dollars=e * c,
        expected_edge_per_contract=e,
        confidence=Decimal("0.8"),
        inference_latency_ms=1,
        rationale="t",
        provider="test",
    )
    base.update(kw)
    return TradeIntent(**base)


def _portfolio() -> PortfolioState:
    return PortfolioState(
        cash_balance_dollars=Decimal("100"),
        starting_capital_dollars=Decimal("100"),
        daily_start_equity_dollars=Decimal("100"),
    )


# --- 1. Slippage ----------------------------------------------------------------------------


def test_book_walk_slippage_over_1_5c_is_rejected() -> None:
    # bid 0.49 / asks 0.50x2, 0.53x8 -> mid 0.495, 10-lot VWAP 0.524 -> 2.9c per contract
    snap = _book("0.49", [("0.50", 2), ("0.53", 8)])
    res = RiskGate(_settings()).evaluate(_intent("0.53", "0.53"), snap, _portfolio())
    assert res.approved is False
    assert any("Estimated slippage" in r for r in res.rejection_reasons)


def test_book_walk_within_1_5c_is_approved() -> None:
    snap = _book("0.49", [("0.50", 50)])
    res = RiskGate(_settings()).evaluate(_intent("0.50", "0.50"), snap, _portfolio())
    assert res.approved is True, res.rejection_reasons


def test_gate_rejects_limit_more_than_max_slip_past_reference() -> None:
    # Pre-fix ratchet outcome: signal 0.70, limit re-clamped to 0.73 against a fresh ask
    snap = _book("0.72", [("0.73", 50)])
    res = RiskGate(_settings()).evaluate(_intent("0.73", "0.70"), snap, _portfolio())
    assert res.approved is False
    assert any("past reference" in r for r in res.rejection_reasons)


async def _engine(tmp_path: Path, mode: str = "shadow") -> tuple[TradingEngine, TelemetryManager]:
    cfg = ConfigManager(
        env_path=tmp_path / ".env",
        config_json_path=tmp_path / "config.json",
        key_dir=tmp_path / "keys",
    )
    cfg.update_and_persist(
        {
            "execution_mode": mode,
            "initial_capital": "100.00",
            "equity_floor": "10.00",
            "allowed_series_prefixes": ["KXBTC15M"],
            "max_slippage_cents": "1.5",
            "min_net_edge_dollars": "0.015",
            "max_position_dollars_per_market": "10",
        }
    )
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    engine = TradingEngine(cfg, telem)
    engine.ws_client = KalshiWebSocketClient(ws_url="wss://example.invalid")
    return engine, telem


async def test_outgoing_limit_never_exceeds_reference_plus_max_slip(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    seen: list[TradeIntent] = []

    def capture(intent, snap, portfolio, **kw):
        seen.append(intent)
        return RiskEvaluation(
            intent_id=intent.intent_id, timestamp_ms=1, ticker=intent.ticker, approved=False,
            rejection_reasons=["capture"], estimated_fill_price=intent.limit_price,
            estimated_slippage_dollars=Decimal("0"), estimated_entry_fee_dollars=Decimal("0"),
            estimated_round_trip_fee_dollars=Decimal("0"), worst_case_equity_after=Decimal("0"),
        )

    try:
        with patch.object(engine.risk_gate, "evaluate", side_effect=capture):
            cases = 0
            for ref_c in range(400, 601, 5):  # signal refs 0.400..0.600 incl. half-cents
                ref = Decimal(ref_c) / Decimal("1000")
                for d in range(-4, 5):  # live ask/bid from ref-4c to ref+4c
                    for start_limit in (ref, min(ref + MAX_SLIP, Decimal("0.99"))):
                        for side in (OrderSide.BID, OrderSide.ASK):
                            px = (ref + Decimal(d) / Decimal("100")).quantize(Decimal("0.01"))
                            snap = (
                                _book(str(px - Decimal("0.01")), [(str(px), 50)])
                                if side == OrderSide.BID
                                else _book(str(px), [(str(px + Decimal("0.01")), 50)])
                            )
                            await engine._evaluate_and_execute_intent(
                                _intent(str(start_limit), str(ref), side=side), snap,
                                engine.config_manager.settings,
                            )
                            out = seen[-1]
                            sent = Decimal(V2CreateOrderRequest.from_intent(out).price)
                            if side == OrderSide.BID:
                                assert sent <= ref + MAX_SLIP, (ref, px, start_limit, sent)
                            else:
                                assert sent >= ref - MAX_SLIP, (ref, px, start_limit, sent)
                            cases += 1
            assert cases > 1000
    finally:
        await telem.close()


async def test_post_inference_ask_drift_no_longer_ratchets(tmp_path: Path) -> None:
    """Signal 0.70; strategy clamped to 0.715 on the pre-LLM book; live ask is now 0.73."""
    engine, telem = await _engine(tmp_path)
    try:
        snap = _book("0.72", [("0.73", 50)])
        await engine._evaluate_and_execute_intent(
            _intent("0.715", "0.70"), snap, engine.config_manager.settings
        )
        assert TICKER not in engine.portfolio.positions
        async with telem._db.execute(
            "SELECT approved, rejection_reasons_json FROM risk_evaluations"
        ) as cur:
            approved, reasons = await cur.fetchone()
        assert approved == 0
        assert "past reference" in reasons
    finally:
        await telem.close()


# --- 2. Duplicate entries -------------------------------------------------------------------


def test_gate_blocks_ticker_with_order_in_flight() -> None:
    p = _portfolio()
    p.pending_order_tickers.add(TICKER)
    snap = _book("0.49", [("0.50", 50)])
    res = RiskGate(_settings()).evaluate(_intent("0.50", "0.50"), snap, p)
    assert res.approved is False
    assert any("in flight" in r for r in res.rejection_reasons)


async def test_concurrent_evaluations_cannot_double_enter(tmp_path: Path) -> None:
    engine, telem = await _engine(tmp_path)
    real_log = telem.log_risk_evaluation

    async def slow_log(ev):  # widen the await window between gate and fill
        await asyncio.sleep(0.05)
        await real_log(ev)

    try:
        snap = _book("0.49", [("0.50", 50)])
        s = engine.config_manager.settings
        with patch.object(telem, "log_risk_evaluation", side_effect=slow_log):
            await asyncio.gather(
                engine._evaluate_and_execute_intent(_intent("0.50", "0.50"), snap, s),
                engine._evaluate_and_execute_intent(_intent("0.50", "0.50"), snap, s),
            )
        assert engine.portfolio.positions[TICKER].net_contracts == Decimal("10")
        assert engine.portfolio.pending_order_tickers == set()
    finally:
        await telem.close()


# --- 3. Fees and net edge -------------------------------------------------------------------


def test_fee_average_is_per_contract() -> None:
    # Live 13-lot @0.72 logged "Fee $0.01": the per-contract average, not the order total
    fee = order_fee_total({"average_fee_paid_dollars": "0.0146"}, Decimal("13"), Decimal("0.72"))
    assert fee == Decimal("0.19")


def test_fee_prefers_order_total_fields() -> None:
    obj = {"taker_fees_dollars": "0.19", "maker_fees_dollars": "0.00", "average_fee_paid": "0.01"}
    assert order_fee_total(obj, Decimal("13"), Decimal("0.72")) == Decimal("0.19")
    assert order_fee_total({"taker_fees": 19}, Decimal("13"), Decimal("0.72")) == Decimal("0.19")


def test_fee_falls_back_to_schedule_and_zero_without_fill() -> None:
    # ceil(0.07 * 13 * 0.72 * 0.28) = ceil(0.1835) = 0.19
    assert order_fee_total({}, Decimal("13"), Decimal("0.72")) == Decimal("0.19")
    assert order_fee_total({"taker_fees": 19}, Decimal("0"), Decimal("0.72")) == Decimal("0.00")


def test_net_edge_is_reduced_by_drift_from_reference() -> None:
    # Hold-to-expiry 15M contract: entry fee only, ceil(0.07*10*0.51*0.49)=0.18 -> $0.018/c
    snap = _book("0.50", [("0.51", 50)])
    gate = RiskGate(_settings())
    at_ref = gate.evaluate(
        _intent("0.51", "0.51", edge_per_c="0.04"), snap, _portfolio(), seconds_to_expiry=600
    )
    assert at_ref.approved is True, at_ref.rejection_reasons
    # Same book, but the signal priced its 4c edge at 0.50: paying 0.51 leaves < 1.5c net
    drifted = gate.evaluate(
        _intent("0.51", "0.50", edge_per_c="0.04"), snap, _portfolio(), seconds_to_expiry=600
    )
    assert drifted.approved is False
    assert any("Net edge" in r for r in drifted.rejection_reasons)


# --- 4. Session-scoped activity feed --------------------------------------------------------


async def test_recent_activity_excludes_prior_sessions(tmp_path: Path) -> None:
    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    try:
        snap = _book("0.49", [("0.50", 50)])
        stale = RiskGate(_settings()).evaluate(_intent("0.50", "0.50"), snap, _portfolio())
        await telem.log_risk_evaluation(stale)
        assert len(await telem.get_recent_activity()) == 1

        telem.new_session()  # what a restart / shadow reset does
        assert await telem.get_recent_activity() == []

        fresh = RiskGate(_settings()).evaluate(_intent("0.50", "0.50"), snap, _portfolio())
        await telem.log_risk_evaluation(fresh)
        events = await telem.get_recent_activity()
        assert [e["data"]["intent_id"] for e in events] == [fresh.intent_id]
    finally:
        await telem.close()
