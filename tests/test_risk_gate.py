"""Test equity floor, fee-edge rejection, allowlists, and kill-switches in RiskGate."""

from decimal import Decimal

from kalshi_poc.config import AppSettings
from kalshi_poc.models import (
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    Position,
    PriceLevel,
    TradeIntent,
)
from kalshi_poc.risk_gate import RiskGate


def _make_snapshot(ticker: str = "KXBTC-26NOV-T90000") -> OrderbookSnapshot:
    # YES bids ascending: 0.48 (50), 0.50 (50)
    # NO bids ascending: 0.45 (20), 0.48 (30) -> YES asks: 0.52 (30), 0.55 (20)
    return OrderbookSnapshot(
        ticker=ticker,
        timestamp_ms=1700000000000,
        yes_bids=[
            PriceLevel(price=Decimal("0.48"), count=Decimal("50")),
            PriceLevel(price=Decimal("0.50"), count=Decimal("50")),
        ],
        no_bids=[
            PriceLevel(price=Decimal("0.45"), count=Decimal("20")),
            PriceLevel(price=Decimal("0.48"), count=Decimal("30")),
        ],
    )


def _make_intent(**overrides) -> TradeIntent:
    base = dict(
        timestamp_ms=1700000000000,
        ticker="KXBTC-26NOV-T90000",
        side=OrderSide.BID,
        count=Decimal("10"),
        limit_price=Decimal("0.52"),
        expected_edge_dollars=Decimal("0.80"),  # $0.08/contract across 10 contracts
        confidence=Decimal("0.80"),
        inference_latency_ms=5,
        rationale="Test signal",
        provider="stub_momentum",
    )
    base.update(overrides)
    return TradeIntent(**base)


def _make_portfolio(cash: str = "5000.00") -> PortfolioState:
    c = Decimal(cash)
    return PortfolioState(
        cash_balance_dollars=c,
        starting_capital_dollars=c,
        daily_start_equity_dollars=c,
    )


def test_risk_gate_approves_valid_trade() -> None:
    gate = RiskGate(AppSettings(_env_file=None))
    eval_res = gate.evaluate(_make_intent(), _make_snapshot(), _make_portfolio())
    assert eval_res.approved is True
    assert eval_res.rejection_reasons == []


def test_risk_gate_rejects_non_allowlisted_series() -> None:
    gate = RiskGate(AppSettings(_env_file=None))
    intent = _make_intent(ticker="KXPOLITICS-26-PRES")
    snap = _make_snapshot(ticker="KXPOLITICS-26-PRES")
    eval_res = gate.evaluate(intent, snap, _make_portfolio())
    assert eval_res.approved is False
    assert any("tax-hygiene allowlist" in r for r in eval_res.rejection_reasons)


def test_risk_gate_rejects_equity_floor_breach() -> None:
    settings = AppSettings(
        _env_file=None,
        initial_capital=Decimal("2005.00"),
        equity_floor=Decimal("2000.00"),
    )
    gate = RiskGate(settings)
    # 20 contracts * 0.52 = $10.40 > $5.00 deployable budget
    intent = _make_intent(count=Decimal("20"), expected_edge_dollars=Decimal("2.00"))
    eval_res = gate.evaluate(intent, _make_snapshot(), _make_portfolio("2005.00"))
    assert eval_res.approved is False
    assert any("Equity floor breach" in r for r in eval_res.rejection_reasons)


def test_risk_gate_rejects_insufficient_fee_edge() -> None:
    gate = RiskGate(AppSettings(_env_file=None))
    # 10 contracts @ 0.52 -> taker fee = ceil(0.07 * 10 * 0.52 * 0.48) = ceil(0.17472) = $0.18
    # round-trip fee = $0.36 >= expected_edge_dollars ($0.30)
    intent = _make_intent(expected_edge_dollars=Decimal("0.30"))
    eval_res = gate.evaluate(intent, _make_snapshot(), _make_portfolio())
    assert eval_res.approved is False
    assert any("Fee + slippage" in r for r in eval_res.rejection_reasons)


def test_risk_gate_rejects_thin_book_depth() -> None:
    gate = RiskGate(AppSettings(_env_file=None))
    # Total YES ask depth is 30 + 20 = 50; requesting 60 should reject
    intent = _make_intent(count=Decimal("60"), expected_edge_dollars=Decimal("5.00"))
    eval_res = gate.evaluate(intent, _make_snapshot(), _make_portfolio())
    assert eval_res.approved is False
    assert any("Insufficient book depth" in r for r in eval_res.rejection_reasons)


def test_risk_gate_kill_switch_and_drawdown() -> None:
    settings = AppSettings(_env_file=None, daily_drawdown_kill_switch=Decimal("100.00"))
    gate = RiskGate(settings)
    portfolio = PortfolioState(
        cash_balance_dollars=Decimal("4850.00"),
        starting_capital_dollars=Decimal("5000.00"),
        daily_start_equity_dollars=Decimal("5000.00"),
    )
    eval_res = gate.evaluate(_make_intent(), _make_snapshot(), portfolio)
    assert eval_res.approved is False
    assert portfolio.kill_switch_tripped is True
    assert any("drawdown kill-switch" in r for r in eval_res.rejection_reasons)


def test_risk_gate_rate_limit_and_max_open_markets() -> None:
    settings = AppSettings(_env_file=None, max_orders_per_minute=2, max_open_markets=1)
    gate = RiskGate(settings)
    now_ms = 1700000000000
    portfolio = _make_portfolio()
    portfolio.recent_order_timestamps_ms = [now_ms - 10_000, now_ms - 5_000]
    portfolio.positions["KXETH-26NOV-T3000"] = Position(
        ticker="KXETH-26NOV-T3000",
        net_contracts=Decimal("5"),
        avg_entry_price=Decimal("0.50"),
        cost_basis_dollars=Decimal("2.50"),
    )
    eval_res = gate.evaluate(_make_intent(), _make_snapshot(), portfolio, now_ms=now_ms)
    assert eval_res.approved is False
    assert any("rate limit" in r for r in eval_res.rejection_reasons)
    assert any("Max open markets" in r for r in eval_res.rejection_reasons)


def test_risk_gate_rejects_high_fee_to_notional_micro_trade() -> None:
    gate = RiskGate(AppSettings(_env_file=None))
    # Penny book: NO bid @ 0.95 -> YES ask @ 0.05 (50). 5 contracts = $0.25 notional, RT fee = $0.04 (16.0% > 15%)
    snap = OrderbookSnapshot(
        ticker="KXBTC-26NOV-T90000",
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.04"), count=Decimal("50"))],
        no_bids=[PriceLevel(price=Decimal("0.95"), count=Decimal("50"))],
    )
    intent = _make_intent(
        count=Decimal("5"),
        limit_price=Decimal("0.05"),
        expected_edge_dollars=Decimal("0.50"),
    )
    eval_res = gate.evaluate(intent, snap, _make_portfolio())
    assert eval_res.approved is False
    assert any("Fee-to-notional ratio" in r for r in eval_res.rejection_reasons)


def test_risk_gate_approves_intraday_hold_to_expiry_multi_contract_edge() -> None:
    settings = AppSettings(
        _env_file=None,
        allowed_series_prefixes=["KXBTCD", "KXBTC"],
    )
    gate = RiskGate(settings)
    # KXBTCD 4 contracts @ $0.48 (mid 0.475 -> 0.5c slippage/c = $0.02 total, entry_fee = $0.07)
    # With expected_edge_per_contract = $0.04 -> total_edge = $0.16 >= $0.07 + $0.02 + $0.04 = $0.13
    # Previously rejected when RT fee ($0.14) + slippage ($0.02) = $0.16 was compared against un-scaled $0.08!
    snap = OrderbookSnapshot(
        ticker="KXBTCD-26NOV1717-T64999.99",
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.47"), count=Decimal("20"))],
        no_bids=[PriceLevel(price=Decimal("0.52"), count=Decimal("20"))],  # YES ask = 0.48
    )
    intent = _make_intent(
        ticker="KXBTCD-26NOV1717-T64999.99",
        count=Decimal("4"),
        limit_price=Decimal("0.48"),
        expected_edge_dollars=Decimal("0.04"),
        expected_edge_per_contract=Decimal("0.04"),
        hold_to_expiry=True,
    )
    # Single-leg fee eligibility now comes from exchange close_time, not the AI's flag
    eval_res = gate.evaluate(intent, snap, _make_portfolio(), seconds_to_expiry=1800.0)
    assert eval_res.approved is True
    assert eval_res.estimated_entry_fee_dollars == Decimal("0.07")
    assert eval_res.estimated_round_trip_fee_dollars == Decimal("0.14")


def test_risk_gate_rejects_outside_015_085_price_band_and_chasing_ask() -> None:
    settings = AppSettings(
        _env_file=None,
        allowed_series_prefixes=["KXETH15M", "KXBTC"],
        max_slippage_cents=Decimal("2"),
    )
    gate = RiskGate(settings)

    # 1. $0.91 favorite (> $0.85) must be rejected on price band
    high_snap = OrderbookSnapshot(
        ticker="KXETH15M-26NOV171330-30",
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.90"), count=Decimal("50"))],
        no_bids=[PriceLevel(price=Decimal("0.09"), count=Decimal("50"))],  # YES ask = 0.91
    )
    high_intent = _make_intent(
        ticker="KXETH15M-26NOV171330-30",
        side=OrderSide.BID,
        count=Decimal("2"),
        limit_price=Decimal("0.91"),
        expected_edge_dollars=Decimal("0.16"),
        expected_edge_per_contract=Decimal("0.08"),
        hold_to_expiry=True,
    )
    res_high = gate.evaluate(high_intent, high_snap, _make_portfolio())
    assert res_high.approved is False
    assert any("outside allowed [$0.15, $0.85] band" in r for r in res_high.rejection_reasons)

    # 2. Live inside ask ($0.55) moved > max_slippage ($0.02) past reference ($0.50) -> rejected
    moved_snap = OrderbookSnapshot(
        ticker="KXETH15M-26NOV171330-30",
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.53"), count=Decimal("50"))],
        no_bids=[PriceLevel(price=Decimal("0.45"), count=Decimal("50"))],  # YES ask = 0.55
    )
    stale_intent = _make_intent(
        ticker="KXETH15M-26NOV171330-30",
        side=OrderSide.BID,
        count=Decimal("2"),
        limit_price=Decimal("0.50"),
        expected_edge_dollars=Decimal("0.16"),
        expected_edge_per_contract=Decimal("0.08"),
        hold_to_expiry=True,
    )
    res_moved = gate.evaluate(stale_intent, moved_snap, _make_portfolio())
    assert res_moved.approved is False
    assert any("moved > max slippage" in r for r in res_moved.rejection_reasons)


def test_risk_gate_allows_reducing_exit_above_085_and_blocks_correlated_expiry_hour() -> None:
    settings = AppSettings(
        _env_file=None,
        allowed_series_prefixes=["KXETHD", "KXETH15M", "KXBTC"],
        max_open_markets=1,
        min_net_edge_dollars=Decimal("0.04"),
    )
    gate = RiskGate(settings)
    portfolio = _make_portfolio()
    portfolio.positions["KXETHD-26NOV1719-T3119.99"] = Position(
        ticker="KXETHD-26NOV1719-T3119.99",
        net_contracts=Decimal("4"),
        avg_entry_price=Decimal("0.55"),
        cost_basis_dollars=Decimal("2.20"),
    )

    # 1. Reducing exit (ASK when net_contracts > 0) at $0.88 (> $0.85) with max_open_markets=1 is approved
    tp_snap = OrderbookSnapshot(
        ticker="KXETHD-26NOV1719-T3119.99",
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.88"), count=Decimal("20"))],
        no_bids=[PriceLevel(price=Decimal("0.10"), count=Decimal("20"))],
    )
    exit_intent = _make_intent(
        ticker="KXETHD-26NOV1719-T3119.99",
        side=OrderSide.ASK,
        count=Decimal("4"),
        limit_price=Decimal("0.88"),
        expected_edge_dollars=Decimal("0.01"),
        expected_edge_per_contract=Decimal("0.0025"),
        hold_to_expiry=False,
    )
    res_exit = gate.evaluate(exit_intent, tp_snap, portfolio)
    assert res_exit.approved is True, res_exit.rejection_reasons

    # 2. Second strike on same underlying + expiry_hour (KXETHD-26NOV1719-*) is blocked
    gate_multi = RiskGate(
        AppSettings(
            _env_file=None,
            allowed_series_prefixes=["KXETHD", "KXETH15M", "KXBTC"],
            max_open_markets=5,
        )
    )
    corr_snap = OrderbookSnapshot(
        ticker="KXETHD-26NOV1719-T3139.99",
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("20"))],
        no_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("20"))],
    )
    corr_intent = _make_intent(
        ticker="KXETHD-26NOV1719-T3139.99",
        side=OrderSide.BID,
        count=Decimal("2"),
        limit_price=Decimal("0.51"),
        expected_edge_dollars=Decimal("0.16"),
        expected_edge_per_contract=Decimal("0.08"),
    )
    res_corr = gate_multi.evaluate(corr_intent, corr_snap, portfolio)
    assert res_corr.approved is False
    assert any("Correlated underlying + expiry_hour" in r for r in res_corr.rejection_reasons)



# --- C4: risk-reducing exits pass the kill-switch and the order-rate limit ----------------


def _long_position_portfolio(cash: str = "5000.00") -> PortfolioState:
    p = _make_portfolio(cash)
    p.positions["KXBTC-26NOV-T90000"] = Position(
        ticker="KXBTC-26NOV-T90000",
        net_contracts=Decimal("10"),
        avg_entry_price=Decimal("0.45"),
        cost_basis_dollars=Decimal("4.50"),
    )
    return p


def test_kill_switch_blocks_entries_but_allows_reducing_exit() -> None:
    gate = RiskGate(AppSettings(_env_file=None))
    portfolio = _long_position_portfolio()
    portfolio.kill_switch_tripped = True
    exit_intent = _make_intent(side=OrderSide.ASK, count=Decimal("10"), limit_price=Decimal("0.50"))
    res_exit = gate.evaluate(exit_intent, _make_snapshot(), portfolio)
    assert res_exit.approved is True, res_exit.rejection_reasons

    entry = _make_intent(ticker="KXETH-26NOV-T3000")
    res_entry = gate.evaluate(entry, _make_snapshot("KXETH-26NOV-T3000"), portfolio)
    assert res_entry.approved is False
    assert any("Kill-switch" in r for r in res_entry.rejection_reasons)


def test_rate_limit_does_not_block_reducing_exit() -> None:
    gate = RiskGate(AppSettings(_env_file=None, max_orders_per_minute=1))
    now_ms = 1700000000000
    portfolio = _long_position_portfolio()
    portfolio.recent_order_timestamps_ms = [now_ms - 1_000]
    exit_intent = _make_intent(side=OrderSide.ASK, count=Decimal("10"), limit_price=Decimal("0.50"))
    assert gate.evaluate(exit_intent, _make_snapshot(), portfolio, now_ms=now_ms).approved is True
    entry = _make_intent(ticker="KXETH-26NOV-T3000")
    res = gate.evaluate(entry, _make_snapshot("KXETH-26NOV-T3000"), portfolio, now_ms=now_ms)
    assert any("rate limit" in r for r in res.rejection_reasons)


def test_oversized_exit_is_not_treated_as_reducing() -> None:
    gate = RiskGate(AppSettings(_env_file=None))
    portfolio = _long_position_portfolio()
    portfolio.kill_switch_tripped = True
    # Selling 20 against a 10-lot long would open a 10-lot NO position -> must be blocked
    flip = _make_intent(side=OrderSide.ASK, count=Decimal("20"), limit_price=Decimal("0.50"))
    assert gate.evaluate(flip, _make_snapshot(), portfolio).approved is False


# --- H1/H2/H3: edge units, AI-independent hold-to-expiry eligibility, KXFED -------------


def _flat_book(ticker: str) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        ticker=ticker,
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("500"))],
        no_bids=[PriceLevel(price=Decimal("0.50"), count=Decimal("500"))],  # YES ask 0.50
    )


def test_total_edge_is_never_reinterpreted_as_per_contract() -> None:
    gate = RiskGate(AppSettings(_env_file=None, allowed_series_prefixes=["KXBTCD"]))
    ticker = "KXBTCD-26NOV1717-T1"
    # $0.15 total over 10 contracts = $0.015/c, below the ~$0.018/c entry fee -> must reject
    intent = _make_intent(
        ticker=ticker, count=Decimal("10"), limit_price=Decimal("0.50"),
        expected_edge_dollars=Decimal("0.15"),
    )
    res = gate.evaluate(intent, _flat_book(ticker), _make_portfolio(), seconds_to_expiry=1800.0)
    assert res.approved is False
    assert any("Fee + slippage" in r or "Net edge" in r for r in res.rejection_reasons)


def test_min_net_edge_is_per_contract_regardless_of_count() -> None:
    settings = AppSettings(_env_file=None, min_net_edge_dollars=Decimal("0.02"))
    gate = RiskGate(settings)
    # $0.05/c gross - (0.036 RT fee + 0.01 slippage) = $0.004/c net < $0.02/c minimum
    intent = _make_intent(expected_edge_dollars=Decimal("0.50"))
    res = gate.evaluate(intent, _make_snapshot(), _make_portfolio())
    assert res.approved is False
    assert any("Net edge" in r for r in res.rejection_reasons)


def test_ai_hold_to_expiry_flag_does_not_loosen_fee_accounting() -> None:
    gate = RiskGate(AppSettings(_env_file=None, allowed_series_prefixes=["KXBTCD"]))
    ticker = "KXBTCD-26NOV1717-T1"
    # $0.04/c gross: passes with single-leg fee (0.018 + 0.005 slip) but not round-trip (0.036 + 0.005)
    intent = _make_intent(
        ticker=ticker, count=Decimal("10"), limit_price=Decimal("0.50"),
        expected_edge_dollars=Decimal("0.40"), expected_edge_per_contract=Decimal("0.04"),
        hold_to_expiry=True,
    )
    no_expiry = gate.evaluate(intent, _flat_book(ticker), _make_portfolio())
    assert no_expiry.approved is False  # AI claimed hold_to_expiry, but no exchange close_time
    far = gate.evaluate(intent, _flat_book(ticker), _make_portfolio(), seconds_to_expiry=86400.0)
    assert far.approved is False
    near = gate.evaluate(intent, _flat_book(ticker), _make_portfolio(), seconds_to_expiry=1800.0)
    assert near.approved is True, near.rejection_reasons


def test_kxfed_is_not_treated_as_intraday_binary() -> None:
    gate = RiskGate(AppSettings(_env_file=None))
    ticker = "KXFED-26DEC-T4.00"
    intent = _make_intent(
        ticker=ticker, count=Decimal("10"), limit_price=Decimal("0.50"),
        expected_edge_dollars=Decimal("0.40"), expected_edge_per_contract=Decimal("0.04"),
        hold_to_expiry=True,
    )
    # A Dec FOMC contract weeks away: round-trip fee applies -> $0.04/c edge is insufficient
    res = gate.evaluate(intent, _flat_book(ticker), _make_portfolio(), seconds_to_expiry=30 * 86400.0)
    assert res.approved is False
    assert res.estimated_round_trip_fee_dollars == Decimal("0.36")
