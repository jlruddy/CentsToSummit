"""Test derived orderbook math, multi-level fill simulation, fee math, and V2 order payload."""

from decimal import Decimal

from kalshi_poc.models import (
    ExecutionMode,
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    PriceLevel,
    TradeIntent,
    V2CreateOrderRequest,
)
from kalshi_poc.simulator import (
    ShadowSimulator,
    calculate_maker_fee,
    calculate_taker_fee,
    walk_book,
)


def test_kalshi_fee_ceiling_math() -> None:
    # Taker: ceil(0.07 * 10 * 0.50 * 0.50) = ceil(0.175) = 0.18
    assert calculate_taker_fee(Decimal("10"), Decimal("0.50")) == Decimal("0.18")
    # Maker: ceil(0.0175 * 10 * 0.50 * 0.50) = ceil(0.04375) = 0.05
    assert calculate_maker_fee(Decimal("10"), Decimal("0.50")) == Decimal("0.05")


def test_derived_yes_no_orderbook_and_multi_level_walk() -> None:
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.47", "15"], ["0.49", "10"]],
            "no_dollars": [["0.46", "20"], ["0.49", "10"]],
        }
    }
    snap = OrderbookSnapshot.from_kalshi_orderbook_fp("KXBTC-TEST", payload, 1700000000000)

    assert snap.best_yes_bid == Decimal("0.49")
    assert snap.best_no_bid == Decimal("0.49")
    # Effective YES asks: 1 - 0.49 = 0.51 (10), 1 - 0.46 = 0.54 (20)
    assert snap.best_yes_ask == Decimal("0.51")
    assert snap.yes_spread == Decimal("0.02")

    # Walk 20 YES buy contracts across 2 ask levels: 10 @ 0.51 + 10 @ 0.54 => VWAP 0.5250
    walk = walk_book(snap, OrderSide.BID, Decimal("20"))
    assert walk.fully_fillable is True
    assert walk.levels_consumed == 2
    assert walk.vwap_price == Decimal("0.5250")


def test_shadow_simulator_updates_portfolio_and_v2_serialization() -> None:
    snap = OrderbookSnapshot(
        ticker="KXINX-TEST",
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("20"))],
        no_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("20"))],
    )
    intent = TradeIntent(
        timestamp_ms=1700000000000,
        ticker="KXINX-TEST",
        side=OrderSide.BID,
        count=Decimal("10"),
        limit_price=Decimal("0.51"),
        expected_edge_dollars=Decimal("0.50"),
        confidence=Decimal("0.75"),
        inference_latency_ms=3,
        rationale="Sim test",
        provider="stub_momentum",
    )

    v2 = V2CreateOrderRequest.from_intent(intent)
    assert v2.ticker == "KXINX-TEST"
    assert v2.side == "bid"
    assert v2.count == "10"
    assert v2.price == "0.5100"
    assert v2.time_in_force.value == "immediate_or_cancel"

    portfolio = PortfolioState(
        cash_balance_dollars=Decimal("1000.00"),
        starting_capital_dollars=Decimal("1000.00"),
        daily_start_equity_dollars=Decimal("1000.00"),
    )
    sim = ShadowSimulator()
    rec = sim.simulate_fill(intent, snap, portfolio=portfolio, mode=ExecutionMode.SHADOW)
    assert rec.status == "filled"
    assert rec.fill_price == Decimal("0.5100")
    assert rec.fee_dollars == Decimal("0.18")
    # Cash reduced by 10 * 0.51 + 0.18 = 5.28
    assert portfolio.cash_balance_dollars == Decimal("994.7200")
    assert portfolio.positions["KXINX-TEST"].net_contracts == Decimal("10")


def test_shadow_auto_settles_expired_positions() -> None:
    from datetime import datetime, timedelta, timezone
    from kalshi_poc.models import Position

    now = datetime(2026, 11, 16, 23, 30, 0, tzinfo=timezone.utc)
    portfolio = PortfolioState(
        cash_balance_dollars=Decimal("10.00"),
        starting_capital_dollars=Decimal("15.00"),
        daily_start_equity_dollars=Decimal("15.00"),
        positions={
            "KXBTC15M-WIN": Position(
                ticker="KXBTC15M-WIN",
                net_contracts=Decimal("4"),
                avg_entry_price=Decimal("0.48"),
                cost_basis_dollars=Decimal("1.92"),
            ),
            "KXBTC15M-OPEN": Position(
                ticker="KXBTC15M-OPEN",
                net_contracts=Decimal("2"),
                avg_entry_price=Decimal("0.50"),
                cost_basis_dollars=Decimal("1.00"),
            ),
        },
    )
    expiries = {
        "KXBTC15M-WIN": now - timedelta(minutes=1),  # expired
        "KXBTC15M-OPEN": now + timedelta(minutes=10),  # still open
    }
    books = {
        "KXBTC15M-WIN": OrderbookSnapshot(
            ticker="KXBTC15M-WIN",
            timestamp_ms=int(now.timestamp() * 1000),
            yes_bids=[PriceLevel(price=Decimal("0.60"), count=Decimal("10"))],
            no_bids=[PriceLevel(price=Decimal("0.38"), count=Decimal("10"))],  # mid=0.61 >= 0.50
        )
    }

    sim = ShadowSimulator()
    settled = sim.settle_expired_positions(portfolio, expiries, books, now_utc=now)
    assert len(settled) == 1
    assert settled[0].ticker == "KXBTC15M-WIN"
    assert settled[0].status == "SHADOW_SETTLEMENT"
    assert settled[0].fill_price == Decimal("1.0000")
    assert settled[0].fee_dollars == Decimal("0.00")
    # 4 * $1.00 = $4.00 credited -> cash = $14.00, KXBTC15M-WIN removed from positions
    assert portfolio.cash_balance_dollars == Decimal("14.00")
    assert "KXBTC15M-WIN" not in portfolio.positions
    assert "KXBTC15M-OPEN" in portfolio.positions


# --- C1: symmetric short (NO) accounting in _apply_fill_to_portfolio ---------------------


def _flat_portfolio(cash: str = "1000.00") -> PortfolioState:
    c = Decimal(cash)
    return PortfolioState(
        cash_balance_dollars=c, starting_capital_dollars=c, daily_start_equity_dollars=c
    )


def test_short_round_trip_at_same_price_returns_cash() -> None:
    p = _flat_portfolio()
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.ASK, Decimal("10"), Decimal("0.40"), Decimal("0"), 0)
    assert p.cash_balance_dollars == Decimal("994.00")
    assert p.positions["T"].net_contracts == Decimal("-10")
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.BID, Decimal("10"), Decimal("0.40"), Decimal("0"), 0)
    assert p.cash_balance_dollars == Decimal("1000.00")
    assert "T" not in p.positions


def test_short_close_at_profit_credits_no_proceeds_minus_fee() -> None:
    p = _flat_portfolio()
    # Buy 10 NO @ (1 - 0.40) = $6.00, fee $0.17
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.ASK, Decimal("10"), Decimal("0.40"), Decimal("0.17"), 0)
    assert p.cash_balance_dollars == Decimal("993.83")
    # YES falls to 0.30 -> close by buying YES: credit 10 * 0.70 - fee 0.15
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.BID, Decimal("10"), Decimal("0.30"), Decimal("0.15"), 0)
    assert p.cash_balance_dollars == Decimal("1000.68")
    assert "T" not in p.positions


def test_partial_short_close_reduces_cost_basis_and_books_realized_pnl() -> None:
    p = _flat_portfolio()
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.ASK, Decimal("10"), Decimal("0.40"), Decimal("0"), 0)
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.BID, Decimal("4"), Decimal("0.30"), Decimal("0"), 0)
    pos = p.positions["T"]
    assert pos.net_contracts == Decimal("-6")
    assert pos.cost_basis_dollars == Decimal("3.60")  # 6 NO @ $0.60
    assert pos.avg_entry_price == Decimal("0.4000")
    assert pos.realized_pnl_dollars == Decimal("0.40")  # 4 * (0.40 - 0.30)
    assert p.cash_balance_dollars == Decimal("996.80")  # 994.00 + 4 * 0.70


def test_short_to_long_flip_opens_remainder_at_fill_price() -> None:
    p = _flat_portfolio()
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.ASK, Decimal("4"), Decimal("0.40"), Decimal("0"), 0)
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.BID, Decimal("10"), Decimal("0.50"), Decimal("0"), 0)
    pos = p.positions["T"]
    assert pos.net_contracts == Decimal("6")
    assert pos.cost_basis_dollars == Decimal("3.00")
    assert pos.avg_entry_price == Decimal("0.5000")
    # 1000 - 2.40 (open NO) + 4 * 0.50 (close NO) - 6 * 0.50 (open YES)
    assert p.cash_balance_dollars == Decimal("996.60")


def test_adding_to_short_averages_yes_entry_price() -> None:
    p = _flat_portfolio()
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.ASK, Decimal("10"), Decimal("0.40"), Decimal("0"), 0)
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.ASK, Decimal("10"), Decimal("0.30"), Decimal("0"), 0)
    pos = p.positions["T"]
    assert pos.net_contracts == Decimal("-20")
    assert pos.cost_basis_dollars == Decimal("13.00")  # 10 * 0.60 + 10 * 0.70
    assert pos.avg_entry_price == Decimal("0.3500")


def test_long_to_short_flip_and_long_round_trip_unchanged() -> None:
    p = _flat_portfolio()
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.BID, Decimal("10"), Decimal("0.40"), Decimal("0"), 0)
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.ASK, Decimal("10"), Decimal("0.40"), Decimal("0"), 0)
    assert p.cash_balance_dollars == Decimal("1000.00")
    assert "T" not in p.positions

    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.BID, Decimal("4"), Decimal("0.50"), Decimal("0"), 0)
    ShadowSimulator._apply_fill_to_portfolio(p, "T", OrderSide.ASK, Decimal("10"), Decimal("0.60"), Decimal("0"), 0)
    pos = p.positions["T"]
    assert pos.net_contracts == Decimal("-6")
    assert pos.cost_basis_dollars == Decimal("2.40")  # 6 NO @ $0.40
    assert pos.avg_entry_price == Decimal("0.6000")
