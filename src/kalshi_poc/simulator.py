"""Shadow execution engine modeling multi-level book slippage and Kalshi fees."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING

from kalshi_poc.models import (
    ExecutionMode,
    ExecutionRecord,
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    Position,
    PriceLevel,
    TradeIntent,
)

TAKER_FEE_RATE = Decimal("0.07")
MAKER_FEE_RATE = Decimal("0.0175")
ONE = Decimal("1.00")
CENT = Decimal("0.01")


def ceil_to_cent(amount: Decimal) -> Decimal:
    """Round a dollar amount up to the nearest cent ($0.01)."""
    if amount <= Decimal("0"):
        return Decimal("0.00")
    cents = (amount * Decimal("100")).to_integral_value(rounding=ROUND_CEILING)
    return (cents / Decimal("100")).quantize(CENT)


def calculate_taker_fee(count: Decimal, price: Decimal) -> Decimal:
    """Kalshi taker fee: ceil(0.07 * C * P * (1 - P)) to nearest cent."""
    raw = TAKER_FEE_RATE * count * price * (ONE - price)
    return ceil_to_cent(raw)


def calculate_maker_fee(count: Decimal, price: Decimal) -> Decimal:
    """Kalshi maker fee: ceil(0.0175 * C * P * (1 - P)) to nearest cent."""
    raw = MAKER_FEE_RATE * count * price * (ONE - price)
    return ceil_to_cent(raw)


@dataclass(frozen=True)
class BookWalkResult:
    """Result of walking an orderbook across multiple depth levels."""

    requested_count: Decimal
    fillable_count: Decimal
    vwap_price: Decimal
    best_level_price: Decimal
    worst_level_price: Decimal
    slippage_dollars: Decimal
    slippage_cents_per_contract: Decimal
    levels_consumed: int

    @property
    def fully_fillable(self) -> bool:
        return self.fillable_count >= self.requested_count


def walk_book(
    snapshot: OrderbookSnapshot,
    side: OrderSide,
    count: Decimal,
    limit_price: Decimal | None = None,
) -> BookWalkResult:
    """Walk live derived orderbook levels to compute exact VWAP fill price and slippage.

    - OrderSide.BID (buy YES): walks `snapshot.effective_yes_asks` ascending.
    - OrderSide.ASK (sell YES / buy NO): walks `snapshot.yes_bids` descending.
    """
    if side == OrderSide.BID:
        levels: list[PriceLevel] = snapshot.effective_yes_asks
        price_ok = lambda p: limit_price is None or p <= limit_price  # noqa: E731
    else:
        levels = list(reversed(snapshot.yes_bids))
        price_ok = lambda p: limit_price is None or p >= limit_price  # noqa: E731

    if not levels:
        return BookWalkResult(
            requested_count=count,
            fillable_count=Decimal("0"),
            vwap_price=Decimal("0.0000"),
            best_level_price=Decimal("0.0000"),
            worst_level_price=Decimal("0.0000"),
            slippage_dollars=Decimal("0.00"),
            slippage_cents_per_contract=Decimal("0"),
            levels_consumed=0,
        )

    remaining = count
    filled = Decimal("0")
    total_notional = Decimal("0")
    best_price = levels[0].price
    worst_price = best_price
    levels_consumed = 0

    for lvl in levels:
        if remaining <= Decimal("0"):
            break
        if not price_ok(lvl.price):
            break
        take = min(remaining, lvl.count)
        filled += take
        total_notional += take * lvl.price
        worst_price = lvl.price
        remaining -= take
        levels_consumed += 1

    if filled == Decimal("0"):
        return BookWalkResult(
            requested_count=count,
            fillable_count=Decimal("0"),
            vwap_price=Decimal("0.0000"),
            best_level_price=best_price,
            worst_level_price=best_price,
            slippage_dollars=Decimal("0.00"),
            slippage_cents_per_contract=Decimal("0"),
            levels_consumed=0,
        )

    vwap = (total_notional / filled).quantize(Decimal("0.0001"))

    # Slippage relative to mid_yes_price if available, else relative to best_level_price
    ref_price = snapshot.mid_yes_price if snapshot.mid_yes_price is not None else best_price
    if side == OrderSide.BID:
        per_contract_slip = max(Decimal("0"), vwap - ref_price)
    else:
        per_contract_slip = max(Decimal("0"), ref_price - vwap)

    slippage_dollars = (per_contract_slip * filled).quantize(Decimal("0.0001"))
    slippage_cents = (per_contract_slip * Decimal("100")).quantize(Decimal("0.01"))

    return BookWalkResult(
        requested_count=count,
        fillable_count=filled,
        vwap_price=vwap,
        best_level_price=best_price,
        worst_level_price=worst_price,
        slippage_dollars=slippage_dollars,
        slippage_cents_per_contract=slippage_cents,
        levels_consumed=levels_consumed,
    )


class ShadowSimulator:
    """Simulates fills against live orderbooks and updates shadow portfolio state."""

    def __init__(self) -> None:
        self.open_positions: dict[str, Position] = {}
        self.resting_orders: dict[str, dict] = {}
        self.realized_pnl: Decimal = Decimal("0.00")

    def reset(self, portfolio: PortfolioState, starting_equity: Decimal) -> None:
        """Atomically reset shadow simulator and portfolio state to a fresh session."""
        self.open_positions.clear()
        self.resting_orders.clear()
        self.realized_pnl = Decimal("0.00")
        portfolio.positions.clear()
        portfolio.resting_order_collateral_dollars = Decimal("0.00")
        portfolio.recent_order_timestamps_ms.clear()
        portfolio.cash_balance_dollars = starting_equity
        portfolio.starting_capital_dollars = starting_equity
        portfolio.daily_start_equity_dollars = starting_equity

    def simulate_fill(
        self,
        intent: TradeIntent,
        snapshot: OrderbookSnapshot,
        portfolio: PortfolioState | None = None,
        mode: ExecutionMode = ExecutionMode.SHADOW,
        is_maker: bool = False,
    ) -> ExecutionRecord:
        walk = walk_book(snapshot, intent.side, intent.count, intent.limit_price)
        now_ms = int(time.time() * 1000)
        order_id = f"sim-{uuid.uuid4().hex[:12]}"
        client_order_id = str(uuid.uuid4())

        if walk.fillable_count <= Decimal("0"):
            return ExecutionRecord(
                order_id=order_id,
                client_order_id=client_order_id,
                intent_id=intent.intent_id,
                ticker=intent.ticker,
                mode=mode,
                side=intent.side,
                count=Decimal("0"),
                requested_price=intent.limit_price,
                fill_price=Decimal("0.0000"),
                slippage_dollars=Decimal("0.00"),
                fee_dollars=Decimal("0.00"),
                is_maker=is_maker,
                status="no_liquidity",
                timestamp_ms=now_ms,
                shadow_fill_price=Decimal("0.0000"),
                shadow_fee_dollars=Decimal("0.00"),
            )

        fill_count = walk.fillable_count
        fill_price = walk.vwap_price
        fee = (
            calculate_maker_fee(fill_count, fill_price)
            if is_maker
            else calculate_taker_fee(fill_count, fill_price)
        )

        if portfolio is not None and mode == ExecutionMode.SHADOW:
            self._apply_fill_to_portfolio(
                portfolio=portfolio,
                ticker=intent.ticker,
                side=intent.side,
                count=fill_count,
                fill_price=fill_price,
                fee=fee,
                timestamp_ms=now_ms,
            )

        status = "filled" if walk.fully_fillable else "partial_fill"
        return ExecutionRecord(
            order_id=order_id,
            client_order_id=client_order_id,
            intent_id=intent.intent_id,
            ticker=intent.ticker,
            mode=mode,
            side=intent.side,
            count=fill_count,
            requested_price=intent.limit_price,
            fill_price=fill_price,
            slippage_dollars=walk.slippage_dollars,
            fee_dollars=fee,
            is_maker=is_maker,
            status=status,
            timestamp_ms=now_ms,
            shadow_fill_price=fill_price,
            shadow_fee_dollars=fee,
        )

    @staticmethod
    def _apply_fill_to_portfolio(
        portfolio: PortfolioState,
        ticker: str,
        side: OrderSide,
        count: Decimal,
        fill_price: Decimal,
        fee: Decimal,
        timestamp_ms: int,
    ) -> None:
        pos = portfolio.positions.get(ticker) or Position(ticker=ticker)
        portfolio.recent_order_timestamps_ms.append(timestamp_ms)

        if side == OrderSide.BID:
            cost = count * fill_price + fee
            portfolio.cash_balance_dollars -= cost
            new_contracts = pos.net_contracts + count
            pos.cost_basis_dollars += count * fill_price
            pos.net_contracts = new_contracts
            if new_contracts != Decimal("0"):
                pos.avg_entry_price = (
                    pos.cost_basis_dollars / abs(new_contracts)
                ).quantize(Decimal("0.0001"))
        else:
            # Selling YES / buying NO at (1 - fill_price)
            no_price = ONE - fill_price
            cost = count * no_price + fee
            if pos.net_contracts > Decimal("0"):
                close_qty = min(pos.net_contracts, count)
                proceeds = close_qty * fill_price - fee
                realized = close_qty * (fill_price - pos.avg_entry_price) - fee
                pos.realized_pnl_dollars += realized
                pos.net_contracts -= close_qty
                pos.cost_basis_dollars = max(
                    Decimal("0.00"), pos.cost_basis_dollars - close_qty * pos.avg_entry_price
                )
                portfolio.cash_balance_dollars += proceeds
                rem = count - close_qty
                if rem > Decimal("0"):
                    portfolio.cash_balance_dollars -= rem * no_price
                    pos.net_contracts -= rem
                    pos.cost_basis_dollars += rem * no_price
                    pos.avg_entry_price = fill_price
            else:
                portfolio.cash_balance_dollars -= cost
                pos.net_contracts -= count
                pos.cost_basis_dollars += count * no_price
                pos.avg_entry_price = fill_price

        if pos.net_contracts == Decimal("0"):
            pos.cost_basis_dollars = Decimal("0.00")
            portfolio.positions.pop(ticker, None)
        else:
            portfolio.positions[ticker] = pos

    def settle_expired_positions(
        self,
        portfolio: PortfolioState,
        market_expiries: dict[str, datetime],
        orderbooks: dict[str, OrderbookSnapshot],
        now_utc: datetime | None = None,
    ) -> list[ExecutionRecord]:
        """Auto-settle expired shadow positions at $1.00 (mid >= 0.50) or $0.00 ($0 fee)."""
        now = now_utc or datetime.now(timezone.utc)
        now_ms = int(now.timestamp() * 1000)
        records: list[ExecutionRecord] = []

        for ticker, pos in list(portfolio.positions.items()):
            if pos.net_contracts == Decimal("0"):
                portfolio.positions.pop(ticker, None)
                self.open_positions.pop(ticker, None)
                continue

            exp = market_expiries.get(ticker)
            if exp is None:
                continue
            exp_utc = exp if exp.tzinfo is not None else exp.replace(tzinfo=timezone.utc)
            if exp_utc > now:
                continue

            snap = orderbooks.get(ticker)
            last_mid = (
                snap.mid_yes_price
                if snap is not None and snap.mid_yes_price is not None
                else pos.avg_entry_price
            )
            settle_yes_price = Decimal("1.0000") if last_mid >= Decimal("0.50") else Decimal("0.0000")

            if pos.net_contracts > Decimal("0"):
                qty = pos.net_contracts
                payout = (qty * settle_yes_price).quantize(Decimal("0.01"))
                realized = payout - pos.cost_basis_dollars
                side = OrderSide.ASK
            else:
                qty = abs(pos.net_contracts)
                no_settle = Decimal("1.0000") - settle_yes_price
                payout = (qty * no_settle).quantize(Decimal("0.01"))
                realized = payout - pos.cost_basis_dollars
                side = OrderSide.BID

            portfolio.cash_balance_dollars += payout
            self.realized_pnl += realized
            portfolio.positions.pop(ticker, None)
            self.open_positions.pop(ticker, None)

            records.append(
                ExecutionRecord(
                    order_id=f"settle-{uuid.uuid4().hex[:10]}",
                    client_order_id=str(uuid.uuid4()),
                    intent_id=f"settle-{ticker}",
                    ticker=ticker,
                    mode=ExecutionMode.SHADOW,
                    side=side,
                    count=qty,
                    requested_price=settle_yes_price,
                    fill_price=settle_yes_price,
                    slippage_dollars=Decimal("0.00"),
                    fee_dollars=Decimal("0.00"),
                    is_maker=False,
                    status="SHADOW_SETTLEMENT",
                    timestamp_ms=now_ms,
                    shadow_fill_price=settle_yes_price,
                    shadow_fee_dollars=Decimal("0.00"),
                )
            )

        return records
