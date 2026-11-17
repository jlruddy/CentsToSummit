"""Deterministic pre-trade risk & tax-hygiene firewall (zero LLM bypass)."""

from __future__ import annotations

import time
from decimal import Decimal

from kalshi_poc.config import AppSettings
from kalshi_poc.models import (
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    RiskEvaluation,
    TradeIntent,
)
from kalshi_poc.simulator import calculate_taker_fee, walk_book

ONE = Decimal("1.00")


def extract_series_prefix(ticker: str) -> str:
    """Extract the series prefix from a Kalshi ticker (e.g. KXBTC-26NOV-T90000 -> KXBTC)."""
    return ticker.strip().upper().split("-", 1)[0]


class RiskGate:
    """Pure deterministic pre-trade firewall. Hot-reloadable via update_settings()."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self.kill_switch_tripped: bool = False
        self.daily_loss: Decimal = Decimal("0.00")
        self.orders_this_minute: list[int] = []

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings

    def reset(self, portfolio: PortfolioState | None = None) -> None:
        """Clear kill-switch and intra-minute / daily risk counters."""
        self.kill_switch_tripped = False
        self.daily_loss = Decimal("0.00")
        self.orders_this_minute.clear()
        if portfolio is not None:
            portfolio.kill_switch_tripped = False
            portfolio.recent_order_timestamps_ms.clear()

    def evaluate(
        self,
        intent: TradeIntent,
        snapshot: OrderbookSnapshot | None,
        portfolio: PortfolioState,
        now_ms: int | None = None,
    ) -> RiskEvaluation:
        ts = now_ms if now_ms is not None else int(time.time() * 1000)
        s = self._settings
        reasons: list[str] = []

        # 1. Kill-switch & daily drawdown check
        self.daily_loss = max(Decimal("0.00"), -portfolio.daily_pnl_dollars)
        if portfolio.kill_switch_tripped or self.kill_switch_tripped:
            self.kill_switch_tripped = True
            portfolio.kill_switch_tripped = True
            reasons.append("Kill-switch is tripped")
        if portfolio.daily_pnl_dollars <= -s.daily_drawdown_kill_switch:
            self.kill_switch_tripped = True
            portfolio.kill_switch_tripped = True
            reasons.append(
                f"Daily drawdown kill-switch breached ({portfolio.daily_pnl_dollars:.2f} <= -{s.daily_drawdown_kill_switch:.2f})"
            )

        # 2. Tax-Hygiene Allowlist
        series_prefix = extract_series_prefix(intent.ticker)
        allowed = [p.upper() for p in s.allowed_series_prefixes]
        if s.enforce_financial_only and not any(
            series_prefix == prefix or series_prefix.startswith(prefix)
            for prefix in allowed
        ):
            reasons.append(
                f"Series '{series_prefix}' not on tax-hygiene allowlist ({', '.join(allowed)})"
            )

        # 3. Rate-limit, open markets & per-market position limits
        one_minute_ago = ts - 60_000
        self.orders_this_minute = [
            t for t in portfolio.recent_order_timestamps_ms if t >= one_minute_ago
        ]
        recent_orders = self.orders_this_minute
        if len(recent_orders) >= s.max_orders_per_minute:
            reasons.append(
                f"Order rate limit reached ({len(recent_orders)}/{s.max_orders_per_minute} per min)"
            )

        active_tickers = {
            k for k, pos in portfolio.positions.items() if pos.net_contracts != Decimal("0")
        }
        if intent.ticker not in active_tickers and len(active_tickers) >= s.max_open_markets:
            reasons.append(
                f"Max open markets reached ({len(active_tickers)}/{s.max_open_markets})"
            )

        existing_pos = portfolio.positions.get(intent.ticker)
        existing_notional = (
            existing_pos.cost_basis_dollars if existing_pos else Decimal("0.00")
        )
        unit_collateral = (
            intent.limit_price if intent.side == OrderSide.BID else (ONE - intent.limit_price)
        )
        proposed_notional = (intent.count * unit_collateral).quantize(Decimal("0.01"))
        if existing_notional + proposed_notional > s.max_position_dollars_per_market:
            reasons.append(
                f"Market position limit exceeded (${existing_notional + proposed_notional:.2f} > ${s.max_position_dollars_per_market:.2f})"
            )

        # 4. Orderbook depth, price-band [0.15, 0.85], slippage, and fee-edge filter
        est_fill_price = intent.limit_price
        est_slippage = Decimal("0.00")
        entry_fee = calculate_taker_fee(intent.count, intent.limit_price)
        round_trip_fee = entry_fee * Decimal("2")
        max_slip_dollars = (s.max_slippage_cents / Decimal("100")).quantize(Decimal("0.0001"))

        if snapshot is None:
            reasons.append(f"No live orderbook snapshot available for {intent.ticker}")
        elif snapshot.best_yes_bid is None or snapshot.best_yes_ask is None:
            reasons.append(
                f"Orderbook for {intent.ticker} is one-sided or empty (bid={snapshot.best_yes_bid}, ask={snapshot.best_yes_ask})"
            )
        else:
            if intent.side == OrderSide.BID and (
                snapshot.best_yes_ask - intent.limit_price
            ) > max_slip_dollars:
                reasons.append(
                    f"Live inside ask (${snapshot.best_yes_ask:.4f}) moved > max slippage (${max_slip_dollars:.2f}) past reference (${intent.limit_price:.4f})"
                )
            elif intent.side == OrderSide.ASK and (
                intent.limit_price - snapshot.best_yes_bid
            ) > max_slip_dollars:
                reasons.append(
                    f"Live inside bid (${snapshot.best_yes_bid:.4f}) moved > max slippage (${max_slip_dollars:.2f}) past reference (${intent.limit_price:.4f})"
                )

            walk = walk_book(
                snapshot, intent.side, intent.count, limit_price=intent.limit_price
            )
            if not walk.fully_fillable:
                reasons.append(
                    f"Insufficient book depth (requested {intent.count}, available {walk.fillable_count})"
                )
            else:
                est_fill_price = walk.vwap_price
                est_slippage = walk.slippage_dollars
                entry_fee = calculate_taker_fee(intent.count, est_fill_price)
                round_trip_fee = entry_fee * Decimal("2")

                if walk.slippage_cents_per_contract > s.max_slippage_cents:
                    reasons.append(
                        f"Estimated slippage {walk.slippage_cents_per_contract}c exceeds max {s.max_slippage_cents}c"
                    )

                is_intraday_binary = (
                    series_prefix.endswith("15M")
                    or series_prefix.endswith("D")
                    or series_prefix.endswith("U")
                )
                hold_to_expiry_eligible = intent.hold_to_expiry or is_intraday_binary
                primary_fee = entry_fee if hold_to_expiry_eligible else round_trip_fee
                total_expected_edge = (
                    intent.expected_edge_per_contract * intent.count
                    if intent.expected_edge_per_contract is not None
                    else intent.expected_edge_dollars
                )

                total_order_cost = (
                    est_fill_price if intent.side == OrderSide.BID else (ONE - est_fill_price)
                ) * intent.count
                if total_order_cost > Decimal("0") and (
                    primary_fee / total_order_cost
                ) > Decimal("0.15"):
                    fee_pct = (primary_fee / total_order_cost * Decimal("100")).quantize(
                        Decimal("0.1")
                    )
                    reasons.append(
                        f"Fee-to-notional ratio {fee_pct}% exceeds 15.0% max (${primary_fee:.2f} / ${total_order_cost:.2f})"
                    )

                total_friction = est_slippage + primary_fee
                net_edge = total_expected_edge - total_friction
                min_required_net = (
                    Decimal("0.01") * intent.count
                    if hold_to_expiry_eligible
                    else s.min_net_edge_dollars
                )
                if total_friction >= total_expected_edge:
                    reasons.append(
                        f"Fee + slippage (${total_friction:.4f}) >= expected edge (${total_expected_edge:.4f})"
                    )
                elif net_edge < min_required_net:
                    reasons.append(
                        f"Net edge (${net_edge:.4f}) below minimum (${min_required_net:.2f})"
                    )

        # Enforce 0.15 <= effective_price <= 0.85 on both YES and NO sides
        effective_fill_price = (
            est_fill_price if intent.side == OrderSide.BID else (ONE - est_fill_price)
        )
        effective_limit_price = (
            intent.limit_price if intent.side == OrderSide.BID else (ONE - intent.limit_price)
        )
        if not (
            Decimal("0.15") <= est_fill_price <= Decimal("0.85")
            and Decimal("0.15") <= (ONE - est_fill_price) <= Decimal("0.85")
            and Decimal("0.15") <= intent.limit_price <= Decimal("0.85")
            and Decimal("0.15") <= (ONE - intent.limit_price) <= Decimal("0.85")
        ):
            reasons.append(
                f"Effective price ${effective_fill_price:.4f} (YES ${est_fill_price:.4f}, limit ${effective_limit_price:.4f}) outside allowed [$0.15, $0.85] band"
            )

        # 5. Hard Equity Floor (EQUITY_FLOOR)
        proposed_collateral = proposed_notional + entry_fee
        worst_case_equity_after = (
            portfolio.current_equity_dollars
            - portfolio.open_risk_collateral_dollars
            - proposed_collateral
        ).quantize(Decimal("0.01"))

        if worst_case_equity_after < s.equity_floor:
            reasons.append(
                f"Equity floor breach (worst-case ${worst_case_equity_after:.2f} < floor ${s.equity_floor:.2f})"
            )

        return RiskEvaluation(
            intent_id=intent.intent_id,
            timestamp_ms=ts,
            ticker=intent.ticker,
            approved=len(reasons) == 0,
            rejection_reasons=reasons,
            estimated_fill_price=est_fill_price,
            estimated_slippage_dollars=est_slippage,
            estimated_entry_fee_dollars=entry_fee,
            estimated_round_trip_fee_dollars=round_trip_fee,
            worst_case_equity_after=worst_case_equity_after,
        )
