"""Deterministic pre-trade risk & tax-hygiene firewall (zero LLM bypass)."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)
ONE = Decimal("1.00")
# Contracts closing within this window are treated as held to settlement (entry fee only).
HOLD_TO_EXPIRY_MAX_SECONDS = 7200.0
HOLD_TO_EXPIRY_MIN_NET_EDGE = Decimal("0.015")


def extract_series_prefix(ticker: str) -> str:
    """Extract the series prefix from a Kalshi ticker (e.g. KXBTC-26NOV-T90000 -> KXBTC)."""
    return ticker.strip().upper().split("-", 1)[0]


def extract_underlying_expiry_hour(ticker: str) -> str | None:
    """Extract normalized (underlying:YYMMMDDHH) key for hourly/intraday correlation gating."""
    parts = ticker.strip().upper().split("-")
    if len(parts) < 2:
        return None
    raw = parts[0]
    if raw.startswith("KXINX"):
        underlying = "INX"
    elif raw.startswith("KXNASDAQ"):
        underlying = "NASDAQ"
    elif raw.startswith("KXBTC"):
        underlying = "BTC"
    elif raw.startswith("KXETH"):
        underlying = "ETH"
    elif raw.startswith("KXSOL"):
        underlying = "SOL"
    elif raw.startswith("KXXRP"):
        underlying = "XRP"
    else:
        underlying = raw

    dt_token = parts[1]
    if len(dt_token) >= 9 and dt_token[:2].isdigit() and dt_token[5:9].isdigit():
        return f"{underlying}:{dt_token[:9]}"
    return None


class RiskGate:
    """Pure deterministic pre-trade firewall. Hot-reloadable via update_settings()."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self.kill_switch_tripped: bool = False
        self.daily_loss: Decimal = Decimal("0.00")
        self.orders_this_minute: list[int] = []

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings

    def effective_max_position_dollars(self, portfolio: PortfolioState) -> Decimal:
        """Return active per-market dollar cap (dynamic % of house money when enabled)."""
        s = self._settings
        if not s.dynamic_sizing_enabled:
            return s.max_position_dollars_per_market
        house_money = max(
            Decimal("0.00"), portfolio.current_equity_dollars - s.equity_floor
        )
        deployable_budget = max(
            Decimal("0.00"),
            portfolio.current_equity_dollars
            - s.equity_floor
            - portfolio.open_risk_collateral_dollars,
        )
        dynamic_cap = max(
            Decimal("1.00"),
            (house_money * s.house_money_risk_pct).quantize(Decimal("0.01")),
        )
        if deployable_budget >= Decimal("1.00"):
            return min(deployable_budget, dynamic_cap)
        return dynamic_cap

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
        *,
        seconds_to_expiry: float | None = None,
        emergency_exit: bool = False,
    ) -> RiskEvaluation:
        """Deterministically approve/reject `intent`.

        `seconds_to_expiry` comes from the exchange's market close_time. Single-leg (entry-only)
        fee accounting applies only when it is known and <= HOLD_TO_EXPIRY_MAX_SECONDS; the
        AI's own `hold_to_expiry` flag is recorded but never loosens any threshold.

        `emergency_exit=True` (operator Emergency Flatten only) lifts the slippage bands for
        risk-reducing exits so a wide book cannot trap a position; depth checks still apply.
        """
        ts = now_ms if now_ms is not None else int(time.time() * 1000)
        s = self._settings
        reasons: list[str] = []

        # 0. Classify risk-reducing exits first: they must stay possible after the kill-switch trips
        existing_pos = portfolio.positions.get(intent.ticker)
        is_reducing_exit = (
            existing_pos is not None
            and existing_pos.net_contracts != Decimal("0")
            and intent.count <= abs(existing_pos.net_contracts)
            and (
                (existing_pos.net_contracts > Decimal("0") and intent.side == OrderSide.ASK)
                or (existing_pos.net_contracts < Decimal("0") and intent.side == OrderSide.BID)
            )
        )

        # 1. Kill-switch & daily drawdown check (trips for everyone; blocks only new risk)
        self.daily_loss = max(Decimal("0.00"), -portfolio.daily_pnl_dollars)
        if portfolio.kill_switch_tripped or self.kill_switch_tripped:
            self.kill_switch_tripped = True
            portfolio.kill_switch_tripped = True
            if not is_reducing_exit:
                reasons.append("Kill-switch is tripped")
        if portfolio.daily_pnl_dollars <= -s.daily_drawdown_kill_switch:
            self.kill_switch_tripped = True
            portfolio.kill_switch_tripped = True
            if not is_reducing_exit:
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
        if not is_reducing_exit and len(recent_orders) >= s.max_orders_per_minute:
            reasons.append(
                f"Order rate limit reached ({len(recent_orders)}/{s.max_orders_per_minute} per min)"
            )

        active_tickers = {
            k for k, pos in portfolio.positions.items() if pos.net_contracts != Decimal("0")
        } | set(portfolio.resting_order_tickers)

        if not is_reducing_exit and intent.ticker not in active_tickers and len(active_tickers) >= s.max_open_markets:
            reasons.append(
                f"Max open markets reached ({len(active_tickers)}/{s.max_open_markets})"
            )

        if not is_reducing_exit:
            target_hour_key = extract_underlying_expiry_hour(intent.ticker)
            if target_hour_key is not None:
                for active_t in sorted(active_tickers):
                    if (
                        active_t != intent.ticker
                        and extract_underlying_expiry_hour(active_t) == target_hour_key
                    ):
                        reasons.append(
                            f"Correlated underlying + expiry_hour position already active on {active_t} ({target_hour_key}); stacking correlated strikes blocked"
                        )
                        break

        if intent.ticker in portfolio.resting_order_tickers:
            reasons.append(
                f"Active resting order already exists on {intent.ticker} (1 order per ticker limit)"
            )
        elif (
            existing_pos is not None
            and existing_pos.net_contracts != Decimal("0")
            and not is_reducing_exit
        ):
            reasons.append(
                f"Active position already held on {intent.ticker} ({existing_pos.net_contracts} contracts); stacking entries blocked"
            )

        existing_notional = (
            existing_pos.cost_basis_dollars if existing_pos else Decimal("0.00")
        )
        unit_collateral = (
            intent.limit_price if intent.side == OrderSide.BID else (ONE - intent.limit_price)
        )
        proposed_notional = (intent.count * unit_collateral).quantize(Decimal("0.01"))
        effective_max_pos = self.effective_max_position_dollars(portfolio)
        if not is_reducing_exit and existing_notional + proposed_notional > effective_max_pos:
            reasons.append(
                f"Market position limit exceeded (${existing_notional + proposed_notional:.2f} > ${effective_max_pos:.2f})"
            )

        # 4. Orderbook depth, price-band [0.15, 0.85], slippage, and fee-edge filter
        est_fill_price = intent.limit_price
        est_slippage = Decimal("0.00")
        entry_fee = calculate_taker_fee(intent.count, intent.limit_price)
        round_trip_fee = entry_fee * Decimal("2")
        slip_limit_cents = s.exit_max_slippage_cents if is_reducing_exit else s.max_slippage_cents
        max_slip_dollars = (slip_limit_cents / Decimal("100")).quantize(Decimal("0.0001"))

        if snapshot is None:
            reasons.append(f"No live orderbook snapshot available for {intent.ticker}")
        elif snapshot.best_yes_bid is None or snapshot.best_yes_ask is None:
            reasons.append(
                f"Orderbook for {intent.ticker} is one-sided or empty (bid={snapshot.best_yes_bid}, ask={snapshot.best_yes_ask})"
            )
        else:
            skip_slippage_bands = emergency_exit and is_reducing_exit
            if skip_slippage_bands:
                pass
            elif intent.side == OrderSide.BID and (
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

                if (
                    not skip_slippage_bands
                    and walk.slippage_cents_per_contract > slip_limit_cents
                ):
                    reasons.append(
                        f"Estimated slippage {walk.slippage_cents_per_contract}c exceeds max {slip_limit_cents}c"
                    )

                if not is_reducing_exit:
                    hold_to_expiry_eligible = (
                        seconds_to_expiry is not None
                        and 0.0 < seconds_to_expiry <= HOLD_TO_EXPIRY_MAX_SECONDS
                    )
                    primary_fee = entry_fee if hold_to_expiry_eligible else round_trip_fee

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

                    # Strict per-contract units: expected_edge_dollars is always the order total
                    # (TradeIntent contract); MIN_NET_EDGE_DOLLARS is always per contract.
                    if intent.expected_edge_per_contract is not None:
                        gross_edge_per_c = intent.expected_edge_per_contract
                    else:
                        gross_edge_per_c = (intent.expected_edge_dollars / intent.count).quantize(
                            Decimal("0.0001")
                        )
                    min_net_edge_per_c = (
                        min(s.min_net_edge_dollars, HOLD_TO_EXPIRY_MIN_NET_EDGE)
                        if hold_to_expiry_eligible
                        else s.min_net_edge_dollars
                    )

                    fee_per_c = (primary_fee / intent.count).quantize(Decimal("0.0001"))
                    slip_per_c = (est_slippage / intent.count).quantize(Decimal("0.0001"))
                    friction_per_c = fee_per_c + slip_per_c
                    net_edge_per_c = gross_edge_per_c - friction_per_c

                    if friction_per_c >= gross_edge_per_c:
                        reasons.append(
                            f"Fee + slippage (${friction_per_c:.4f}/c) >= expected edge (${gross_edge_per_c:.4f}/c)"
                        )
                    elif net_edge_per_c < min_net_edge_per_c:
                        reasons.append(
                            f"Net edge (${net_edge_per_c:.4f}) below minimum (${min_net_edge_per_c:.4f})"
                        )

        # Enforce 0.15 <= effective_price <= 0.85 on both YES and NO sides for new entries
        effective_fill_price = (
            est_fill_price if intent.side == OrderSide.BID else (ONE - est_fill_price)
        )
        effective_limit_price = (
            intent.limit_price if intent.side == OrderSide.BID else (ONE - intent.limit_price)
        )
        if not is_reducing_exit and not (
            Decimal("0.15") <= est_fill_price <= Decimal("0.85")
            and Decimal("0.15") <= (ONE - est_fill_price) <= Decimal("0.85")
            and Decimal("0.15") <= intent.limit_price <= Decimal("0.85")
            and Decimal("0.15") <= (ONE - intent.limit_price) <= Decimal("0.85")
        ):
            reasons.append(
                f"Effective price ${effective_fill_price:.4f} (YES ${est_fill_price:.4f}, limit ${effective_limit_price:.4f}) outside allowed [$0.15, $0.85] band"
            )

        # 5. Hard Equity Floor (EQUITY_FLOOR)
        proposed_collateral = Decimal("0.00") if is_reducing_exit else (proposed_notional + entry_fee)
        worst_case_equity_after = (
            portfolio.current_equity_dollars
            - portfolio.open_risk_collateral_dollars
            - proposed_collateral
        ).quantize(Decimal("0.01"))

        if not is_reducing_exit and worst_case_equity_after < s.equity_floor:
            reasons.append(
                f"Equity floor breach (worst-case ${worst_case_equity_after:.2f} < floor ${s.equity_floor:.2f})"
            )

        approved = len(reasons) == 0
        logger.info(
            "[RISK_GATE] ticker=%s approved=%s notional=$%.2f effective_max_pos=$%.2f (dynamic=%s, risk_pct=%.0f%%)",
            intent.ticker,
            approved,
            proposed_notional,
            effective_max_pos,
            s.dynamic_sizing_enabled,
            float(s.house_money_risk_pct) * 100.0,
        )

        return RiskEvaluation(
            intent_id=intent.intent_id,
            timestamp_ms=ts,
            ticker=intent.ticker,
            approved=approved,
            rejection_reasons=reasons,
            estimated_fill_price=est_fill_price,
            estimated_slippage_dollars=est_slippage,
            estimated_entry_fee_dollars=entry_fee,
            estimated_round_trip_fee_dollars=round_trip_fee,
            worst_case_equity_after=worst_case_equity_after,
        )
