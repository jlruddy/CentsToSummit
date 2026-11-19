"""Unified asyncio orchestrator running FastAPI/Uvicorn + TradingEngine + graceful shutdown."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import uvicorn

from kalshi_poc.auth import KalshiSigner
from kalshi_poc.client import KalshiApiError, KalshiRestClient, KalshiWebSocketClient
from kalshi_poc.config import AppSettings, ConfigManager
from kalshi_poc.models import (
    CLIENT_ORDER_ID_PREFIX,
    Environment,
    ExecutionMode,
    ExecutionRecord,
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    Position,
    RiskEvaluation,
    StrategyProvider,
    TradeIntent,
    V2CreateOrderRequest,
)
from kalshi_poc.risk_gate import RiskGate
from kalshi_poc.simulator import (
    ShadowSimulator,
    calculate_maker_fee,
    calculate_taker_fee,
    walk_book,
)
from kalshi_poc.strategy_ai import BaseStrategy, StrategyFactory
from kalshi_poc.telemetry import TelemetryManager, export_sqlite_to_parquet
from kalshi_poc.web_server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("kalshi_poc")

EXIT_RETRY_INTERVAL_SEC = 3.0


def _to_dollars(raw: Any, *, cents: bool) -> Decimal:
    val = Decimal(str(raw))
    return val / Decimal("100") if cents else val


def order_fee_total(order_obj: dict[str, Any], filled_qty: Decimal, fill_price: Decimal) -> Decimal:
    """Total fee in dollars actually paid on an order's fills.

    Prefers Kalshi's order-total fee fields (taker_fees/maker_fees, cents; *_dollars, dollars).
    `average_fee_paid*` is a per-contract average, so it is scaled by the filled quantity; it
    was previously recorded as the order total, under-counting fees by ~the contract count.
    """
    if filled_qty <= Decimal("0"):
        return Decimal("0.00")
    total: Decimal | None = None
    for key, cents in (
        ("taker_fees_dollars", False),
        ("maker_fees_dollars", False),
        ("taker_fees", True),
        ("maker_fees", True),
    ):
        raw = order_obj.get(key)
        if raw is None:
            continue
        # Dollar and cent variants of the same field describe the same fee; take one of each kind
        if not cents or (key.replace("_fees", "_fees_dollars") not in order_obj):
            total = (total or Decimal("0")) + _to_dollars(raw, cents=cents)
    if total is not None:
        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    avg_raw = order_obj.get("average_fee_paid_dollars")
    avg = _to_dollars(avg_raw, cents=False) if avg_raw is not None else None
    if avg is None and order_obj.get("average_fee_paid") is not None:
        avg = Decimal(str(order_obj["average_fee_paid"]))
        if avg > Decimal("1"):
            avg = avg / Decimal("100")  # legacy integer-cents encoding
    if avg is not None:
        return (avg * filled_qty).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return calculate_taker_fee(filled_qty, fill_price)


class TradingEngine:
    """Asyncio trading engine strictly separating AI decisions from deterministic RiskGate."""

    def __init__(
        self,
        config_manager: ConfigManager,
        telemetry: TelemetryManager,
    ) -> None:
        self.config_manager = config_manager
        self.telemetry = telemetry
        s = config_manager.settings

        self.risk_gate = RiskGate(s)
        self.strategy: BaseStrategy = StrategyFactory.create(s)
        self.simulator = ShadowSimulator()
        self.portfolio = PortfolioState(
            cash_balance_dollars=s.initial_capital,
            starting_capital_dollars=s.initial_capital,
            daily_start_equity_dollars=s.initial_capital,
        )
        self.equity_history: deque[dict[str, Any]] = deque(maxlen=200)

        self.rest_client: KalshiRestClient | None = None
        self.ws_client: KalshiWebSocketClient | None = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_mode_before_pause = ExecutionMode.SHADOW
        self._market_expiries: dict[str, datetime] = {}
        self._subscribed_at: dict[str, float] = {}
        self._last_discovery_monotonic: float = 0.0
        self._last_eviction_check_monotonic: float = 0.0
        self._last_hold_broadcast_monotonic: float = 0.0
        self._last_portfolio_sync_monotonic: float = 0.0
        self._last_resting_reap_monotonic: float = 0.0
        self._last_quarter_bucket: int = int(time.time()) // 900
        self._last_rollover_03_bucket: int = int(time.time()) // 900
        self._resting_order_first_seen: dict[str, float] = {}
        self._initial_sync_done: bool = False
        self._funded_shards: set[int] = set()
        # Single-application fill ledger (H8): quantities per order_id from each source
        self._fills_applied: dict[str, Decimal] = {}
        self._fills_rest: dict[str, Decimal] = {}
        self._fills_ws: dict[str, Decimal] = {}
        self._seen_trade_ids: set[str] = set()
        self._seen_trade_order: deque[str] = deque(maxlen=20_000)
        self._own_order_ids: set[str] = set()
        # C4/H6: exit management while paused/killed, exit throttling, persisted risk state
        self._exit_mode_when_paused: ExecutionMode | None = None
        self._last_exit_attempt_mono: dict[str, float] = {}
        # Trailing stop: ticker -> ((avg_entry, direction), peak exit-side price seen)
        self._trail_state: dict[str, tuple[tuple[Decimal, int], Decimal]] = {}
        self._persisted_kill_switch: bool | None = None
        self._baseline_key: tuple[str, str] | None = None
        self._engine_lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()
        self.session_start_ts: int = int(time.time() * 1000)
        self.session_starting_equity: Decimal = s.initial_capital

        config_manager.register_reload_callback(self._on_config_reload)

    @staticmethod
    def _ticker_series_key(ticker: str) -> str:
        raw = ticker.strip().upper().split("-", 1)[0]
        if raw.startswith("KXINX"):
            return "INX"
        if raw.startswith("KXNASDAQ"):
            return "NASDAQ"
        if raw.startswith("KXBTC"):
            return "BTC"
        if raw.startswith("KXETH"):
            return "ETH"
        if raw.startswith("KXSOL"):
            return "SOL"
        if raw.startswith("KXXRP"):
            return "XRP"
        return raw

    def _effective_mode(self, s: AppSettings | None = None) -> ExecutionMode:
        """Active mode, or the last active mode when PAUSED (used to route exits and syncs)."""
        s = s or self.config_manager.settings
        if s.execution_mode != ExecutionMode.PAUSED:
            return s.execution_mode
        return self._last_mode_before_pause

    def _is_exchange_scope(self, s: AppSettings | None = None) -> bool:
        return self._effective_mode(s) in (ExecutionMode.DEMO, ExecutionMode.LIVE)

    def _apply_order_fill(
        self,
        order_id: str,
        ticker: str,
        side: OrderSide,
        qty: Decimal,
        price: Decimal,
        *,
        source: str,
        fee_total: Decimal | None,
        ts_ms: int,
        is_maker: bool = False,
    ) -> Decimal:
        """Apply only the not-yet-applied fill quantity for `order_id` (REST or WS, any order).

        REST reports a cumulative fill count for the order; WS reports one trade at a time.
        The portfolio is moved to max(REST cumulative, WS sum), so each contract counts once.
        Returns the quantity newly applied.
        """
        if qty <= Decimal("0"):
            return Decimal("0")
        if source == "rest":
            self._fills_rest[order_id] = max(self._fills_rest.get(order_id, Decimal("0")), qty)
        else:
            self._fills_ws[order_id] = self._fills_ws.get(order_id, Decimal("0")) + qty
        target = max(
            self._fills_rest.get(order_id, Decimal("0")),
            self._fills_ws.get(order_id, Decimal("0")),
        )
        delta = target - self._fills_applied.get(order_id, Decimal("0"))
        if delta <= Decimal("0"):
            return Decimal("0")
        if fee_total is not None:
            fee = (fee_total * delta / qty).quantize(Decimal("0.01")) if delta != qty else fee_total
        else:
            fee = (
                calculate_maker_fee(delta, price) if is_maker else calculate_taker_fee(delta, price)
            )
        self.simulator._apply_fill_to_portfolio(
            self.portfolio, ticker, side, delta, price, fee, ts_ms
        )
        self._fills_applied[order_id] = target
        return delta

    @staticmethod
    def _fill_direction(msg: dict[str, Any]) -> OrderSide:
        """Map Kalshi fill (side, action) to the YES-leg direction of the trade."""
        side_raw = str(msg.get("side") or "yes").lower()
        action = str(msg.get("action") or "buy").lower()
        if side_raw in ("bid", "ask"):
            return OrderSide(side_raw)
        buys_yes = (side_raw == "yes") == (action == "buy")
        return OrderSide.BID if buys_yes else OrderSide.ASK

    @staticmethod
    def _fill_yes_price(msg: dict[str, Any]) -> Decimal:
        raw = msg.get("yes_price_dollars") or msg.get("price_dollars") or msg.get("yes_price")
        if raw is None:
            no_raw = msg.get("no_price_dollars") or msg.get("no_price")
            if no_raw is not None:
                no_px = Decimal(str(no_raw))
                if no_px > Decimal("1"):
                    no_px = no_px / Decimal("100")
                return (Decimal("1.00") - no_px).quantize(Decimal("0.0001"))
            raw = "0.50"
        px = Decimal(str(raw))
        if px > Decimal("1"):
            px = px / Decimal("100")
        return px.quantize(Decimal("0.0001"))

    def _risk_scope(self, s: AppSettings | None = None) -> str:
        """Daily-baseline scope: shadow capital and exchange capital are tracked separately."""
        s = s or self.config_manager.settings
        if self._effective_mode(s) == ExecutionMode.SHADOW:
            return "shadow"
        return f"exchange:{s.kalshi_environment.value}"

    async def _persist_kill_state(self, tripped: bool) -> None:
        await self.telemetry.save_state(
            "kill_switch", {"tripped": tripped, "ts_ms": int(time.time() * 1000)}
        )
        self._persisted_kill_switch = tripped

    async def _persist_kill_state_if_changed(self) -> None:
        if self._persisted_kill_switch != self.portfolio.kill_switch_tripped:
            await self._persist_kill_state(self.portfolio.kill_switch_tripped)

    async def _save_daily_baseline(self, scope: str, day: str) -> None:
        await self.telemetry.save_state(
            f"daily_baseline:{scope}",
            {"date": day, "equity": str(self.portfolio.daily_start_equity_dollars)},
        )
        self._baseline_key = (day, scope)

    async def _load_persisted_state(self) -> None:
        """Restore kill-switch and last active mode so a restart never silently resets risk state."""
        kill = await self.telemetry.load_state("kill_switch")
        tripped = bool(kill and kill.get("tripped"))
        if tripped:
            self.portfolio.kill_switch_tripped = True
            self.risk_gate.kill_switch_tripped = True
            logger.warning("[RISK] Kill-switch restored as TRIPPED from persisted state; press Start to clear.")
        self._persisted_kill_switch = tripped

        s = self.config_manager.settings
        last = await self.telemetry.load_state("last_active_mode")
        if last and s.execution_mode == ExecutionMode.PAUSED:
            try:
                mode = ExecutionMode(str(last.get("mode")))
            except ValueError:
                mode = ExecutionMode.PAUSED
            if mode != ExecutionMode.PAUSED:
                self._last_mode_before_pause = mode

    async def _ensure_daily_baseline(self, s: AppSettings, now_utc: datetime) -> None:
        """Keep one daily drawdown baseline per UTC day and scope, surviving restarts."""
        day = now_utc.date().isoformat()
        scope = self._risk_scope(s)
        if self._baseline_key == (day, scope):
            return
        if (
            scope != "shadow"
            and self.rest_client is not None
            and self.rest_client.signer is not None
            and not self._initial_sync_done
        ):
            return  # wait for the real exchange equity before fixing the baseline
        stored = await self.telemetry.load_state(f"daily_baseline:{scope}")
        if stored and stored.get("date") == day:
            self.portfolio.daily_start_equity_dollars = Decimal(str(stored["equity"]))
            self._baseline_key = (day, scope)
            return
        self.portfolio.daily_start_equity_dollars = self.portfolio.current_equity_dollars
        await self._save_daily_baseline(scope, day)
        logger.info(
            "[RISK] Daily drawdown baseline set to $%s for %s (%s UTC)",
            self.portfolio.daily_start_equity_dollars,
            scope,
            day,
        )

    def _build_signer(self, s: AppSettings) -> KalshiSigner | None:
        if not s.kalshi_api_key_id.strip():
            return None
        key_path = s.resolved_private_key_path
        if not key_path.exists():
            return None
        try:
            return KalshiSigner.from_path(s.kalshi_api_key_id, key_path)
        except Exception as exc:
            logger.warning("Could not load Kalshi private key: %s", exc)
            return None

    def _on_config_reload(self, old_s: AppSettings, new_s: AppSettings) -> None:
        """Hot-reload RiskGate, Strategy, and Portfolio without dropping active WebSocket feeds."""
        logger.info("Hot-reloading RiskGate and Strategy parameters...")
        self.risk_gate.update_settings(new_s)
        if old_s.strategy_provider != new_s.strategy_provider:
            self.strategy = StrategyFactory.create(new_s)
        else:
            # Keep rolling mid-price buffers, LLM cooldown and single-flight lock across saves
            self.strategy.settings = new_s

        env_changed = old_s.kalshi_environment != new_s.kalshi_environment
        creds_changed = (
            old_s.kalshi_api_key_id != new_s.kalshi_api_key_id
            or old_s.kalshi_private_key_path != new_s.kalshi_private_key_path
        )
        mode_changed = old_s.execution_mode != new_s.execution_mode
        capital_changed = old_s.initial_capital != new_s.initial_capital
        floor_changed = old_s.equity_floor != new_s.equity_floor

        if self.rest_client is not None:
            self.rest_client.base_url = new_s.rest_base_url
            if creds_changed or env_changed:
                self.rest_client.signer = self._build_signer(new_s)

        no_trades_yet = (
            not self.portfolio.positions and not self.portfolio.recent_order_timestamps_ms
        )
        if capital_changed and (
            new_s.execution_mode in (ExecutionMode.SHADOW, ExecutionMode.PAUSED)
            or no_trades_yet
        ):
            self.portfolio.cash_balance_dollars = new_s.initial_capital
            self.portfolio.starting_capital_dollars = new_s.initial_capital
            self.portfolio.daily_start_equity_dollars = new_s.initial_capital
            capital_reset = True
        else:
            capital_reset = False

        if new_s.execution_mode != ExecutionMode.PAUSED:
            self._last_mode_before_pause = new_s.execution_mode
            self._exit_mode_when_paused = None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            if new_s.execution_mode != ExecutionMode.PAUSED and mode_changed:
                loop.create_task(
                    self.telemetry.save_state(
                        "last_active_mode", {"mode": new_s.execution_mode.value}
                    )
                )
            if capital_reset:
                loop.create_task(
                    self._save_daily_baseline(
                        self._risk_scope(new_s), datetime.now(timezone.utc).date().isoformat()
                    )
                )
            should_sync_api_balance = (
                new_s.execution_mode in (ExecutionMode.DEMO, ExecutionMode.LIVE)
                and (mode_changed or creds_changed or env_changed)
            ) or (creds_changed and new_s.execution_mode != ExecutionMode.SHADOW)

            if should_sync_api_balance:
                loop.create_task(self.sync_balance_from_api())
            elif capital_changed or floor_changed or mode_changed:
                loop.create_task(self._broadcast_equity_and_status())

            if env_changed and self.ws_client is not None:
                loop.create_task(self._restart_ws(new_s))
            elif self.ws_client is not None and (
                old_s.subscribed_tickers != new_s.subscribed_tickers
                or old_s.allowed_series_prefixes != new_s.allowed_series_prefixes
                or old_s.prefer_15m_turnover != new_s.prefer_15m_turnover
            ):
                loop.create_task(self._refresh_subscriptions(new_s))

    async def _broadcast_equity_and_status(self) -> None:
        s = self.config_manager.settings
        ts_ms = int(time.time() * 1000)
        await self.telemetry.log_equity(self.portfolio, str(s.equity_floor), ts_ms)
        await self.telemetry.broadcast("STATUS", self.get_status())

    async def _reap_stale_resting_orders(
        self, now_utc: datetime | None = None, now_mono: float | None = None
    ) -> None:
        """Poll resting orders every 10s and cancel any resting order older than 20s."""
        if self.rest_client is None or self.rest_client.signer is None:
            return
        now = now_utc or datetime.now(timezone.utc)
        mono = now_mono if now_mono is not None else time.monotonic()
        self._last_resting_reap_monotonic = mono
        try:
            resting = await self.rest_client.get_resting_orders()
        except Exception as exc:
            logger.debug("Resting orders poll failed: %s", exc)
            return

        seen_ids: set[str] = set()
        active_tickers: set[str] = set()
        resting_collateral = Decimal("0.00")

        for o in resting:
            if not isinstance(o, dict):
                continue
            oid = str(o.get("order_id") or "")
            ticker = str(o.get("ticker") or o.get("market_ticker") or "")
            if oid:
                seen_ids.add(oid)
            first_seen = self._resting_order_first_seen.setdefault(oid or ticker, mono)
            age_sec = mono - first_seen

            created_raw = o.get("created_time") or o.get("last_update_time")
            if isinstance(created_raw, str) and created_raw:
                try:
                    cdt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                    if cdt.tzinfo is None:
                        cdt = cdt.replace(tzinfo=timezone.utc)
                    age_sec = max(age_sec, (now - cdt).total_seconds())
                except Exception:
                    pass

            client_oid = str(o.get("client_order_id") or "")
            is_own = client_oid.startswith(CLIENT_ORDER_ID_PREFIX) or oid in self._own_order_ids
            if age_sec > 20.0 and oid and is_own:
                try:
                    await self.rest_client.cancel_order(oid, market_ticker=ticker or None)
                    logger.info(
                        "[REAPER] Canceled stale resting order %s on %s (age=%.1fs)",
                        oid,
                        ticker,
                        age_sec,
                    )
                    self._resting_order_first_seen.pop(oid, None)
                    continue
                except Exception as exc:
                    logger.warning(
                        "[REAPER] Failed to cancel stale resting order %s on %s: %s",
                        oid,
                        ticker,
                        exc,
                    )

            if ticker:
                active_tickers.add(ticker)
            rem_raw = o.get("remaining_count_fp") or o.get("remaining_count") or o.get("count") or "0"
            px_raw = (
                o.get("yes_price_dollars")
                or o.get("price_dollars")
                or o.get("price")
                or o.get("yes_price")
                or "0.50"
            )
            try:
                rem_cnt = Decimal(str(rem_raw))
                px = Decimal(str(px_raw))
                if px > Decimal("1"):
                    px = px / Decimal("100")
                side_str = str(o.get("side") or o.get("action") or "bid").lower()
                unit_col = px if side_str in ("yes", "bid", "buy") else (Decimal("1.00") - px)
                resting_collateral += (rem_cnt * max(Decimal("0.01"), unit_col)).quantize(
                    Decimal("0.01")
                )
            except Exception:
                pass

        for k in list(self._resting_order_first_seen.keys()):
            if k not in seen_ids and k not in active_tickers:
                self._resting_order_first_seen.pop(k, None)

        self.portfolio.resting_order_tickers = active_tickers
        self.portfolio.resting_order_collateral_dollars = resting_collateral

    async def sync_balance_from_api(self) -> None:
        """Sync portfolio cash_balance and active non-expired positions from Kalshi API."""
        if self.rest_client is None or self.rest_client.signer is None or self._sync_lock.locked():
            return
        if not self._is_exchange_scope():
            # M2: shadow capital/positions are never overwritten by the real exchange account
            return
        async with self._sync_lock:
            await self._sync_balance_from_api_locked()

    async def _sync_balance_from_api_locked(self) -> None:
        assert self.rest_client is not None
        now_utc = datetime.now(timezone.utc)
        now_mono = time.monotonic()
        self._last_portfolio_sync_monotonic = now_mono
        try:
            bal, _ = await self.rest_client.get_balance()
            self.portfolio.cash_balance_dollars = bal.balance_dollars

            # Rebuild active open_positions strictly from non-zero, non-expired Kalshi positions
            try:
                open_pos_raw = await self.rest_client.get_open_positions()
                rebuilt_positions: dict[str, Position] = {}
                for p in open_pos_raw:
                    ticker = str(p.get("ticker") or p.get("market_ticker") or "")
                    if not ticker:
                        continue
                    raw_cnt = (
                        p.get("position_fp")
                        if p.get("position_fp") is not None
                        else p.get("position", 0)
                    )
                    cnt = Decimal(str(raw_cnt))
                    if cnt == Decimal("0"):
                        continue

                    exp_cached = self._market_expiries.get(ticker)
                    if exp_cached is not None and exp_cached <= now_utc:
                        continue
                    is_open, exp_dt = await self.rest_client.is_market_open_and_unexpired(
                        ticker, now_utc=now_utc
                    )
                    if exp_dt is not None:
                        self._market_expiries[ticker] = exp_dt
                    if not is_open or (exp_dt is not None and exp_dt <= now_utc):
                        continue

                    exp_val_raw = p.get("market_exposure_dollars")
                    if exp_val_raw is None and p.get("market_exposure") is not None:
                        exp_val_raw = Decimal(str(p["market_exposure"])) / Decimal("100")
                    cost_basis = (
                        Decimal(str(exp_val_raw)).quantize(Decimal("0.01"))
                        if exp_val_raw is not None
                        else (abs(cnt) * Decimal("0.50")).quantize(Decimal("0.01"))
                    )
                    unit_cost = (
                        (cost_basis / abs(cnt)).quantize(Decimal("0.0001"))
                        if abs(cnt) > Decimal("0")
                        else Decimal("0.5000")
                    )
                    avg_entry = (
                        min(Decimal("0.9900"), max(Decimal("0.0100"), unit_cost))
                        if cnt > Decimal("0")
                        else min(
                            Decimal("0.9900"),
                            max(Decimal("0.0100"), Decimal("1.0000") - unit_cost),
                        )
                    )
                    rpnl_raw = p.get("realized_pnl_dollars")
                    if rpnl_raw is None and p.get("realized_pnl") is not None:
                        rpnl_raw = Decimal(str(p["realized_pnl"])) / Decimal("100")
                    rpnl = (
                        Decimal(str(rpnl_raw)).quantize(Decimal("0.01"))
                        if rpnl_raw is not None
                        else Decimal("0.00")
                    )
                    rebuilt_positions[ticker] = Position(
                        ticker=ticker,
                        net_contracts=cnt,
                        avg_entry_price=avg_entry,
                        cost_basis_dollars=cost_basis,
                        realized_pnl_dollars=rpnl,
                    )
                    logger.info(
                        "[PORTFOLIO] Open position: ticker=%s position=%s exposure=$%s close_in=%ss",
                        ticker,
                        cnt,
                        cost_basis,
                        f"{(exp_dt - now_utc).total_seconds():.0f}" if exp_dt else "open",
                    )
                s = self.config_manager.settings
                if s.execution_mode != ExecutionMode.SHADOW:
                    self.portfolio.positions = rebuilt_positions
                if not rebuilt_positions:
                    logger.info("[PORTFOLIO] Open positions: NONE (0 active non-expired positions)")
            except Exception as pos_exc:
                logger.debug("Open positions sync skipped/failed: %s", pos_exc)

            await self._reap_stale_resting_orders(now_utc=now_utc, now_mono=now_mono)
            if not self.portfolio.resting_order_tickers:
                logger.info("[PORTFOLIO] Resting orders: NONE (0 resting orders)")

            if not self._initial_sync_done:
                self._initial_sync_done = True
                self.portfolio.starting_capital_dollars = self.portfolio.current_equity_dollars
                # Fix today's drawdown baseline from real exchange equity before any evaluation
                await self._ensure_daily_baseline(self.config_manager.settings, now_utc)

            logger.info(
                "Synced Kalshi balance: $%s (equity=$%s, open_markets=%d, resting_orders=%d, shards=%s)",
                bal.balance_dollars,
                f"{self.portfolio.current_equity_dollars:.2f}",
                len(self.portfolio.positions),
                len(self.portfolio.resting_order_tickers),
                bal.balance_breakdown or {},
            )
            await self._broadcast_equity_and_status()
        except Exception as exc:
            logger.warning("Balance sync failed: %s", exc)

    async def _refresh_subscriptions(self, s: AppSettings) -> None:
        if self.ws_client is None:
            return
        discovered = await self._discover_tickers(s)
        merged: list[str] = []
        seen_series: set[str] = set()
        seen_prefixes: set[str] = set()

        def _add_candidate(t: str, *, require_distinct_series: bool = True) -> None:
            if len(merged) >= 6 or t in merged:
                return
            pref = t.strip().upper().split("-", 1)[0]
            sk = self._ticker_series_key(t)
            if pref in seen_prefixes:
                return
            if require_distinct_series and sk in seen_series:
                return
            seen_prefixes.add(pref)
            seen_series.add(sk)
            merged.append(t)

        if s.prefer_15m_turnover:
            # Bind fresh 15M contracts first (displacing temporary hourly rollover fallbacks)
            for t in discovered:
                if t.strip().upper().split("-", 1)[0].endswith("15M"):
                    _add_candidate(t, require_distinct_series=True)
            for t in self.ws_client.tickers:
                if t.strip().upper().split("-", 1)[0].endswith("15M"):
                    _add_candidate(t, require_distinct_series=True)

        # Keep existing healthy subscriptions, at most 1 per series_ticker
        for t in self.ws_client.tickers:
            _add_candidate(t, require_distinct_series=True)

        # Pass 1: Add discovered tickers from distinct underlying assets
        for t in discovered:
            _add_candidate(t, require_distinct_series=True)

        # Pass 2: Add discovered tickers from distinct series_tickers (e.g. KXBTCD alongside KXBTC15M)
        for t in discovered:
            _add_candidate(t, require_distinct_series=False)
        for t in self.ws_client.tickers:
            _add_candidate(t, require_distinct_series=False)

        # Pass 3: Only if < 6 distinct series_tickers exist, fill remaining slots
        for t in discovered:
            if len(merged) >= 6:
                break
            if t not in merged:
                merged.append(t)

        # Always retain subscriptions for open positions until settlement or exit
        for t in list(self.portfolio.positions.keys()):
            if t not in merged:
                merged.append(t)

        displaced = [
            t
            for t in self.ws_client.tickers
            if t not in merged and t not in self.portfolio.positions
        ]
        if displaced:
            await self.ws_client.unsubscribe_tickers(displaced)
            for t in displaced:
                self._subscribed_at.pop(t, None)

        now_mono = time.monotonic()
        for t in merged:
            if t not in self._subscribed_at:
                self._subscribed_at[t] = now_mono
        if merged:
            await self._seed_orderbooks(merged)
            await self.ws_client.update_subscriptions(merged)

    async def _restart_ws(self, s: AppSettings) -> None:
        if self.ws_client is not None:
            await self.ws_client.stop()
            self.ws_client.orderbooks.clear()
            self.ws_client.ws_url = s.ws_url
            self.ws_client.signer = self._build_signer(s)
            tickers = await self._discover_tickers(s)
            await self._seed_orderbooks(tickers)
            self.ws_client.start(tickers)

    async def start(self) -> None:
        s = self.config_manager.settings
        signer = self._build_signer(s)
        self.rest_client = KalshiRestClient(base_url=s.rest_base_url, signer=signer)
        self.ws_client = KalshiWebSocketClient(
            ws_url=s.ws_url,
            signer=signer,
            on_orderbook=self._on_orderbook,
            on_fill=self._on_fill,
        )

        # Restore persisted kill-switch / last active mode; sync starting balance from API if configured
        await self._load_persisted_state()
        if signer is not None:
            await self.sync_balance_from_api()

        tickers = await self._discover_tickers(s)
        logger.info("Subscribing to Kalshi WS tickers: %s", tickers)
        now_mono = time.monotonic()
        for t in tickers:
            self._subscribed_at[t] = now_mono
        await self._seed_orderbooks(tickers)
        self.ws_client.start(tickers)

        await self._ensure_single_engine_loop()

    async def _ensure_single_engine_loop(self) -> None:
        """Cancel any existing strategy evaluation loop before starting a single loop task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._running = True
        self._task = asyncio.create_task(self._engine_loop(), name="trading-engine")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.ws_client is not None:
            await self.ws_client.stop()
        if self.rest_client is not None:
            await self.rest_client.close()

    async def _seed_orderbooks(self, tickers: list[str]) -> None:
        """Seed initial orderbook snapshots via REST only when missing or stale (> 10s)."""
        if self.rest_client is None or self.ws_client is None:
            return
        now_ms = int(time.time() * 1000)
        fetched = 0
        for ticker in tickers:
            existing = self.ws_client.orderbooks.get(ticker)
            if existing is not None and (now_ms - existing.timestamp_ms) <= 10_000:
                continue
            if fetched > 0:
                await asyncio.sleep(0.065)
            try:
                snap = await self.rest_client.get_orderbook(ticker)
                self.ws_client.set_snapshot(snap)
                await self.telemetry.log_orderbook(snap)
                fetched += 1
            except Exception as exc:
                logger.debug("REST orderbook seed failed for %s: %s", ticker, exc)

    def _deployable_budget(self, s: AppSettings) -> Decimal:
        return max(
            Decimal("0.00"),
            self.portfolio.current_equity_dollars
            - s.equity_floor
            - self.portfolio.open_risk_collateral_dollars,
        )

    async def _ensure_shard_funded(
        self, ticker: str, s: AppSettings, min_required: Decimal | None = None
    ) -> None:
        if (
            self.rest_client is None
            or self.rest_client.signer is None
            or s.execution_mode not in (ExecutionMode.DEMO, ExecutionMode.LIVE)
        ):
            return
        shard = await self.rest_client.get_market_exchange_index(ticker)
        eff_max_pos = self.risk_gate.effective_max_position_dollars(self.portfolio)
        req = (
            min_required
            if min_required is not None
            else (eff_max_pos + Decimal("0.50"))
        )

        other_shards = [
            (idx, bal)
            for idx, bal in self.rest_client.shard_balances.items()
            if idx != shard and bal > Decimal("0")
        ]
        if other_shards:
            target_bal = self.rest_client.shard_balances.get(shard, Decimal("0.00"))
            if target_bal >= req:
                self._funded_shards.add(shard)
                return
            other_shards.sort(key=lambda x: x[1], reverse=True)
            donor_shard, donor_bal = other_shards[0]
            shortfall = max(Decimal("0.01"), (req - target_bal).quantize(Decimal("0.01")))
            desired = max(
                shortfall,
                (eff_max_pos + Decimal("0.50") - target_bal).quantize(Decimal("0.01")),
            )
            avail_donor = max(Decimal("0.00"), donor_bal - Decimal("0.50"))
            transfer_amt = min(
                desired, avail_donor if avail_donor >= shortfall else donor_bal
            ).quantize(Decimal("0.01"))
            if transfer_amt > Decimal("0"):
                try:
                    await self.rest_client.fund_shard(
                        shard, transfer_amt, source_shard=donor_shard
                    )
                    self._funded_shards.add(shard)
                except Exception as exc:
                    logger.warning(
                        "Auto-fund shard %d from shard %d failed: %s",
                        shard,
                        donor_shard,
                        exc,
                    )
            return

        if shard == 0:
            return
        shard_bal = self.rest_client.shard_balances.get(shard, Decimal("0.00"))
        if shard_bal < req or shard not in self._funded_shards:
            budget = self._deployable_budget(s)
            if budget > Decimal("0"):
                try:
                    await self.rest_client.fund_shard(shard, budget)
                    self._funded_shards.add(shard)
                except Exception as exc:
                    logger.warning("Auto-fund shard %d failed: %s", shard, exc)

    async def _discover_tickers(self, s: AppSettings) -> list[str]:
        self._last_discovery_monotonic = time.monotonic()
        if s.subscribed_tickers:
            return s.subscribed_tickers
        if self.rest_client is None:
            return []
        markets = await self.rest_client.discover_active_markets(
            s.allowed_series_prefixes,
            max_tickers=6,
            environment=s.kalshi_environment.value,
            prefer_15m_turnover=s.prefer_15m_turnover,
        )
        # Remove expired entries not held in open positions
        now_utc = datetime.now(timezone.utc)
        for t, exp in list(self._market_expiries.items()):
            if exp <= now_utc and t not in self.portfolio.positions:
                self._market_expiries.pop(t, None)
        for m in markets:
            if m.expiry_dt is not None:
                dt = m.expiry_dt
                self._market_expiries[m.ticker] = (
                    dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
                )
            await self._ensure_shard_funded(m.ticker, s)
        return [m.ticker for m in markets]

    async def _on_orderbook(self, snap: OrderbookSnapshot) -> None:
        await self.telemetry.log_orderbook(snap)

    async def _on_fill(self, msg: dict[str, Any]) -> None:
        """Track live fills from the private WS channel, applying each contract exactly once."""
        logger.info("Received live fill event: %s", msg)
        ts_ms = int(time.time() * 1000)
        ticker = str(msg.get("market_ticker") or msg.get("ticker") or "UNKNOWN")
        trade_id = str(msg.get("trade_id") or "")
        order_id = str(msg.get("order_id") or (f"trade-{trade_id}" if trade_id else ""))
        await self.telemetry.log_exchange_event(
            "fill", msg, ticker=ticker, order_id=order_id, trade_id=trade_id, timestamp_ms=ts_ms
        )

        if not self._is_exchange_scope():
            logger.info("[FILL] Ignoring exchange fill in shadow scope (ticker=%s)", ticker)
            return
        if trade_id:
            if trade_id in self._seen_trade_ids:
                return
            if len(self._seen_trade_order) == self._seen_trade_order.maxlen:
                self._seen_trade_ids.discard(self._seen_trade_order[0])
            self._seen_trade_order.append(trade_id)
            self._seen_trade_ids.add(trade_id)
        if not order_id:
            return

        count = Decimal(str(msg.get("count_fp") or msg.get("count") or "0"))
        price = self._fill_yes_price(msg)
        is_maker = msg.get("is_taker") is False
        applied = self._apply_order_fill(
            order_id,
            ticker,
            self._fill_direction(msg),
            count,
            price,
            source="ws",
            fee_total=None,
            ts_ms=ts_ms,
            is_maker=is_maker,
        )
        if applied > Decimal("0") and self.rest_client is not None:
            await self.rest_client.upgrade_api_tier_once()
            await self.sync_balance_from_api()

    async def _evict_unhealthy_subscriptions(
        self, s: AppSettings, now_utc: datetime, now_mono: float | None = None
    ) -> list[str]:
        """Evict tickers if <120s left, empty book (bid/ask is None), wide spread, or mid off-band."""
        if self.ws_client is None or s.subscribed_tickers:
            return []
        mono = now_mono if now_mono is not None else time.monotonic()
        evicted: list[str] = []
        all_tracked = list(
            dict.fromkeys(list(self.ws_client.tickers) + list(self.ws_client.orderbooks.keys()))
        )
        for ticker in all_tracked:
            if ticker in self.portfolio.positions:
                continue
            exp = self._market_expiries.get(ticker)
            close_in = (exp - now_utc).total_seconds() if exp is not None else None
            snap = self.ws_client.orderbooks.get(ticker)
            bid = snap.best_yes_bid if snap is not None else None
            ask = snap.best_yes_ask if snap is not None else None

            sub_ts = self._subscribed_at.get(ticker, 0.0)
            in_grace = (mono - sub_ts) < 10.0

            reason: str | None = None
            if close_in is not None and close_in < 120.0:
                reason = "expired"
            elif not in_grace:
                if snap is None or bid is None or ask is None:
                    reason = "empty_book"
                elif snap.yes_spread is not None and snap.yes_spread > Decimal("0.08"):
                    reason = "wide_spread"
                elif snap.mid_yes_price is not None and (
                    snap.mid_yes_price < Decimal("0.15")
                    or snap.mid_yes_price > Decimal("0.85")
                ):
                    reason = "off_mid"

            if reason is not None:
                evicted.append(ticker)
                logger.info(
                    "[EVICT] ticker=%s reason=%s (bid=%s, ask=%s, close_in=%ss)",
                    ticker,
                    reason,
                    bid,
                    ask,
                    f"{close_in:.0f}" if close_in is not None else "unknown",
                )

        if evicted:
            await self.ws_client.unsubscribe_tickers(evicted)
            for t in evicted:
                self._subscribed_at.pop(t, None)
                if t not in self.portfolio.positions:
                    self._market_expiries.pop(t, None)
        return evicted

    async def _engine_loop(self) -> None:
        while self._running:
            s = self.config_manager.settings
            try:
                now_utc = datetime.now(timezone.utc)
                now_mono = time.monotonic()
                ts_int = int(now_utc.timestamp())
                quarter_bucket = ts_int // 900
                sec_into_quarter = ts_int % 900
                crossed_settlement_boundary = quarter_bucket != self._last_quarter_bucket
                if crossed_settlement_boundary:
                    self._last_quarter_bucket = quarter_bucket

                rollover_03_trigger = (
                    sec_into_quarter >= 3 and quarter_bucket != self._last_rollover_03_bucket
                )
                if rollover_03_trigger:
                    self._last_rollover_03_bucket = quarter_bucket
                    logger.info(
                        "[ROLLOVER] :03s past 15-minute boundary -> triggering immediate 15M discovery refresh"
                    )

                # Never count expired contracts toward Max Open Markets
                expired_cleared = False
                if self._is_exchange_scope(s) and self.portfolio.positions:
                    for t in list(self.portfolio.positions.keys()):
                        exp = self._market_expiries.get(t)
                        if exp is not None and exp <= now_utc:
                            self.portfolio.positions.pop(t, None)
                            expired_cleared = True

                # Poll resting orders every 10s (reaping orders > 20s old) and sync portfolio every 15s
                # Always sync on settlement boundaries or when expired positions are cleared, even when PAUSED
                if (
                    self.rest_client is not None
                    and self.rest_client.signer is not None
                    and self._is_exchange_scope(s)
                ):
                    if (
                        expired_cleared
                        or crossed_settlement_boundary
                        or rollover_03_trigger
                        or (
                            s.execution_mode in (ExecutionMode.DEMO, ExecutionMode.LIVE)
                            and (now_mono - self._last_portfolio_sync_monotonic) >= 15.0
                        )
                    ):
                        await self.sync_balance_from_api()
                    elif (
                        s.execution_mode in (ExecutionMode.DEMO, ExecutionMode.LIVE)
                        and (now_mono - self._last_resting_reap_monotonic) >= 10.0
                    ):
                        await self._reap_stale_resting_orders(now_utc=now_utc, now_mono=now_mono)

                # Every 30s (or on 15m boundary / :03 rollover), evict unhealthy tickers and refill missing slots up to 6
                evicted: list[str] = []
                maintenance_due = (
                    crossed_settlement_boundary
                    or rollover_03_trigger
                    or (now_mono - self._last_eviction_check_monotonic) >= 30.0
                )
                if maintenance_due:
                    self._last_eviction_check_monotonic = now_mono
                    evicted = await self._evict_unhealthy_subscriptions(s, now_utc, now_mono)

                discovery_cooldown_ok = (
                    rollover_03_trigger
                    or (now_mono - self._last_discovery_monotonic) >= 15.0
                )
                if (
                    self.ws_client is not None
                    and discovery_cooldown_ok
                    and (
                        rollover_03_trigger
                        or evicted
                        or (
                            maintenance_due
                            and (
                                len(self.ws_client.tickers) < 6
                                or any(exp <= now_utc for exp in self._market_expiries.values())
                            )
                        )
                    )
                ):
                    await self._refresh_subscriptions(s)

                await self._ensure_daily_baseline(s, now_utc)

                async with self._engine_lock:
                    if (
                        self._effective_mode(s) == ExecutionMode.SHADOW
                        and self.ws_client is not None
                        and self.portfolio.positions
                    ):
                        settlements = self.simulator.settle_expired_positions(
                            self.portfolio,
                            self._market_expiries,
                            self.ws_client.orderbooks,
                            now_utc=now_utc,
                        )
                        for rec in settlements:
                            self._market_expiries.pop(rec.ticker, None)
                            await self.telemetry.log_execution(rec)

                    # Exit management runs even when the kill-switch is tripped or the engine
                    # was paused mid-session; only new entries are gated on mode + kill-switch.
                    if self.ws_client is not None:
                        await self._run_exit_management(s, now_utc)
                    if (
                        s.execution_mode != ExecutionMode.PAUSED
                        and not self.portfolio.kill_switch_tripped
                        and self.ws_client is not None
                    ):
                        active_snapshots: list[OrderbookSnapshot] = []
                        stale_tickers: list[str] = []
                        for snap in list(self.ws_client.orderbooks.values()):
                            exp = self._market_expiries.get(snap.ticker)
                            if exp is not None and (exp - now_utc).total_seconds() < 120.0:
                                if snap.ticker not in self.portfolio.positions:
                                    stale_tickers.append(snap.ticker)
                                continue
                            if snap.best_yes_bid is None or snap.best_yes_ask is None:
                                continue
                            active_snapshots.append(snap)
                        if stale_tickers:
                            await self.ws_client.unsubscribe_tickers(stale_tickers)
                            for t in stale_tickers:
                                self._subscribed_at.pop(t, None)
                                if t not in self.portfolio.positions:
                                    self._market_expiries.pop(t, None)
                        if active_snapshots:
                            await self._process_batch(
                                active_snapshots[:8], s, execute_orders=True
                            )

                    ts_ms = int(time.time() * 1000)
                    eq_point = {
                        "timestamp_ms": ts_ms,
                        "equity": str(self.portfolio.current_equity_dollars),
                        "floor": str(s.equity_floor),
                    }
                    self.equity_history.append(eq_point)
                    await self.telemetry.log_equity(self.portfolio, str(s.equity_floor), ts_ms)
                    await self.telemetry.broadcast("STATUS", self.get_status())
                await self._persist_kill_state_if_changed()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Error in trading engine loop: %s", exc)

            await asyncio.sleep(s.poll_interval_ms / 1000.0)

    async def _run_exit_management(self, s: AppSettings, now_utc: datetime) -> list[TradeIntent]:
        """Manage open-position exits in the active mode, or the mode active before a pause/kill."""
        if not self.portfolio.positions:
            return []
        if s.execution_mode != ExecutionMode.PAUSED:
            mode = s.execution_mode
        elif self._exit_mode_when_paused is not None:
            mode = self._exit_mode_when_paused
        else:
            return []  # paused since boot: never act on positions until the operator starts
        return await self._evaluate_deterministic_exits(s, now_utc, mode=mode)

    async def _evaluate_deterministic_exits(
        self, s: AppSettings, now_utc: datetime, mode: ExecutionMode | None = None
    ) -> list[TradeIntent]:
        """Evaluate open positions every tick for take-profit, trailing-stop and stop-loss exits.

        All prices are on the position's own side: the YES bid for YES positions and the NO bid
        (1 - YES ask) for NO positions, i.e. the price the position can be sold at right now.
        """
        for t in list(self._trail_state):
            if t not in self.portfolio.positions:
                self._trail_state.pop(t, None)
        if self.ws_client is None or not self.portfolio.positions:
            return []

        submitted: list[TradeIntent] = []
        ts_ms = int(now_utc.timestamp() * 1000)
        for ticker, pos in list(self.portfolio.positions.items()):
            if pos.net_contracts == Decimal("0") or ticker in self.portfolio.resting_order_tickers:
                continue
            exp = self._market_expiries.get(ticker)
            close_in = (exp - now_utc).total_seconds() if exp is not None else None
            if close_in is not None and close_in <= 0.0:
                continue

            snap = self.ws_client.orderbooks.get(ticker)
            if snap is None or snap.best_yes_bid is None or snap.best_yes_ask is None:
                continue

            is_long = pos.net_contracts > Decimal("0")
            if is_long:
                sell_px = snap.best_yes_bid
                entry_px = pos.avg_entry_price
                exit_side = OrderSide.ASK
                raw_exit_px = snap.best_yes_bid
            else:
                sell_px = Decimal("1.00") - snap.best_yes_ask
                entry_px = Decimal("1.00") - pos.avg_entry_price
                exit_side = OrderSide.BID
                raw_exit_px = snap.best_yes_ask
            unrealized_gain = sell_px - entry_px
            drop_below_entry = entry_px - sell_px

            # Trailing stop: track the best sell price since this position was opened
            trail_key = (pos.avg_entry_price, 1 if is_long else -1)
            state = self._trail_state.get(ticker)
            peak = sell_px if state is None or state[0] != trail_key else max(state[1], sell_px)
            self._trail_state[ticker] = (trail_key, peak)

            take_profit = (
                unrealized_gain >= s.take_profit_min_gain and sell_px >= s.take_profit_min_price
            ) or sell_px >= s.take_profit_lock_price
            trailing_stop = (
                s.trailing_stop_enabled
                and (peak - entry_px) >= s.trailing_stop_activation_gain
                and (peak - sell_px) >= s.trailing_stop_distance
            )
            stop_loss = drop_below_entry >= s.stop_loss_drop and (
                close_in is None or close_in > float(s.stop_loss_disable_secs)
            )

            if not (take_profit or trailing_stop or stop_loss):
                continue
            now_mono = time.monotonic()
            last_try = self._last_exit_attempt_mono.get(ticker)
            if last_try is not None and (now_mono - last_try) < EXIT_RETRY_INTERVAL_SEC:
                continue
            self._last_exit_attempt_mono[ticker] = now_mono

            limit_px = min(Decimal("0.9900"), max(Decimal("0.0100"), raw_exit_px))
            walk = walk_book(snap, exit_side, abs(pos.net_contracts), limit_price=limit_px)
            exit_count = (
                walk.fillable_count
                if walk.fillable_count > Decimal("0")
                else abs(pos.net_contracts)
            )
            edge_per_c = max(Decimal("0.0500"), abs(unrealized_gain)).quantize(Decimal("0.0001"))
            total_edge = (edge_per_c * exit_count).quantize(Decimal("0.01"))
            reason_tag = (
                "TAKE_PROFIT_LOCK"
                if take_profit
                else "TRAILING_STOP"
                if trailing_stop
                else "HARD_STOP_LOSS"
            )
            rationale = (
                f"Deterministic {reason_tag}: entry=${entry_px:.4f} sell=${sell_px:.4f} "
                f"peak=${peak:.4f} unrealized={unrealized_gain:+.4f}/c"
            )
            intent = TradeIntent(
                timestamp_ms=ts_ms,
                ticker=ticker,
                side=exit_side,
                count=exit_count,
                limit_price=limit_px,
                expected_edge_dollars=total_edge,
                expected_edge_per_contract=edge_per_c,
                hold_to_expiry=False,
                confidence=Decimal("0.99"),
                inference_latency_ms=0,
                rationale=rationale,
                provider="deterministic_exit",
                reduce_only=True,
            )
            logger.info(
                "[EXIT] %s ticker=%s side=%s count=%s price=$%s unrealized=%+.4f/c",
                reason_tag,
                ticker,
                exit_side.value,
                exit_count,
                limit_px,
                unrealized_gain,
            )
            d = intent.model_dump(mode="json")
            d["action"] = "BUY" if intent.side == OrderSide.BID else "SELL"
            d["prompt_tokens"] = 0
            d["completion_tokens"] = 0
            d["estimated_cost_usd"] = "0.000000"
            d["decision_status"] = "DETERMINISTIC_EXIT"
            await self.telemetry.log_ai_decision(d)
            await self._evaluate_and_execute_intent(
                intent, snap, s, execute_orders=True, mode=mode
            )
            submitted.append(intent)

        return submitted

    async def eval_now(self) -> dict[str, Any]:
        """Force an immediate strategy + RiskGate evaluation across active tickers."""
        s = self.config_manager.settings
        if self.ws_client is not None and not self.ws_client.orderbooks:
            tickers = self.ws_client.tickers or await self._discover_tickers(s)
            if tickers:
                await self._seed_orderbooks(tickers)

        snapshots = list(self.ws_client.orderbooks.values()) if self.ws_client else []
        execute_orders = (
            s.execution_mode != ExecutionMode.PAUSED and not self.portfolio.kill_switch_tripped
        )
        if snapshots:
            # Serialize with the engine loop: two overlapping batches could each approve the
            # same ticker before either order is recorded.
            async with self._engine_lock:
                await self._process_batch(
                    snapshots, s, execute_orders=execute_orders, force=True
                )

        await self._broadcast_equity_and_status()
        return {"evaluated": len(snapshots), "execution_mode": s.execution_mode.value}

    async def _process_batch(
        self,
        snapshots: list[OrderbookSnapshot],
        s: AppSettings,
        execute_orders: bool = True,
        force: bool = False,
    ) -> None:
        batch = await self.strategy.evaluate_batch(
            snapshots,
            self.portfolio,
            positions_by_ticker=self.portfolio.positions,
            force=force,
            market_expiries=self._market_expiries,
        )
        if batch.llm_call is not None:
            await self.telemetry.log_llm_call(batch.llm_call)
        is_gated_hold = not batch.invoked_llm and not batch.intents and not force
        if is_gated_hold:
            now_mono = time.monotonic()
            if (now_mono - self._last_hold_broadcast_monotonic) >= 30.0:
                self._last_hold_broadcast_monotonic = now_mono
                for decision in batch.decisions:
                    await self.telemetry.log_ai_decision(decision)
        else:
            for decision in batch.decisions:
                await self.telemetry.log_ai_decision(decision)

        snap_map = {snap.ticker: snap for snap in snapshots}
        for intent in batch.intents:
            live_snap = (
                self.ws_client.orderbooks.get(intent.ticker)
                if self.ws_client is not None
                else None
            ) or snap_map.get(intent.ticker)
            if live_snap is not None:
                await self._evaluate_and_execute_intent(
                    intent, live_snap, s, execute_orders=execute_orders
                )

    async def _process_snapshot(
        self, snap: OrderbookSnapshot, s: AppSettings, execute_orders: bool = True
    ) -> None:
        await self._process_batch([snap], s, execute_orders=execute_orders, force=True)

    async def _evaluate_and_execute_intent(
        self,
        intent: TradeIntent,
        snap: OrderbookSnapshot,
        s: AppSettings,
        execute_orders: bool = True,
        mode: ExecutionMode | None = None,
        emergency: bool = False,
    ) -> None:
        route = mode or s.execution_mode
        max_slip_dollars = (s.max_slippage_cents / Decimal("100")).quantize(Decimal("0.0001"))
        # Pin the signal reference before any re-pricing so drift is always measured from the
        # price the edge was computed at, never from an already-clamped limit.
        if intent.reference_price is None:
            intent = intent.model_copy(update={"reference_price": intent.limit_price})
        ref_px = intent.reference_price
        assert ref_px is not None
        if snap.best_yes_bid is not None and snap.best_yes_ask is not None:
            if intent.side == OrderSide.BID:
                inside_ask = snap.best_yes_ask
                if (inside_ask - ref_px) <= max_slip_dollars:
                    clamped_px = min(
                        max(intent.limit_price, inside_ask),
                        inside_ask + Decimal("0.01"),
                        (ref_px + max_slip_dollars).quantize(Decimal("0.01"), rounding=ROUND_FLOOR),
                    )
                    if clamped_px != intent.limit_price:
                        intent = intent.model_copy(update={"limit_price": clamped_px})
            else:
                inside_bid = snap.best_yes_bid
                if (ref_px - inside_bid) <= max_slip_dollars:
                    clamped_px = max(
                        min(intent.limit_price, inside_bid),
                        inside_bid - Decimal("0.01"),
                        (ref_px - max_slip_dollars).quantize(Decimal("0.01"), rounding=ROUND_CEILING),
                    )
                    if clamped_px != intent.limit_price:
                        intent = intent.model_copy(update={"limit_price": clamped_px})

        exp = self._market_expiries.get(intent.ticker)
        if exp is None and self.rest_client is not None:
            exp = self.rest_client.market_expiries.get(intent.ticker)
        seconds_to_expiry = (
            (exp - datetime.now(timezone.utc)).total_seconds() if exp is not None else None
        )
        evaluation = self.risk_gate.evaluate(
            intent,
            snap,
            self.portfolio,
            seconds_to_expiry=seconds_to_expiry,
            emergency_exit=emergency,
        )
        # Claim the ticker synchronously with the approval, before any await, so a concurrent
        # evaluation cannot also pass the gate until this order is recorded as a position or a
        # resting order (or fails).
        claimed = evaluation.approved and execute_orders
        if claimed:
            self.portfolio.pending_order_tickers.add(intent.ticker)
        try:
            await self.telemetry.log_risk_evaluation(evaluation)
            if claimed:
                await self._execute_approved_intent(intent, snap, s, route, evaluation)
        finally:
            if claimed:
                self.portfolio.pending_order_tickers.discard(intent.ticker)

    async def _execute_approved_intent(
        self,
        intent: TradeIntent,
        snap: OrderbookSnapshot,
        s: AppSettings,
        route: ExecutionMode,
        evaluation: RiskEvaluation,
    ) -> None:
        if route == ExecutionMode.SHADOW:
            rec = self.simulator.simulate_fill(
                intent, snap, portfolio=self.portfolio, mode=ExecutionMode.SHADOW
            )
            await self.telemetry.log_execution(rec)

        elif route in (ExecutionMode.DEMO, ExecutionMode.LIVE):
            # Record shadow prediction simultaneously for fill-quality calibration
            shadow_rec = self.simulator.simulate_fill(
                intent, snap, portfolio=None, mode=route
            )
            v2_order = V2CreateOrderRequest.from_intent(intent)

            if route == ExecutionMode.DEMO and s.kalshi_environment == Environment.PRODUCTION:
                err = "Environment mismatch: execution_mode=demo cannot run when environment=production."
                logger.error(err)
                await self.telemetry.log_execution(
                    ExecutionRecord(
                        order_id=f"err-{uuid.uuid4().hex[:8]}",
                        client_order_id=v2_order.client_order_id,
                        intent_id=intent.intent_id,
                        ticker=intent.ticker,
                        mode=route,
                        side=intent.side,
                        count=intent.count,
                        requested_price=intent.limit_price,
                        fill_price=Decimal("0.0000"),
                        slippage_dollars=Decimal("0.00"),
                        fee_dollars=Decimal("0.00"),
                        status="REJECTED_BY_EXCHANGE",
                        timestamp_ms=int(time.time() * 1000),
                        error_message=err,
                    )
                )
                return

            if self.rest_client is None or self.rest_client.signer is None:
                err = f"Cannot send {route.value} order: Kalshi credentials not configured."
                logger.warning(err)
                await self.telemetry.log_execution(
                    ExecutionRecord(
                        order_id=f"err-{uuid.uuid4().hex[:8]}",
                        client_order_id=v2_order.client_order_id,
                        intent_id=intent.intent_id,
                        ticker=intent.ticker,
                        mode=route,
                        side=intent.side,
                        count=intent.count,
                        requested_price=intent.limit_price,
                        fill_price=Decimal("0.0000"),
                        slippage_dollars=Decimal("0.00"),
                        fee_dollars=Decimal("0.00"),
                        status="REJECTED_BY_EXCHANGE",
                        timestamp_ms=int(time.time() * 1000),
                        error_message=err,
                    )
                )
                return

            try:
                required_cost = (
                    (
                        intent.limit_price
                        if intent.side == OrderSide.BID
                        else (Decimal("1.00") - intent.limit_price)
                    )
                    * intent.count
                    + evaluation.estimated_entry_fee_dollars
                )
                await self._ensure_shard_funded(
                    intent.ticker, s, min_required=required_cost + Decimal("0.50")
                )
                try:
                    resp = await self.rest_client.create_order(v2_order)
                except KalshiApiError as first_exc:
                    shard = await self.rest_client.get_market_exchange_index(intent.ticker)
                    msg_low = first_exc.message.lower()
                    if "insufficient" in msg_low:
                        other_shards = [
                            (idx, bal)
                            for idx, bal in self.rest_client.shard_balances.items()
                            if idx != shard and bal > Decimal("0")
                        ]
                        if other_shards:
                            other_shards.sort(key=lambda x: x[1], reverse=True)
                            donor_idx, donor_bal = other_shards[0]
                            target_bal = self.rest_client.shard_balances.get(
                                shard, Decimal("0.00")
                            )
                            shortfall = max(
                                Decimal("0.50"),
                                (required_cost + Decimal("0.50") - target_bal).quantize(
                                    Decimal("0.01")
                                ),
                            )
                            amt = min(shortfall, donor_bal).quantize(Decimal("0.01"))
                            await self.rest_client.fund_shard(
                                shard, amt, source_shard=donor_idx
                            )
                            self._funded_shards.add(shard)
                            resp = await self.rest_client.create_order(v2_order)
                        elif shard != 0:
                            await self.rest_client.fund_shard(
                                shard, self._deployable_budget(s)
                            )
                            self._funded_shards.add(shard)
                            resp = await self.rest_client.create_order(v2_order)
                        else:
                            raise
                    else:
                        raise
                order_obj = resp.get("order", resp)
                order_id = str(order_obj.get("order_id") or uuid.uuid4().hex[:12])
                fill_count_raw = (
                    order_obj.get("fill_count")
                    if order_obj.get("fill_count") is not None
                    else order_obj.get("fill_count_fp")
                )
                rem_count_raw = (
                    order_obj.get("remaining_count")
                    if order_obj.get("remaining_count") is not None
                    else order_obj.get("remaining_count_fp")
                )
                avg_fill_raw = (
                    order_obj.get("average_fill_price")
                    if order_obj.get("average_fill_price") is not None
                    else order_obj.get("average_fill_price_dollars")
                )
                ts_ms = int(order_obj.get("ts_ms") or time.time() * 1000)

                await self.telemetry.log_exchange_event(
                    "order_response",
                    resp,
                    ticker=intent.ticker,
                    order_id=order_id,
                    timestamp_ms=ts_ms,
                )
                self._own_order_ids.add(order_id)
                raw_status = str(order_obj.get("status") or "").lower()
                if fill_count_raw is not None:
                    filled_qty = Decimal(str(fill_count_raw))
                elif raw_status in ("executed", "filled"):
                    filled_qty = intent.count
                else:
                    # Unknown fill quantity: never invent a fill; WS fills / portfolio sync reconcile
                    filled_qty = Decimal("0")
                terminal = raw_status in ("canceled", "cancelled", "executed", "filled", "expired")
                if rem_count_raw is not None:
                    rem_qty = Decimal(str(rem_count_raw))
                elif terminal:
                    rem_qty = Decimal("0")
                else:
                    rem_qty = max(Decimal("0"), intent.count - filled_qty)

                # Update in-memory shard balance immediately so back-to-back orders see true available collateral
                order_shard = await self.rest_client.get_market_exchange_index(intent.ticker)
                if order_shard in self.rest_client.shard_balances:
                    self.rest_client.shard_balances[order_shard] = max(
                        Decimal("0.00"),
                        self.rest_client.shard_balances[order_shard] - required_cost,
                    )

                if avg_fill_raw is not None:
                    actual_fill_price = Decimal(str(avg_fill_raw))
                    if actual_fill_price > Decimal("1"):
                        actual_fill_price = actual_fill_price / Decimal("100")
                    actual_fill_price = actual_fill_price.quantize(Decimal("0.0001"))
                else:
                    actual_fill_price = evaluation.estimated_fill_price

                actual_fee = order_fee_total(order_obj, filled_qty, actual_fill_price)

                status = str(
                    order_obj.get("status")
                    or ("ORDER_FILLED" if filled_qty > Decimal("0") and rem_qty == Decimal("0") else "ORDER_SUBMITTED")
                )
                self.portfolio.recent_order_timestamps_ms.append(ts_ms)
                if rem_qty > Decimal("0") or status.upper() in ("RESTING", "ORDER_SUBMITTED"):
                    self._resting_order_first_seen[order_id] = time.monotonic()
                    self.portfolio.resting_order_tickers.add(intent.ticker)
                if filled_qty > Decimal("0") and fill_count_raw is not None:
                    self._apply_order_fill(
                        order_id,
                        intent.ticker,
                        intent.side,
                        filled_qty,
                        actual_fill_price,
                        source="rest",
                        fee_total=actual_fee,
                        ts_ms=ts_ms,
                    )
                await self.rest_client.upgrade_api_tier_once()

                expected_fill_price = evaluation.estimated_fill_price
                if intent.side == OrderSide.BID:
                    adverse_unit_slip = max(
                        Decimal("0.0000"), actual_fill_price - expected_fill_price
                    )
                else:
                    adverse_unit_slip = max(
                        Decimal("0.0000"), expected_fill_price - actual_fill_price
                    )
                effective_qty = filled_qty
                actual_slippage_dollars = (adverse_unit_slip * effective_qty).quantize(
                    Decimal("0.0001")
                )

                rec = ExecutionRecord(
                    order_id=order_id,
                    client_order_id=v2_order.client_order_id,
                    intent_id=intent.intent_id,
                    ticker=intent.ticker,
                    mode=route,
                    side=intent.side,
                    count=effective_qty,
                    requested_price=intent.limit_price,
                    fill_price=actual_fill_price,
                    slippage_dollars=actual_slippage_dollars,
                    fee_dollars=actual_fee,
                    status=status,
                    timestamp_ms=ts_ms,
                    shadow_fill_price=shadow_rec.fill_price,
                    shadow_fee_dollars=shadow_rec.fee_dollars,
                )
                await self.telemetry.log_execution(rec)
                if filled_qty > Decimal("0") and isinstance(self.rest_client.signer, KalshiSigner):
                    # Also covers "executed" with no fill count: no fill was applied locally, so
                    # rebuild positions from the exchange before the in-flight marker is cleared.
                    await self.sync_balance_from_api()
            except KalshiApiError as exc:
                err = f"HTTP {exc.status_code}: {exc.message}"
                logger.error("Order submission rejected by exchange: %s", err)
                await self.telemetry.log_execution(
                    ExecutionRecord(
                        order_id=f"err-{uuid.uuid4().hex[:8]}",
                        client_order_id=v2_order.client_order_id,
                        intent_id=intent.intent_id,
                        ticker=intent.ticker,
                        mode=route,
                        side=intent.side,
                        count=intent.count,
                        requested_price=intent.limit_price,
                        fill_price=Decimal("0.0000"),
                        slippage_dollars=Decimal("0.00"),
                        fee_dollars=Decimal("0.00"),
                        status="REJECTED_BY_EXCHANGE",
                        timestamp_ms=int(time.time() * 1000),
                        shadow_fill_price=shadow_rec.fill_price,
                        shadow_fee_dollars=shadow_rec.fee_dollars,
                        error_message=err,
                    )
                )
            except Exception as exc:
                err = str(exc)
                logger.error("Order submission failed: %s", err)
                await self.telemetry.log_execution(
                    ExecutionRecord(
                        order_id=f"err-{uuid.uuid4().hex[:8]}",
                        client_order_id=v2_order.client_order_id,
                        intent_id=intent.intent_id,
                        ticker=intent.ticker,
                        mode=route,
                        side=intent.side,
                        count=intent.count,
                        requested_price=intent.limit_price,
                        fill_price=Decimal("0.0000"),
                        slippage_dollars=Decimal("0.00"),
                        fee_dollars=Decimal("0.00"),
                        status="REJECTED_BY_EXCHANGE",
                        timestamp_ms=int(time.time() * 1000),
                        error_message=err,
                    )
                )

    def get_status(self) -> dict[str, Any]:
        s = self.config_manager.settings
        if self.portfolio.kill_switch_tripped:
            state = "KILL-SWITCH TRIPPED"
        elif s.execution_mode == ExecutionMode.PAUSED:
            state = "PAUSED"
        else:
            state = "RUNNING"

        deployable = max(
            Decimal("0.00"),
            self.portfolio.current_equity_dollars
            - s.equity_floor
            - self.portfolio.open_risk_collateral_dollars,
        )
        books: dict[str, Any] = {}
        if self.ws_client is not None:
            for ticker, snap in self.ws_client.orderbooks.items():
                books[ticker] = {
                    "ticker": ticker,
                    "best_yes_bid": str(snap.best_yes_bid) if snap.best_yes_bid else None,
                    "best_yes_ask": str(snap.best_yes_ask) if snap.best_yes_ask else None,
                    "best_no_bid": str(snap.best_no_bid) if snap.best_no_bid else None,
                    "spread": str(snap.yes_spread) if snap.yes_spread else None,
                    "top5_yes_bids": [[str(l.price), str(l.count)] for l in reversed(snap.yes_bids[-5:])],
                    "top5_yes_asks": [[str(l.price), str(l.count)] for l in snap.effective_yes_asks[:5]],
                }

        return {
            "engine_state": state,
            "environment": s.kalshi_environment.value,
            "execution_mode": s.execution_mode.value,
            "kill_switch_tripped": self.portfolio.kill_switch_tripped,
            "ws_health": self.ws_client.health() if self.ws_client else {"connected": False},
            "cash_balance_dollars": f"{self.portfolio.cash_balance_dollars:.2f}",
            "current_equity_dollars": f"{self.portfolio.current_equity_dollars:.2f}",
            "equity_floor_dollars": f"{s.equity_floor:.2f}",
            "deployable_budget_dollars": f"{deployable:.2f}",
            "effective_max_position_dollars": f"{self.risk_gate.effective_max_position_dollars(self.portfolio):.2f}",
            "session_net_pnl_dollars": f"{self.portfolio.session_net_pnl_dollars:.2f}",
            "estimated_llm_spend_usd": f"{self.telemetry.session_llm_spend_usd:.4f}",
            "session_start_ts": self.session_start_ts,
            "orderbooks": books,
            "equity_history": [
                pt for pt in self.equity_history if pt["timestamp_ms"] >= self.session_start_ts
            ],
        }

    async def start_engine(self) -> None:
        s = self.config_manager.settings
        target_mode = (
            self._last_mode_before_pause
            if s.execution_mode == ExecutionMode.PAUSED
            else s.execution_mode
        )
        if target_mode == ExecutionMode.DEMO and s.kalshi_environment == Environment.PRODUCTION:
            raise ValueError(
                "Cannot start in 'demo' mode while environment is set to 'production'."
            )
        self.risk_gate.reset(self.portfolio)
        await self._persist_kill_state(False)
        self._exit_mode_when_paused = None
        new_s = self.config_manager.update_and_persist({"execution_mode": target_mode.value})
        if self._task is None or self._task.done():
            await self._ensure_single_engine_loop()
        if self.ws_client is not None and not self.ws_client.tickers:
            await self._refresh_subscriptions(new_s)
        if target_mode in (ExecutionMode.DEMO, ExecutionMode.LIVE):
            await self.sync_balance_from_api()
        else:
            await self._broadcast_equity_and_status()

    async def pause_engine(self) -> None:
        s = self.config_manager.settings
        if s.execution_mode != ExecutionMode.PAUSED:
            self._last_mode_before_pause = s.execution_mode
            self._exit_mode_when_paused = s.execution_mode
        self.config_manager.update_and_persist({"execution_mode": ExecutionMode.PAUSED.value})

    async def reset_shadow(self) -> dict[str, Any]:
        """Atomically reset shadow simulator, risk gate, and engine to a fresh paused session."""
        async with self._engine_lock:
            s = self.config_manager.settings
            self.config_manager.update_and_persist(
                {"execution_mode": ExecutionMode.PAUSED.value}
            )

            # Shadow sessions always start from the configured INITIAL_CAPITAL (M2 isolation)
            starting_equity = s.initial_capital

            reset_ts = int(time.time() * 1000)
            self.session_start_ts = reset_ts
            self.session_starting_equity = starting_equity

            self.simulator.reset(self.portfolio, starting_equity)
            self.risk_gate.reset(self.portfolio)
            await self._persist_kill_state(False)
            await self._save_daily_baseline("shadow", datetime.now(timezone.utc).date().isoformat())
            session_id = self.telemetry.new_session(session_start_ts=reset_ts)

            self.equity_history.clear()
            self.equity_history.append(
                {
                    "timestamp_ms": reset_ts,
                    "equity": str(starting_equity),
                    "floor": str(s.equity_floor),
                }
            )
            await self.telemetry.log_equity(self.portfolio, str(s.equity_floor), reset_ts)
            await self.telemetry.broadcast("STATUS", self.get_status())

            return {
                "session_id": session_id,
                "session_start_ts": reset_ts,
                "cash_balance_dollars": f"{self.portfolio.cash_balance_dollars:.2f}",
            }

    async def emergency_kill(self) -> dict[str, Any]:
        """Trip the kill-switch, pause new entries, cancel engine orders, and flatten positions."""
        s = self.config_manager.settings
        prior_mode = self._effective_mode(s)
        self.portfolio.kill_switch_tripped = True
        self.risk_gate.kill_switch_tripped = True
        await self._persist_kill_state(True)
        if prior_mode != ExecutionMode.PAUSED:
            self._last_mode_before_pause = prior_mode
            self._exit_mode_when_paused = prior_mode
        if s.execution_mode != ExecutionMode.PAUSED:
            self.config_manager.update_and_persist({"execution_mode": ExecutionMode.PAUSED.value})
            s = self.config_manager.settings

        canceled = 0
        async with self._engine_lock:
            if (
                prior_mode in (ExecutionMode.DEMO, ExecutionMode.LIVE)
                and self.rest_client is not None
                and self.rest_client.signer is not None
            ):
                try:
                    results = await self.rest_client.cancel_all_orders()
                    canceled = len(results)
                except Exception as exc:
                    logger.error("Emergency cancel_all_orders error: %s", exc)
            flatten = await self._flatten_positions(s, prior_mode)
        await self.telemetry.broadcast("STATUS", self.get_status())
        return {"canceled_orders": canceled, "flatten": flatten}

    async def _flatten_positions(
        self, s: AppSettings, mode: ExecutionMode
    ) -> list[dict[str, Any]]:
        """Send one IOC closing order per open position at the live inside quote."""
        results: list[dict[str, Any]] = []
        for ticker, pos in list(self.portfolio.positions.items()):
            qty_open = abs(pos.net_contracts)
            if qty_open == Decimal("0"):
                continue
            entry: dict[str, Any] = {"ticker": ticker, "requested_contracts": str(qty_open)}
            snap = self.ws_client.orderbooks.get(ticker) if self.ws_client is not None else None
            if mode == ExecutionMode.PAUSED:
                entry.update(status="skipped_no_active_mode", remaining_contracts=str(qty_open))
                results.append(entry)
                continue
            if snap is None or snap.best_yes_bid is None or snap.best_yes_ask is None:
                entry.update(status="no_book", remaining_contracts=str(qty_open))
                results.append(entry)
                continue
            side = OrderSide.ASK if pos.net_contracts > Decimal("0") else OrderSide.BID
            px = snap.best_yes_bid if side == OrderSide.ASK else snap.best_yes_ask
            walk = walk_book(snap, side, qty_open, limit_price=px)
            if walk.fillable_count <= Decimal("0"):
                entry.update(status="no_liquidity", remaining_contracts=str(qty_open))
                results.append(entry)
                continue
            intent = TradeIntent(
                timestamp_ms=int(time.time() * 1000),
                ticker=ticker,
                side=side,
                count=walk.fillable_count,
                limit_price=px,
                expected_edge_dollars=Decimal("0.00"),
                hold_to_expiry=False,
                confidence=Decimal("1.00"),
                inference_latency_ms=0,
                rationale=f"Emergency flatten {qty_open} @ inside ${px:.4f}",
                provider="emergency_flatten",
                reduce_only=True,
            )
            d = intent.model_dump(mode="json")
            d["action"] = "BUY" if side == OrderSide.BID else "SELL"
            d["decision_status"] = "DETERMINISTIC_EXIT"
            await self.telemetry.log_ai_decision(d)
            await self._evaluate_and_execute_intent(
                intent, snap, s, execute_orders=True, mode=mode, emergency=True
            )
            remaining_pos = self.portfolio.positions.get(ticker)
            remaining = abs(remaining_pos.net_contracts) if remaining_pos is not None else Decimal("0")
            entry.update(
                status="flattened" if remaining == Decimal("0") else "partial_or_pending",
                remaining_contracts=str(remaining),
            )
            results.append(entry)
            logger.warning("[FLATTEN] %s", entry)
        return results


async def run_orchestrator(host: str | None = None, port: int | None = None, confirm_live: bool = False) -> None:
    cfg_mgr = ConfigManager(allow_live=confirm_live)
    s = cfg_mgr.settings

    if s.execution_mode == ExecutionMode.LIVE and not confirm_live:
        logger.warning(
            "EXECUTION_MODE=live requires --confirm-live flag on CLI; defaulting to paused for safety."
        )
        cfg_mgr.update_and_persist({"execution_mode": ExecutionMode.PAUSED.value})
        s = cfg_mgr.settings
    elif (
        s.execution_mode == ExecutionMode.LIVE
        and s.strategy_provider == StrategyProvider.STUB_MOMENTUM
    ):
        logger.warning(
            "EXECUTION_MODE=live with STRATEGY_PROVIDER=stub_momentum is not allowed; "
            "defaulting to paused for safety."
        )
        cfg_mgr.update_and_persist({"execution_mode": ExecutionMode.PAUSED.value})
        s = cfg_mgr.settings

    telemetry = TelemetryManager(s.db_path)
    await telemetry.initialize()

    engine = TradingEngine(cfg_mgr, telemetry)
    app = create_app(config_manager=cfg_mgr, telemetry=telemetry, engine=engine)

    await engine.start()

    uvi_config = uvicorn.Config(
        app,
        host=host or s.web_host,
        port=port or s.web_port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(uvi_config)

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received...")
        server.should_exit = True
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    try:
        await server.serve()
    finally:
        await engine.stop()
        await telemetry.close()


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kalshi-poc", description="Kalshi AI POC Trading System")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the unified Web UI + Trading Engine")
    run_p.add_argument("--host", default=None, help="Bind host (default from config)")
    run_p.add_argument("--port", type=int, default=None, help="Bind port (default from config)")
    run_p.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required to allow execution_mode=live on startup",
    )

    exp_p = sub.add_parser("export", help="Export all telemetry SQLite WAL tables (full history) to Parquet")
    exp_p.add_argument("--format", choices=["parquet"], default="parquet")
    exp_p.add_argument("--output-dir", default="data/exports", help="Output directory")
    exp_p.add_argument("--db-path", default="data/kalshi_telemetry.db", help="SQLite DB path")

    args = parser.parse_args(argv)
    if args.command == "export":
        files = export_sqlite_to_parquet(args.db_path, args.output_dir)
        for f in files:
            print(f"Exported: {f}")
        return

    # Default to `run`
    host = getattr(args, "host", None)
    port = getattr(args, "port", None)
    confirm_live = getattr(args, "confirm_live", False)
    asyncio.run(run_orchestrator(host=host, port=port, confirm_live=confirm_live))


if __name__ == "__main__":
    cli(sys.argv[1:])
