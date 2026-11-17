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
from decimal import Decimal
from pathlib import Path
from typing import Any

import uvicorn

from kalshi_poc.auth import KalshiSigner
from kalshi_poc.client import KalshiApiError, KalshiRestClient, KalshiWebSocketClient
from kalshi_poc.config import AppSettings, ConfigManager
from kalshi_poc.models import (
    Environment,
    ExecutionMode,
    ExecutionRecord,
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    TradeIntent,
    V2CreateOrderRequest,
)
from kalshi_poc.risk_gate import RiskGate
from kalshi_poc.simulator import ShadowSimulator, calculate_taker_fee
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
        self._funded_shards: set[int] = set()
        self._engine_lock = asyncio.Lock()
        self.session_start_ts: int = int(time.time() * 1000)
        self.session_starting_equity: Decimal = s.initial_capital

        config_manager.register_reload_callback(self._on_config_reload)

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
        self.strategy = StrategyFactory.create(new_s)

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

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
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
            ):
                loop.create_task(self._refresh_subscriptions(new_s))

    async def _broadcast_equity_and_status(self) -> None:
        s = self.config_manager.settings
        ts_ms = int(time.time() * 1000)
        await self.telemetry.log_equity(self.portfolio, str(s.equity_floor), ts_ms)
        await self.telemetry.broadcast("STATUS", self.get_status())

    async def sync_balance_from_api(self) -> None:
        """Sync portfolio cash_balance from Kalshi /portfolio/balance."""
        if self.rest_client is None or self.rest_client.signer is None:
            return
        try:
            bal, _ = await self.rest_client.get_balance()
            self.portfolio.cash_balance_dollars = bal.balance_dollars
            if not self.portfolio.positions and not self.portfolio.recent_order_timestamps_ms:
                self.portfolio.starting_capital_dollars = bal.balance_dollars
                self.portfolio.daily_start_equity_dollars = bal.balance_dollars
            logger.info(
                "Synced Kalshi balance: $%s (portfolio_value=$%s, shards=%s)",
                bal.balance_dollars,
                bal.portfolio_value_dollars,
                bal.balance_breakdown or {},
            )
            await self.rest_client.log_open_positions_and_resting_orders()
            await self._broadcast_equity_and_status()
        except Exception as exc:
            logger.warning("Balance sync failed: %s", exc)

    async def _refresh_subscriptions(self, s: AppSettings) -> None:
        if self.ws_client is None:
            return
        discovered = await self._discover_tickers(s)
        merged = list(self.ws_client.tickers)
        for t in discovered:
            if t not in merged and len(merged) < 6:
                merged.append(t)
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

        # Fresh session startup: never inherit a tripped kill-switch; sync starting balance from API if configured
        self.portfolio.kill_switch_tripped = False
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
        self, ticker: str, s: AppSettings, min_required: Decimal = Decimal("2.00")
    ) -> None:
        if (
            self.rest_client is None
            or self.rest_client.signer is None
            or s.execution_mode not in (ExecutionMode.DEMO, ExecutionMode.LIVE)
        ):
            return
        shard = await self.rest_client.get_market_exchange_index(ticker)
        if shard == 0:
            return
        shard_bal = self.rest_client.shard_balances.get(shard, Decimal("0.00"))
        if shard_bal < min_required or shard not in self._funded_shards:
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
        """Track live cash/position changes via private fill WebSocket channel."""
        logger.info("Received live fill event: %s", msg)
        ts_ms = int(time.time() * 1000)
        ticker = msg.get("market_ticker") or msg.get("ticker") or "UNKNOWN"
        side_raw = msg.get("side", "yes")
        side = OrderSide.BID if side_raw in ("yes", "bid") else OrderSide.ASK
        count = Decimal(str(msg.get("count_fp") or msg.get("count") or "0"))
        price_raw = msg.get("yes_price_dollars") or msg.get("price_dollars") or msg.get("yes_price") or "0.50"
        price = Decimal(str(price_raw))
        if price > Decimal("1"):
            price = price / Decimal("100")

        if count > Decimal("0"):
            fee = calculate_taker_fee(count, price)
            self.simulator._apply_fill_to_portfolio(
                self.portfolio, ticker, side, count, price, fee, ts_ms
            )
            if self.rest_client is not None:
                await self.rest_client.upgrade_api_tier_once()

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

                # Every 30s, evict unhealthy tickers and refill missing slots up to 6
                evicted: list[str] = []
                maintenance_due = (now_mono - self._last_eviction_check_monotonic) >= 30.0
                if maintenance_due:
                    self._last_eviction_check_monotonic = now_mono
                    evicted = await self._evict_unhealthy_subscriptions(s, now_utc, now_mono)

                discovery_cooldown_ok = (now_mono - self._last_discovery_monotonic) >= 15.0
                if (
                    self.ws_client is not None
                    and discovery_cooldown_ok
                    and (
                        evicted
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

                async with self._engine_lock:
                    if (
                        s.execution_mode == ExecutionMode.SHADOW
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
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Error in trading engine loop: %s", exc)

            await asyncio.sleep(s.poll_interval_ms / 1000.0)

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
            await self._process_batch(snapshots, s, execute_orders=execute_orders, force=True)

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
        )
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
    ) -> None:
        max_slip_dollars = (s.max_slippage_cents / Decimal("100")).quantize(Decimal("0.0001"))
        if snap.best_yes_bid is not None and snap.best_yes_ask is not None:
            if intent.side == OrderSide.BID:
                inside_ask = snap.best_yes_ask
                if (inside_ask - intent.limit_price) <= max_slip_dollars:
                    clamped_px = min(
                        max(intent.limit_price, inside_ask),
                        inside_ask + Decimal("0.01"),
                    )
                    if clamped_px != intent.limit_price:
                        intent = intent.model_copy(update={"limit_price": clamped_px})
            else:
                inside_bid = snap.best_yes_bid
                if (intent.limit_price - inside_bid) <= max_slip_dollars:
                    clamped_px = max(
                        min(intent.limit_price, inside_bid),
                        inside_bid - Decimal("0.01"),
                    )
                    if clamped_px != intent.limit_price:
                        intent = intent.model_copy(update={"limit_price": clamped_px})

        evaluation = self.risk_gate.evaluate(intent, snap, self.portfolio)
        await self.telemetry.log_risk_evaluation(evaluation)

        if not evaluation.approved or not execute_orders:
            return

        if s.execution_mode == ExecutionMode.SHADOW:
            rec = self.simulator.simulate_fill(
                intent, snap, portfolio=self.portfolio, mode=ExecutionMode.SHADOW
            )
            await self.telemetry.log_execution(rec)

        elif s.execution_mode in (ExecutionMode.DEMO, ExecutionMode.LIVE):
            # Record shadow prediction simultaneously for fill-quality calibration
            shadow_rec = self.simulator.simulate_fill(
                intent, snap, portfolio=None, mode=s.execution_mode
            )
            v2_order = V2CreateOrderRequest.from_intent(intent)

            if s.execution_mode == ExecutionMode.DEMO and s.kalshi_environment == Environment.PRODUCTION:
                err = "Environment mismatch: execution_mode=demo cannot run when environment=production."
                logger.error(err)
                await self.telemetry.log_execution(
                    ExecutionRecord(
                        order_id=f"err-{uuid.uuid4().hex[:8]}",
                        client_order_id=v2_order.client_order_id,
                        intent_id=intent.intent_id,
                        ticker=intent.ticker,
                        mode=s.execution_mode,
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
                err = f"Cannot send {s.execution_mode.value} order: Kalshi credentials not configured."
                logger.warning(err)
                await self.telemetry.log_execution(
                    ExecutionRecord(
                        order_id=f"err-{uuid.uuid4().hex[:8]}",
                        client_order_id=v2_order.client_order_id,
                        intent_id=intent.intent_id,
                        ticker=intent.ticker,
                        mode=s.execution_mode,
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
                    intent.ticker, s, min_required=max(required_cost, Decimal("2.00"))
                )
                try:
                    resp = await self.rest_client.create_order(v2_order)
                except KalshiApiError as first_exc:
                    shard = await self.rest_client.get_market_exchange_index(intent.ticker)
                    if (
                        "insufficient_shard_balance" in first_exc.message.lower()
                        and shard != 0
                    ):
                        await self.rest_client.fund_shard(shard, self._deployable_budget(s))
                        self._funded_shards.add(shard)
                        resp = await self.rest_client.create_order(v2_order)
                    else:
                        raise
                order_obj = resp.get("order", resp)
                order_id = str(order_obj.get("order_id") or uuid.uuid4().hex[:12])
                fill_count_raw = order_obj.get("fill_count") or order_obj.get("fill_count_fp")
                rem_count_raw = order_obj.get("remaining_count") or order_obj.get("remaining_count_fp")
                avg_fill_raw = order_obj.get("average_fill_price") or order_obj.get("average_fill_price_dollars")
                avg_fee_raw = order_obj.get("average_fee_paid") or order_obj.get("average_fee_paid_dollars")
                ts_ms = int(order_obj.get("ts_ms") or time.time() * 1000)

                filled_qty = (
                    Decimal(str(fill_count_raw))
                    if fill_count_raw is not None
                    else intent.count
                )
                rem_qty = Decimal(str(rem_count_raw)) if rem_count_raw is not None else Decimal("0")

                if avg_fill_raw is not None:
                    actual_fill_price = Decimal(str(avg_fill_raw))
                    if actual_fill_price > Decimal("1"):
                        actual_fill_price = actual_fill_price / Decimal("100")
                    actual_fill_price = actual_fill_price.quantize(Decimal("0.0001"))
                else:
                    actual_fill_price = evaluation.estimated_fill_price

                if avg_fee_raw is not None:
                    actual_fee = Decimal(str(avg_fee_raw))
                    if actual_fee > Decimal("1") and filled_qty <= Decimal("10"):
                        actual_fee = actual_fee / Decimal("100")
                    actual_fee = actual_fee.quantize(Decimal("0.01"))
                else:
                    actual_fee = (
                        calculate_taker_fee(filled_qty, actual_fill_price)
                        if filled_qty > Decimal("0")
                        else Decimal("0.00")
                    )

                status = str(
                    order_obj.get("status")
                    or ("ORDER_FILLED" if filled_qty > Decimal("0") and rem_qty == Decimal("0") else "ORDER_SUBMITTED")
                )
                self.portfolio.recent_order_timestamps_ms.append(ts_ms)
                if filled_qty > Decimal("0") and fill_count_raw is not None:
                    self.simulator._apply_fill_to_portfolio(
                        self.portfolio,
                        intent.ticker,
                        intent.side,
                        filled_qty,
                        actual_fill_price,
                        actual_fee,
                        ts_ms,
                    )
                await self.rest_client.upgrade_api_tier_once()
                rec = ExecutionRecord(
                    order_id=order_id,
                    client_order_id=v2_order.client_order_id,
                    intent_id=intent.intent_id,
                    ticker=intent.ticker,
                    mode=s.execution_mode,
                    side=intent.side,
                    count=filled_qty if filled_qty > Decimal("0") else intent.count,
                    requested_price=intent.limit_price,
                    fill_price=actual_fill_price,
                    slippage_dollars=evaluation.estimated_slippage_dollars,
                    fee_dollars=actual_fee,
                    status=status,
                    timestamp_ms=ts_ms,
                    shadow_fill_price=shadow_rec.fill_price,
                    shadow_fee_dollars=shadow_rec.fee_dollars,
                )
                await self.telemetry.log_execution(rec)
            except KalshiApiError as exc:
                err = f"HTTP {exc.status_code}: {exc.message}"
                logger.error("Order submission rejected by exchange: %s", err)
                await self.telemetry.log_execution(
                    ExecutionRecord(
                        order_id=f"err-{uuid.uuid4().hex[:8]}",
                        client_order_id=v2_order.client_order_id,
                        intent_id=intent.intent_id,
                        ticker=intent.ticker,
                        mode=s.execution_mode,
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
                        mode=s.execution_mode,
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
        self.config_manager.update_and_persist({"execution_mode": ExecutionMode.PAUSED.value})

    async def reset_shadow(self) -> dict[str, Any]:
        """Atomically reset shadow simulator, risk gate, and engine to a fresh paused session."""
        async with self._engine_lock:
            s = self.config_manager.settings
            self.config_manager.update_and_persist(
                {"execution_mode": ExecutionMode.PAUSED.value}
            )

            starting_equity = s.initial_capital
            if self.rest_client is not None and self.rest_client.signer is not None:
                try:
                    bal, _ = await self.rest_client.get_balance()
                    starting_equity = bal.balance_dollars
                except Exception as exc:
                    logger.warning("Reset shadow balance fetch failed, using initial_capital: %s", exc)

            reset_ts = int(time.time() * 1000)
            self.session_start_ts = reset_ts
            self.session_starting_equity = starting_equity

            self.simulator.reset(self.portfolio, starting_equity)
            self.risk_gate.reset(self.portfolio)
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
        """Immediately trip kill-switch, pause engine, and cancel all resting orders."""
        self.portfolio.kill_switch_tripped = True
        self.config_manager.update_and_persist({"execution_mode": ExecutionMode.PAUSED.value})
        canceled = 0
        if self.rest_client is not None and self.rest_client.signer is not None:
            try:
                results = await self.rest_client.cancel_all_orders()
                canceled = len(results)
            except Exception as exc:
                logger.error("Emergency cancel_all_orders error: %s", exc)
        await self.telemetry.broadcast("STATUS", self.get_status())
        return {"canceled_orders": canceled}


async def run_orchestrator(host: str | None = None, port: int | None = None, confirm_live: bool = False) -> None:
    cfg_mgr = ConfigManager()
    s = cfg_mgr.settings

    if s.execution_mode == ExecutionMode.LIVE and not confirm_live:
        logger.warning(
            "EXECUTION_MODE=live requires --confirm-live flag on CLI; defaulting to paused for safety."
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

    exp_p = sub.add_parser("export", help="Export telemetry SQLite WAL tables")
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
