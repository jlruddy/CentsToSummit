"""Pluggable AI decision engine emitting structured TradeIntent objects."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx

from kalshi_poc.config import AppSettings
from kalshi_poc.models import (
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    Position,
    StrategyProvider,
    TradeIntent,
)
from kalshi_poc.simulator import calculate_taker_fee

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SYSTEM_PROMPT = (
    "You are an independent prediction-market microstructure research agent analyzing "
    "a batch of live Kalshi orderbooks. Only propose a trade when the expected settlement "
    "or short-term move exceeds $0.04 (4 cents) per contract AFTER crossing the current "
    "spread; otherwise return [] (HOLD). Return ONLY a valid JSON array of trade objects: "
    '{"ticker": "TICKER", "side": "bid"|"ask", "count": "2", '
    '"limit_price": "0.5200", "expected_edge_per_contract": "0.05", '
    '"hold_to_expiry": true, "confidence": "0.75", "rationale": "short reason"}. '
    "Respect remaining_deployable_budget and open_positions."
)


@dataclass
class BatchEvaluationResult:
    """Result of evaluating all active orderbook snapshots in one cycle."""

    intents: list[TradeIntent] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: Decimal = Decimal("0.000000")
    invoked_llm: bool = False


def estimate_llm_cost_usd(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int
) -> Decimal:
    """Estimate USD cost from token counts using conservative provider/model rates."""
    if provider == StrategyProvider.OLLAMA_LOCAL.value or (
        prompt_tokens == 0 and completion_tokens == 0
    ):
        return Decimal("0.000000")
    m = model.lower()
    if provider == StrategyProvider.GEMINI.value:
        in_rate, out_rate = (
            (Decimal("1.25"), Decimal("5.00"))
            if "pro" in m
            else (Decimal("0.15"), Decimal("0.60"))
        )
    elif provider == StrategyProvider.ANTHROPIC.value:
        in_rate, out_rate = (
            (Decimal("0.80"), Decimal("4.00"))
            if "haiku" in m
            else (Decimal("3.00"), Decimal("15.00"))
        )
    else:
        in_rate, out_rate = Decimal("1.00"), Decimal("3.00")

    million = Decimal("1000000")
    cost = (
        Decimal(prompt_tokens) * in_rate + Decimal(completion_tokens) * out_rate
    ) / million
    return cost.quantize(Decimal("0.000001"))


class BaseStrategy(ABC):
    """Abstract base class for pluggable AI decision engines."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    @abstractmethod
    async def evaluate(
        self, snapshot: OrderbookSnapshot, portfolio: PortfolioState
    ) -> TradeIntent | None:
        """Analyze an OrderbookSnapshot and optionally emit a TradeIntent."""

    async def generate_intent(
        self,
        snapshot: OrderbookSnapshot,
        portfolio: PortfolioState,
        positions_by_ticker: dict[str, Position] | None = None,
    ) -> TradeIntent | None:
        """Generate a position- and liquidity-aware TradeIntent."""
        intent, _ = await self.evaluate_detail(
            snapshot, portfolio, positions_by_ticker=positions_by_ticker
        )
        return intent

    async def evaluate_detail(
        self,
        snapshot: OrderbookSnapshot,
        portfolio: PortfolioState,
        positions_by_ticker: dict[str, Position] | None = None,
    ) -> tuple[TradeIntent | None, dict[str, Any]]:
        """Evaluate snapshot and always return a structured AI_DECISION detail dict."""
        t0 = time.perf_counter()
        intent = await self.evaluate(snapshot, portfolio)
        latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
        if intent is not None:
            detail = intent.model_dump(mode="json")
            detail["action"] = "BUY" if intent.side == OrderSide.BID else "SELL"
            detail["mid_price"] = str(snapshot.mid_yes_price) if snapshot.mid_yes_price else None
            return intent, detail

        mid = snapshot.mid_yes_price
        return None, {
            "timestamp_ms": int(time.time() * 1000),
            "intent_id": f"hold-{uuid.uuid4().hex[:8]}",
            "ticker": snapshot.ticker,
            "action": "HOLD",
            "side": "hold",
            "count": "0",
            "limit_price": str(mid) if mid is not None else "0.0000",
            "expected_edge_dollars": "0.00",
            "confidence": "0.00",
            "inference_latency_ms": latency_ms,
            "provider": self.settings.strategy_provider.value,
            "mid_price": str(mid) if mid is not None else None,
            "imbalance": "0.00",
            "rationale": f"{self.settings.strategy_provider.value}: HOLD (no actionable edge)",
        }

    async def evaluate_batch(
        self,
        snapshots: list[OrderbookSnapshot],
        portfolio: PortfolioState,
        positions_by_ticker: dict[str, Position] | None = None,
        force: bool = False,
    ) -> BatchEvaluationResult:
        """Evaluate all active snapshots in one batched cycle."""
        result = BatchEvaluationResult()
        for snap in snapshots:
            intent, detail = await self.evaluate_detail(
                snap, portfolio, positions_by_ticker=positions_by_ticker
            )
            result.decisions.append(detail)
            if intent is not None:
                result.intents.append(intent)
        return result


class StubMomentumStrategy(BaseStrategy):
    """Position-, liquidity-, and budget-aware momentum & book-imbalance strategy."""

    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings)
        self._mid_history: dict[str, deque[Decimal]] = {}

    @staticmethod
    def _hold_detail(
        *,
        ts_ms: int,
        ticker: str,
        mid: Decimal | None,
        imbalance: Decimal,
        confidence: Decimal,
        latency_ms: int,
        rationale: str,
    ) -> dict[str, Any]:
        return {
            "timestamp_ms": ts_ms,
            "intent_id": f"hold-{uuid.uuid4().hex[:8]}",
            "ticker": ticker,
            "action": "HOLD",
            "side": "hold",
            "count": "0",
            "limit_price": str(mid) if mid is not None else "0.0000",
            "expected_edge_dollars": "0.00",
            "confidence": f"{confidence:.2f}",
            "inference_latency_ms": latency_ms,
            "provider": StrategyProvider.STUB_MOMENTUM.value,
            "mid_price": str(mid) if mid is not None else None,
            "imbalance": f"{imbalance:.2f}",
            "rationale": rationale,
        }

    def _top_liquidity_within_slippage(
        self, snapshot: OrderbookSnapshot, side: OrderSide
    ) -> tuple[Decimal, Decimal]:
        """Inspect top 1-2 levels of derived book within max_slippage_cents.

        Returns (available_count, worst_acceptable_price).
        """
        levels = (
            snapshot.effective_yes_asks[:2]
            if side == OrderSide.BID
            else list(reversed(snapshot.yes_bids[-2:]))
        )
        if not levels:
            return Decimal("0"), Decimal("0")

        ref_price = snapshot.mid_yes_price if snapshot.mid_yes_price is not None else levels[0].price
        max_slip_dollars = self.settings.max_slippage_cents / Decimal("100")
        avail = Decimal("0")
        inside_px = levels[0].price
        limit_px = inside_px
        for lvl in levels:
            slip = (
                max(Decimal("0"), lvl.price - ref_price)
                if side == OrderSide.BID
                else max(Decimal("0"), ref_price - lvl.price)
            )
            within_inside_plus_1c = (
                (lvl.price <= inside_px + Decimal("0.01"))
                if side == OrderSide.BID
                else (lvl.price >= inside_px - Decimal("0.01"))
            )
            if slip <= max_slip_dollars and within_inside_plus_1c:
                avail += lvl.count
                limit_px = lvl.price
        return avail, limit_px

    def _size_for_budget(
        self,
        side: OrderSide,
        limit_price: Decimal,
        avail_liquidity: Decimal,
        remaining_budget: Decimal,
        cap_contracts: int = 10,
    ) -> int:
        """Size contract count (down to 1) from top-of-book liquidity & remaining budget."""
        if avail_liquidity < Decimal("1") or remaining_budget <= Decimal("0"):
            return 0
        unit_collateral = (
            limit_price if side == OrderSide.BID else (Decimal("1.00") - limit_price)
        )
        if unit_collateral <= Decimal("0"):
            return 0
        # Include 1-contract taker fee buffer so RiskGate floor check does not reject
        est_unit_fee = calculate_taker_fee(Decimal("1"), limit_price)
        unit_total = unit_collateral + est_unit_fee
        max_by_budget = int(remaining_budget // unit_total)
        return min(int(avail_liquidity), max_by_budget, cap_contracts)

    async def evaluate(
        self, snapshot: OrderbookSnapshot, portfolio: PortfolioState
    ) -> TradeIntent | None:
        intent, _ = await self.evaluate_detail(snapshot, portfolio)
        return intent

    async def evaluate_detail(
        self,
        snapshot: OrderbookSnapshot,
        portfolio: PortfolioState,
        positions_by_ticker: dict[str, Position] | None = None,
    ) -> tuple[TradeIntent | None, dict[str, Any]]:
        t0 = time.perf_counter()
        mid = snapshot.mid_yes_price
        best_bid = snapshot.best_yes_bid
        best_ask = snapshot.best_yes_ask
        ts_ms = int(time.time() * 1000)

        if mid is None or best_bid is None or best_ask is None:
            latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
            return None, self._hold_detail(
                ts_ms=ts_ms,
                ticker=snapshot.ticker,
                mid=mid,
                imbalance=Decimal("0"),
                confidence=Decimal("0"),
                latency_ms=latency_ms,
                rationale="StubMomentum: HOLD (one-sided or empty orderbook)",
            )

        history = self._mid_history.setdefault(snapshot.ticker, deque(maxlen=10))
        history.append(mid)

        yes_top_vol = (
            min(snapshot.yes_bids[-1].count, Decimal("500"))
            if snapshot.yes_bids
            else Decimal("0")
        )
        no_top_vol = (
            min(snapshot.no_bids[-1].count, Decimal("500"))
            if snapshot.no_bids
            else Decimal("0")
        )
        total_vol = yes_top_vol + no_top_vol
        imbalance = (
            ((yes_top_vol - no_top_vol) / total_vol) if total_vol > Decimal("0") else Decimal("0")
        )
        momentum = mid - history[0] if len(history) >= 2 else Decimal("0")
        latency_ms = max(1, int((time.perf_counter() - t0) * 1000))

        pos_map = positions_by_ticker if positions_by_ticker is not None else portfolio.positions
        pos = pos_map.get(snapshot.ticker)

        # 1. Position-Aware Exit Logic (take-profit >= +$0.04, stop-loss <= -$0.05, or flipped imbalance)
        if pos is not None and pos.net_contracts != Decimal("0"):
            if pos.net_contracts > Decimal("0"):
                unrealized = best_bid - pos.avg_entry_price
                exit_reason = (
                    "take-profit"
                    if unrealized >= Decimal("0.04")
                    else "stop-loss"
                    if unrealized <= Decimal("-0.05")
                    else "imbalance-flip"
                    if imbalance <= Decimal("-0.20")
                    else None
                )
                if exit_reason:
                    avail, limit_px = self._top_liquidity_within_slippage(snapshot, OrderSide.ASK)
                    close_qty = min(int(abs(pos.net_contracts)), int(avail))
                    if close_qty >= 1:
                        count = Decimal(str(close_qty))
                        edge = max(Decimal("0.08") * count, abs(unrealized) * count).quantize(
                            Decimal("0.01")
                        )
                        intent = TradeIntent(
                            timestamp_ms=ts_ms,
                            ticker=snapshot.ticker,
                            side=OrderSide.ASK,
                            count=count,
                            limit_price=limit_px,
                            expected_edge_dollars=edge,
                            confidence=Decimal("0.85"),
                            inference_latency_ms=latency_ms,
                            rationale=(
                                f"StubMomentum: Exit LONG ({exit_reason}, "
                                f"unrealized={unrealized:+.4f}/c, imb={imbalance:+.2f})"
                            ),
                            provider=StrategyProvider.STUB_MOMENTUM.value,
                        )
                        detail = intent.model_dump(mode="json")
                        detail["action"] = "SELL"
                        detail["mid_price"] = str(mid)
                        detail["imbalance"] = f"{imbalance:.2f}"
                        return intent, detail
            else:
                unrealized = pos.avg_entry_price - best_ask
                exit_reason = (
                    "take-profit"
                    if unrealized >= Decimal("0.04")
                    else "stop-loss"
                    if unrealized <= Decimal("-0.05")
                    else "imbalance-flip"
                    if imbalance >= Decimal("0.20")
                    else None
                )
                if exit_reason:
                    avail, limit_px = self._top_liquidity_within_slippage(snapshot, OrderSide.BID)
                    close_qty = min(int(abs(pos.net_contracts)), int(avail))
                    if close_qty >= 1:
                        count = Decimal(str(close_qty))
                        edge = max(Decimal("0.08") * count, abs(unrealized) * count).quantize(
                            Decimal("0.01")
                        )
                        intent = TradeIntent(
                            timestamp_ms=ts_ms,
                            ticker=snapshot.ticker,
                            side=OrderSide.BID,
                            count=count,
                            limit_price=limit_px,
                            expected_edge_dollars=edge,
                            confidence=Decimal("0.85"),
                            inference_latency_ms=latency_ms,
                            rationale=(
                                f"StubMomentum: Exit SHORT ({exit_reason}, "
                                f"unrealized={unrealized:+.4f}/c, imb={imbalance:+.2f})"
                            ),
                            provider=StrategyProvider.STUB_MOMENTUM.value,
                        )
                        detail = intent.model_dump(mode="json")
                        detail["action"] = "BUY"
                        detail["mid_price"] = str(mid)
                        detail["imbalance"] = f"{imbalance:.2f}"
                        return intent, detail

        # 2. Liquidity- & Budget-Aware Entry Logic
        deployable_budget = max(
            Decimal("0.00"),
            portfolio.current_equity_dollars
            - self.settings.equity_floor
            - portfolio.open_risk_collateral_dollars,
        )
        current_pos_cost = pos.cost_basis_dollars if pos is not None else Decimal("0.00")
        remaining_budget = min(
            deployable_budget,
            self.settings.max_position_dollars_per_market - current_pos_cost,
        )

        if total_vol > Decimal("0") and imbalance >= Decimal("0.20") and momentum >= Decimal("0"):
            avail, limit_px = self._top_liquidity_within_slippage(snapshot, OrderSide.BID)
            sized = self._size_for_budget(OrderSide.BID, limit_px, avail, remaining_budget)
            if sized >= 1:
                count = Decimal(str(sized))
                expected_edge = (Decimal("0.08") * count).quantize(Decimal("0.01"))
                confidence = min(Decimal("0.95"), Decimal("0.55") + abs(imbalance) * Decimal("0.4"))
                intent = TradeIntent(
                    timestamp_ms=ts_ms,
                    ticker=snapshot.ticker,
                    side=OrderSide.BID,
                    count=count,
                    limit_price=limit_px,
                    expected_edge_dollars=expected_edge,
                    expected_edge_per_contract=Decimal("0.08"),
                    confidence=confidence.quantize(Decimal("0.01")),
                    inference_latency_ms=latency_ms,
                    rationale=(
                        f"StubMomentum: BUY {sized} (imb={imbalance:+.2f}, mom={momentum:+.4f}, "
                        f"rem_budget=${remaining_budget:.2f}, top_liq={avail})"
                    ),
                    provider=StrategyProvider.STUB_MOMENTUM.value,
                )
                detail = intent.model_dump(mode="json")
                detail["action"] = "BUY"
                detail["mid_price"] = str(mid)
                detail["imbalance"] = f"{imbalance:.2f}"
                return intent, detail
            return None, self._hold_detail(
                ts_ms=ts_ms,
                ticker=snapshot.ticker,
                mid=mid,
                imbalance=imbalance,
                confidence=abs(imbalance),
                latency_ms=latency_ms,
                rationale=(
                    f"StubMomentum: HOLD BUY signal (rem_budget=${remaining_budget:.2f}, "
                    f"top_liq={avail} insufficient for 1 contract)"
                ),
            )

        if total_vol > Decimal("0") and imbalance <= Decimal("-0.20") and momentum <= Decimal("0"):
            avail, limit_px = self._top_liquidity_within_slippage(snapshot, OrderSide.ASK)
            sized = self._size_for_budget(OrderSide.ASK, limit_px, avail, remaining_budget)
            if sized >= 1:
                count = Decimal(str(sized))
                expected_edge = (Decimal("0.08") * count).quantize(Decimal("0.01"))
                confidence = min(Decimal("0.95"), Decimal("0.55") + abs(imbalance) * Decimal("0.4"))
                intent = TradeIntent(
                    timestamp_ms=ts_ms,
                    ticker=snapshot.ticker,
                    side=OrderSide.ASK,
                    count=count,
                    limit_price=limit_px,
                    expected_edge_dollars=expected_edge,
                    expected_edge_per_contract=Decimal("0.08"),
                    confidence=confidence.quantize(Decimal("0.01")),
                    inference_latency_ms=latency_ms,
                    rationale=(
                        f"StubMomentum: SELL {sized} (imb={imbalance:+.2f}, mom={momentum:+.4f}, "
                        f"rem_budget=${remaining_budget:.2f}, top_liq={avail})"
                    ),
                    provider=StrategyProvider.STUB_MOMENTUM.value,
                )
                detail = intent.model_dump(mode="json")
                detail["action"] = "SELL"
                detail["mid_price"] = str(mid)
                detail["imbalance"] = f"{imbalance:.2f}"
                return intent, detail
            return None, self._hold_detail(
                ts_ms=ts_ms,
                ticker=snapshot.ticker,
                mid=mid,
                imbalance=imbalance,
                confidence=abs(imbalance),
                latency_ms=latency_ms,
                rationale=(
                    f"StubMomentum: HOLD SELL signal (rem_budget=${remaining_budget:.2f}, "
                    f"top_liq={avail} insufficient for 1 contract)"
                ),
            )

        return None, self._hold_detail(
            ts_ms=ts_ms,
            ticker=snapshot.ticker,
            mid=mid,
            imbalance=imbalance,
            confidence=abs(imbalance),
            latency_ms=latency_ms,
            rationale=(
                f"StubMomentum: HOLD mid=${mid:.4f}, imbalance={imbalance:+.2f}, "
                f"momentum={momentum:+.4f}"
            ),
        )


class _LLMStrategyBase(BaseStrategy):
    """Shared batched prompt formatting, event-gating, single-flight lock, and JSON parsing."""

    provider_name: str = "llm"

    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings)
        self._inflight_lock = asyncio.Lock()
        self._last_llm_call_monotonic: float = 0.0
        self._last_llm_mids: dict[str, Decimal] = {}

    @staticmethod
    def _compute_imbalance(snapshot: OrderbookSnapshot) -> Decimal:
        """Compute Level 1 inside bid/ask imbalance with each level capped at 500 contracts."""
        bid_size_1 = (
            min(snapshot.yes_bids[-1].count, Decimal("500"))
            if snapshot.yes_bids
            else Decimal("0")
        )
        ask_size_1 = (
            min(snapshot.no_bids[-1].count, Decimal("500"))
            if snapshot.no_bids
            else Decimal("0")
        )
        total = bid_size_1 + ask_size_1
        return ((bid_size_1 - ask_size_1) / total) if total > Decimal("0") else Decimal("0")

    def _should_trigger_llm(
        self,
        snapshots: list[OrderbookSnapshot],
        pos_map: dict[str, Position],
        force: bool,
    ) -> tuple[bool, str]:
        if force:
            return True, "forced eval"
        now = time.monotonic()
        elapsed = (
            now - self._last_llm_call_monotonic
            if self._last_llm_call_monotonic > 0
            else float(self.settings.llm_heartbeat_sec)
        )
        if elapsed < self.settings.llm_min_cooldown_sec:
            return False, "Event gate: waiting for spread/imbalance/heartbeat trigger"

        if elapsed >= self.settings.llm_heartbeat_sec:
            return True, "heartbeat"

        if any(p.net_contracts != Decimal("0") for p in pos_map.values()):
            return True, "open_position"

        for snap in snapshots:
            spread = snap.yes_spread
            imb = abs(self._compute_imbalance(snap))
            if spread is not None and spread <= Decimal("0.04") and imb >= Decimal("0.35"):
                return True, f"spread_imbalance({snap.ticker})"
            mid = snap.mid_yes_price
            prev_mid = self._last_llm_mids.get(snap.ticker)
            if mid is not None and prev_mid is not None and abs(mid - prev_mid) >= Decimal("0.02"):
                return True, f"mid_move({snap.ticker})"

        return False, "Event gate: waiting for spread/imbalance/heartbeat trigger"

    def _build_batch_prompt(
        self,
        snapshots: list[OrderbookSnapshot],
        portfolio: PortfolioState,
        pos_map: dict[str, Position],
    ) -> str:
        deployable = max(
            Decimal("0.00"),
            portfolio.current_equity_dollars
            - self.settings.equity_floor
            - portfolio.open_risk_collateral_dollars,
        )
        markets_payload = []
        for snap in snapshots:
            mid = snap.mid_yes_price
            prev_mid = self._last_llm_mids.get(snap.ticker)
            delta = (mid - prev_mid) if (mid is not None and prev_mid is not None) else Decimal("0")
            markets_payload.append(
                {
                    "ticker": snap.ticker,
                    "best_yes_bid": str(snap.best_yes_bid),
                    "best_yes_ask": str(snap.best_yes_ask),
                    "spread": str(snap.yes_spread),
                    "mid_price": str(mid) if mid is not None else None,
                    "price_delta_since_last": f"{delta:+.4f}",
                    "imbalance": f"{self._compute_imbalance(snap):+.2f}",
                    "top3_yes_bids": [
                        [str(l.price), str(min(l.count, Decimal("500")))]
                        for l in reversed(snap.yes_bids[-3:])
                    ],
                    "top3_yes_asks": [
                        [str(l.price), str(min(l.count, Decimal("500")))]
                        for l in snap.effective_yes_asks[:3]
                    ],
                }
            )
        open_positions = {
            k: {
                "net_contracts": str(p.net_contracts),
                "avg_entry_price": str(p.avg_entry_price),
                "cost_basis_dollars": str(p.cost_basis_dollars),
            }
            for k, p in pos_map.items()
            if p.net_contracts != Decimal("0")
        }
        payload = {
            "remaining_deployable_budget_dollars": f"{deployable:.2f}",
            "max_position_dollars_per_market": str(self.settings.max_position_dollars_per_market),
            "open_positions": open_positions,
            "markets": markets_payload,
        }
        return (
            "Evaluate all markets below in one pass and return a JSON array of TradeIntent objects "
            f"(or [] if all are HOLD):\n{json.dumps(payload)}"
        )

    def _system_prompt(self) -> str:
        return self.settings.system_prompt_override.strip() or DEFAULT_BATCH_SYSTEM_PROMPT

    def _parse_batch_intents(
        self,
        raw_text: str,
        snapshots: list[OrderbookSnapshot],
        latency_ms: int,
    ) -> list[TradeIntent]:
        snap_map = {s.ticker: s for s in snapshots}
        max_slip_dollars = self.settings.max_slippage_cents / Decimal("100")
        try:
            text = raw_text.strip()
            if "```" in text:
                for part in text.split("```"):
                    cleaned = part.lstrip("json").strip()
                    if cleaned.startswith("[") or cleaned.startswith("{"):
                        text = cleaned
                        break
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed = parsed.get("intents") or ([parsed] if "ticker" in parsed else [])
            if not isinstance(parsed, list):
                return []

            intents: list[TradeIntent] = []
            ts_ms = int(time.time() * 1000)
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                if str(item.get("action", "trade")).lower() == "hold":
                    continue
                ticker = str(item.get("ticker", ""))
                snap = snap_map.get(ticker)
                if snap is None or snap.best_yes_bid is None or snap.best_yes_ask is None:
                    continue
                side = OrderSide(str(item["side"]).lower())
                raw_price = Decimal(str(item["limit_price"]))
                # Abort if live inside quote moved > max_slippage past AI reference price;
                # otherwise clamp limit_price to at most inside_ask + $0.01 on that side.
                if side == OrderSide.BID:
                    inside_ask = snap.best_yes_ask
                    if (inside_ask - raw_price) > max_slip_dollars:
                        logger.info(
                            "[SLIPPAGE_ABORT] ticker=%s side=bid ref_price=%s inside_ask=%s max_slip=%s",
                            ticker,
                            raw_price,
                            inside_ask,
                            max_slip_dollars,
                        )
                        continue
                    limit_px = min(
                        max(raw_price, inside_ask),
                        inside_ask + Decimal("0.01"),
                        raw_price + max_slip_dollars,
                    )
                else:
                    inside_bid = snap.best_yes_bid
                    if (raw_price - inside_bid) > max_slip_dollars:
                        logger.info(
                            "[SLIPPAGE_ABORT] ticker=%s side=ask ref_price=%s inside_bid=%s max_slip=%s",
                            ticker,
                            raw_price,
                            inside_bid,
                            max_slip_dollars,
                        )
                        continue
                    limit_px = max(
                        min(raw_price, inside_bid),
                        inside_bid - Decimal("0.01"),
                        raw_price - max_slip_dollars,
                    )

                if not (
                    Decimal("0.15") <= limit_px <= Decimal("0.85")
                    and Decimal("0.15") <= (Decimal("1.00") - limit_px) <= Decimal("0.85")
                ):
                    continue

                count_dec = Decimal(str(item["count"]))
                if "expected_edge_per_contract" in item and item["expected_edge_per_contract"] is not None:
                    edge_per_c = Decimal(str(item["expected_edge_per_contract"]))
                    total_edge = (edge_per_c * count_dec).quantize(Decimal("0.01"))
                else:
                    raw_edge = Decimal(str(item.get("expected_edge_dollars", "0.00")))
                    # Normalize per-contract LLM edge outputs when count > 1
                    if count_dec > Decimal("1") and raw_edge <= Decimal("0.15"):
                        edge_per_c = raw_edge
                        total_edge = (raw_edge * count_dec).quantize(Decimal("0.01"))
                    else:
                        edge_per_c = (raw_edge / count_dec).quantize(Decimal("0.0001"))
                        total_edge = raw_edge
                intents.append(
                    TradeIntent(
                        timestamp_ms=ts_ms,
                        ticker=ticker,
                        side=side,
                        count=count_dec,
                        limit_price=limit_px,
                        expected_edge_dollars=total_edge,
                        expected_edge_per_contract=edge_per_c,
                        hold_to_expiry=bool(item.get("hold_to_expiry", False)),
                        confidence=Decimal(str(item.get("confidence", "0.50"))),
                        inference_latency_ms=latency_ms,
                        rationale=str(item.get("rationale", "LLM batch signal"))[:400],
                        provider=self.provider_name,
                    )
                )
            return intents
        except Exception as exc:
            logger.debug("Failed to parse batched LLM output: %s", exc)
            return []

    async def _call_llm_batch(
        self, prompt: str
    ) -> tuple[str, int, int]:
        """Execute single provider API call. Returns (raw_text, prompt_tokens, completion_tokens)."""
        raise NotImplementedError

    async def evaluate(
        self, snapshot: OrderbookSnapshot, portfolio: PortfolioState
    ) -> TradeIntent | None:
        batch = await self.evaluate_batch([snapshot], portfolio, force=True)
        return batch.intents[0] if batch.intents else None

    async def evaluate_batch(
        self,
        snapshots: list[OrderbookSnapshot],
        portfolio: PortfolioState,
        positions_by_ticker: dict[str, Position] | None = None,
        force: bool = False,
    ) -> BatchEvaluationResult:
        if not snapshots:
            return BatchEvaluationResult()

        pos_map = positions_by_ticker if positions_by_ticker is not None else portfolio.positions
        ts_ms = int(time.time() * 1000)

        def _hold_batch(reason: str) -> BatchEvaluationResult:
            snap0 = snapshots[0]
            return BatchEvaluationResult(
                decisions=[
                    {
                        "timestamp_ms": ts_ms,
                        "intent_id": f"gate-{uuid.uuid4().hex[:8]}",
                        "ticker": snap0.ticker if len(snapshots) == 1 else f"BATCH({len(snapshots)})",
                        "action": "HOLD",
                        "side": "hold",
                        "count": "0",
                        "limit_price": str(snap0.mid_yes_price or "0.0000"),
                        "expected_edge_dollars": "0.00",
                        "confidence": "0.00",
                        "inference_latency_ms": 0,
                        "provider": self.provider_name,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "estimated_cost_usd": "0.000000",
                        "rationale": reason,
                    }
                ]
            )

        if self._inflight_lock.locked():
            return _hold_batch("Single-flight: LLM call already in progress")

        should_call, trigger_reason = self._should_trigger_llm(snapshots, pos_map, force)
        if not should_call:
            return _hold_batch(trigger_reason)

        async with self._inflight_lock:
            prompt = self._build_batch_prompt(snapshots, portfolio, pos_map)
            self._last_llm_call_monotonic = time.monotonic()
            t0 = time.perf_counter()
            try:
                raw_text, p_tok, c_tok = await asyncio.wait_for(
                    self._call_llm_batch(prompt), timeout=8.0
                )
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code in (401, 402, 403):
                    self._last_llm_call_monotonic = time.monotonic() + 60.0
                logger.warning(
                    "%s HTTP %s error during batch inference", self.provider_name, status_code
                )
                return _hold_batch(f"{self.provider_name} error: HTTP {status_code}")
            except Exception as exc:
                logger.warning("%s batch inference error/timeout: %r", self.provider_name, exc)
                return _hold_batch(f"{self.provider_name} error: {exc!r}")

            latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
            for snap in snapshots:
                if snap.mid_yes_price is not None:
                    self._last_llm_mids[snap.ticker] = snap.mid_yes_price

            cost_usd = estimate_llm_cost_usd(
                self.provider_name, self.settings.llm_model, p_tok, c_tok
            )
            intents = self._parse_batch_intents(raw_text, snapshots, latency_ms)

            decisions: list[dict[str, Any]] = []
            if intents:
                # Attach token/cost metadata to the first intent's telemetry record
                for idx, intent in enumerate(intents):
                    d = intent.model_dump(mode="json")
                    d["action"] = "BUY" if intent.side == OrderSide.BID else "SELL"
                    d["prompt_tokens"] = p_tok if idx == 0 else 0
                    d["completion_tokens"] = c_tok if idx == 0 else 0
                    d["estimated_cost_usd"] = f"{cost_usd:.6f}" if idx == 0 else "0.000000"
                    decisions.append(d)
            else:
                decisions.append(
                    {
                        "timestamp_ms": int(time.time() * 1000),
                        "intent_id": f"llm-{uuid.uuid4().hex[:8]}",
                        "ticker": f"BATCH({len(snapshots)})",
                        "action": "HOLD",
                        "side": "hold",
                        "count": "0",
                        "limit_price": str(snapshots[0].mid_yes_price or "0.0000"),
                        "expected_edge_dollars": "0.00",
                        "confidence": "0.00",
                        "inference_latency_ms": latency_ms,
                        "provider": self.provider_name,
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "estimated_cost_usd": f"{cost_usd:.6f}",
                        "rationale": f"{self.provider_name} ({trigger_reason}): HOLD across {len(snapshots)} markets",
                    }
                )

            return BatchEvaluationResult(
                intents=intents,
                decisions=decisions,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                estimated_cost_usd=cost_usd,
                invoked_llm=True,
            )


class AnthropicStrategy(_LLMStrategyBase):
    provider_name = StrategyProvider.ANTHROPIC.value

    async def _call_llm_batch(self, prompt: str) -> tuple[str, int, int]:
        if not self.settings.llm_api_key:
            raise ValueError("LLM API key not configured")
        endpoint = self.settings.llm_endpoint or "https://api.anthropic.com/v1/messages"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                endpoint,
                headers={
                    "x-api-key": self.settings.llm_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.settings.llm_model or "claude-3-5-sonnet-latest",
                    "max_tokens": 500,
                    "temperature": float(self.settings.llm_temperature),
                    "system": self._system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            body = resp.json()
            text = body["content"][0]["text"]
            usage = body.get("usage") or {}
            return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


class GeminiStrategy(_LLMStrategyBase):
    provider_name = StrategyProvider.GEMINI.value

    async def _call_llm_batch(self, prompt: str) -> tuple[str, int, int]:
        if not self.settings.llm_api_key:
            raise ValueError("LLM API key not configured")
        model = self.settings.llm_model or "gemini-2.5-flash"
        endpoint = (
            self.settings.llm_endpoint
            or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self.settings.llm_api_key,
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=8.0) as client:
            base_payload = {
                "system_instruction": {"parts": [{"text": self._system_prompt()}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": float(self.settings.llm_temperature),
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingLevel": "low"},
                },
            }
            resp = await client.post(
                endpoint,
                headers=headers,
                json=base_payload,
            )
            if resp.status_code == 400:
                fallback_payload = {
                    **base_payload,
                    "generationConfig": {
                        "temperature": float(self.settings.llm_temperature),
                        "responseMimeType": "application/json",
                    },
                }
                resp = await client.post(
                    endpoint,
                    headers=headers,
                    json=fallback_payload,
                )
            resp.raise_for_status()
            body = resp.json()
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            meta = body.get("usageMetadata") or {}
            return (
                text,
                int(meta.get("promptTokenCount", 0)),
                int(meta.get("candidatesTokenCount", 0)),
            )


class OllamaLocalStrategy(_LLMStrategyBase):
    provider_name = StrategyProvider.OLLAMA_LOCAL.value

    async def _call_llm_batch(self, prompt: str) -> tuple[str, int, int]:
        endpoint = self.settings.llm_endpoint or "http://127.0.0.1:11434/api/chat"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                endpoint,
                json={
                    "model": self.settings.llm_model or "llama3.2",
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": float(self.settings.llm_temperature)},
                    "messages": [
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            body = resp.json()
            text = body["message"]["content"]
            return text, int(body.get("prompt_eval_count", 0)), int(body.get("eval_count", 0))


class StrategyFactory:
    """Create the configured AI strategy instance."""

    @staticmethod
    def create(settings: AppSettings) -> BaseStrategy:
        mapping: dict[StrategyProvider, type[BaseStrategy]] = {
            StrategyProvider.STUB_MOMENTUM: StubMomentumStrategy,
            StrategyProvider.ANTHROPIC: AnthropicStrategy,
            StrategyProvider.GEMINI: GeminiStrategy,
            StrategyProvider.OLLAMA_LOCAL: OllamaLocalStrategy,
        }
        cls = mapping.get(settings.strategy_provider, StubMomentumStrategy)
        return cls(settings)
