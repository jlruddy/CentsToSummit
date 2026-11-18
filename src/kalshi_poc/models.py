"""Strict Pydantic v2 domain and Kalshi v2 wire models using Decimal exclusively."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Environment(str, Enum):
    """Kalshi API environment."""

    DEMO = "demo"
    PRODUCTION = "production"


class ExecutionMode(str, Enum):
    """Engine execution mode."""

    PAUSED = "paused"
    SHADOW = "shadow"
    DEMO = "demo"
    LIVE = "live"


class StrategyProvider(str, Enum):
    """Pluggable AI strategy provider."""

    STUB_MOMENTUM = "stub_momentum"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA_LOCAL = "ollama_local"


class OrderSide(str, Enum):
    """V2 order side quoted from the YES leg."""

    BID = "bid"  # Buy YES / Sell NO
    ASK = "ask"  # Sell YES / Buy NO at 1 - price


class TimeInForce(str, Enum):
    """Kalshi V2 time-in-force options."""

    GOOD_TILL_CANCELED = "good_till_canceled"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"
    FILL_OR_KILL = "fill_or_kill"


class SelfTradePrevention(str, Enum):
    """Kalshi V2 self-trade prevention modes."""

    TAKER_AT_CROSS = "taker_at_cross"
    MAKER = "maker"


CLIENT_ORDER_ID_PREFIX = "c2s-"


def new_client_order_id() -> str:
    """Engine-tagged client_order_id so the reaper only ever touches orders this engine placed."""
    return f"{CLIENT_ORDER_ID_PREFIX}{uuid.uuid4()}"


def _to_decimal(v: object) -> Decimal:
    """Coerce string, int, or Decimal to Decimal without float imprecision."""
    if isinstance(v, Decimal):
        return v
    if isinstance(v, float):
        return Decimal(str(v))
    return Decimal(str(v).strip())


class BalanceResponse(BaseModel):
    """Kalshi /trade-api/v2/portfolio/balance fixed-point response."""

    model_config = ConfigDict(extra="ignore")

    balance_dollars: Decimal = Field(default=Decimal("0.00"))
    portfolio_value_dollars: Decimal = Field(default=Decimal("0.00"))
    updated_ts: int | None = None
    balance_breakdown: dict[int, Decimal] = Field(default_factory=dict)

    @field_validator("balance_dollars", "portfolio_value_dollars", mode="before")
    @classmethod
    def parse_decimal(cls, v: object) -> Decimal:
        if v is None:
            return Decimal("0.00")
        return _to_decimal(v)

    @classmethod
    def from_api_payload(cls, data: dict) -> BalanceResponse:
        """Parse fixed-point dollar fields and per-shard balance_breakdown."""
        bal = data.get("balance_dollars")
        if bal is None and "balance" in data:
            bal = Decimal(str(data["balance"])) / Decimal("100")
        pv = data.get("portfolio_value_dollars")
        if pv is None and "portfolio_value" in data:
            pv = Decimal(str(data["portfolio_value"])) / Decimal("100")

        breakdown: dict[int, Decimal] = {}
        raw_bd = data.get("balance_breakdown")
        if isinstance(raw_bd, list):
            for entry in raw_bd:
                if isinstance(entry, dict) and "exchange_index" in entry:
                    idx = int(entry["exchange_index"])
                    val = entry.get("balance_dollars")
                    if val is None and "balance" in entry:
                        val = Decimal(str(entry["balance"])) / Decimal("100")
                    if val is not None:
                        breakdown[idx] = _to_decimal(val)
        elif isinstance(raw_bd, dict):
            for k, v in raw_bd.items():
                try:
                    idx = int(k)
                    if isinstance(v, dict):
                        val = v.get("balance_dollars")
                        if val is None and "balance" in v:
                            val = Decimal(str(v["balance"])) / Decimal("100")
                        if val is not None:
                            breakdown[idx] = _to_decimal(val)
                    elif v is not None:
                        breakdown[idx] = _to_decimal(v)
                except (ValueError, TypeError):
                    continue

        bal_dec = _to_decimal(bal) if bal is not None else Decimal("0.00")
        bd_sum = sum(breakdown.values(), Decimal("0.00"))
        if (
            breakdown
            and bal_dec > Decimal("0")
            and abs(bd_sum * Decimal("100") - bal_dec) <= Decimal("0.05")
        ):
            breakdown = {k: (v * Decimal("100")).quantize(Decimal("0.0001")) for k, v in breakdown.items()}

        return cls(
            balance_dollars=bal_dec,
            portfolio_value_dollars=pv if pv is not None else Decimal("0.00"),
            updated_ts=data.get("updated_ts"),
            balance_breakdown=breakdown,
        )


class PriceLevel(BaseModel):
    """Single price level in an orderbook."""

    model_config = ConfigDict(frozen=True)

    price: Decimal
    count: Decimal

    @field_validator("price", "count", mode="before")
    @classmethod
    def parse_decimal(cls, v: object) -> Decimal:
        return _to_decimal(v)


class OrderbookSnapshot(BaseModel):
    """Kalshi bids-only orderbook snapshot with derived effective asks.

    Kalshi's orderbook_fp returns `yes_dollars` and `no_dollars` arrays sorted
    ascending by price, so the best bid is the last element.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str
    timestamp_ms: int
    yes_bids: list[PriceLevel] = Field(default_factory=list)
    no_bids: list[PriceLevel] = Field(default_factory=list)

    @property
    def best_yes_bid(self) -> Decimal | None:
        return self.yes_bids[-1].price if self.yes_bids else None

    @property
    def best_no_bid(self) -> Decimal | None:
        return self.no_bids[-1].price if self.no_bids else None

    @property
    def effective_yes_asks(self) -> list[PriceLevel]:
        """Derive YES asks sorted ascending from NO bids: ask_price = 1.00 - no_bid."""
        asks = [
            PriceLevel(
                price=(Decimal("1.00") - lvl.price).quantize(Decimal("0.0001")),
                count=lvl.count,
            )
            for lvl in self.no_bids
        ]
        asks.sort(key=lambda x: x.price)
        return asks

    @property
    def effective_no_asks(self) -> list[PriceLevel]:
        """Derive NO asks sorted ascending from YES bids: ask_price = 1.00 - yes_bid."""
        asks = [
            PriceLevel(
                price=(Decimal("1.00") - lvl.price).quantize(Decimal("0.0001")),
                count=lvl.count,
            )
            for lvl in self.yes_bids
        ]
        asks.sort(key=lambda x: x.price)
        return asks

    @property
    def best_yes_ask(self) -> Decimal | None:
        if not self.no_bids:
            return None
        return Decimal("1.00") - self.no_bids[-1].price

    @property
    def best_no_ask(self) -> Decimal | None:
        if not self.yes_bids:
            return None
        return Decimal("1.00") - self.yes_bids[-1].price

    @property
    def yes_spread(self) -> Decimal | None:
        bid = self.best_yes_bid
        ask = self.best_yes_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def mid_yes_price(self) -> Decimal | None:
        bid = self.best_yes_bid
        ask = self.best_yes_ask
        if bid is not None and ask is not None:
            return (bid + ask) / Decimal("2")
        return bid if bid is not None else ask

    @classmethod
    def from_kalshi_orderbook_fp(
        cls, ticker: str, payload: dict, timestamp_ms: int
    ) -> OrderbookSnapshot:
        """Parse Kalshi `orderbook_fp` or REST orderbook payload."""
        book = payload.get("orderbook_fp") or payload.get("orderbook") or payload
        raw_yes = (
            book.get("yes_dollars_fp")
            or book.get("yes_dollars")
            or book.get("yes")
            or []
        )
        raw_no = (
            book.get("no_dollars_fp")
            or book.get("no_dollars")
            or book.get("no")
            or []
        )

        def _parse_levels(raw: list) -> list[PriceLevel]:
            levels: list[PriceLevel] = []
            for entry in raw:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    p, c = _to_decimal(entry[0]), _to_decimal(entry[1])
                    if p > Decimal("1"):
                        p = p / Decimal("100")
                    if c > Decimal("0"):
                        levels.append(PriceLevel(price=p, count=c))
            levels.sort(key=lambda x: x.price)
            return levels

        return cls(
            ticker=ticker,
            timestamp_ms=timestamp_ms,
            yes_bids=_parse_levels(raw_yes),
            no_bids=_parse_levels(raw_no),
        )


class MarketTicker(BaseModel):
    """Market summary parsed from Kalshi fixed-point fields."""

    model_config = ConfigDict(extra="ignore")

    ticker: str
    event_ticker: str = ""
    series_ticker: str = ""
    title: str = ""
    status: str = "open"
    yes_bid_dollars: Decimal | None = None
    yes_ask_dollars: Decimal | None = None
    no_bid_dollars: Decimal | None = None
    no_ask_dollars: Decimal | None = None
    last_price_dollars: Decimal | None = None
    volume_fp: Decimal = Field(default=Decimal("0"))
    close_time: datetime | None = None
    expiration_time: datetime | None = None
    exchange_index: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _fallback_legacy_cent_quotes(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for d_key, c_key in (
            ("yes_bid_dollars", "yes_bid"),
            ("yes_ask_dollars", "yes_ask"),
            ("no_bid_dollars", "no_bid"),
            ("no_ask_dollars", "no_ask"),
        ):
            if out.get(d_key) is None and out.get(c_key) is not None:
                out[d_key] = Decimal(str(out[c_key])) / Decimal("100")
        return out

    @property
    def expiry_dt(self) -> datetime | None:
        return self.close_time or self.expiration_time

    @field_validator(
        "yes_bid_dollars",
        "yes_ask_dollars",
        "no_bid_dollars",
        "no_ask_dollars",
        "last_price_dollars",
        mode="before",
    )
    @classmethod
    def parse_opt_decimal(cls, v: object) -> Decimal | None:
        if v is None or v == "":
            return None
        d = _to_decimal(v)
        return (d / Decimal("100")) if d > Decimal("1.00") else d

    @field_validator("volume_fp", mode="before")
    @classmethod
    def parse_vol(cls, v: object) -> Decimal:
        if v is None or v == "":
            return Decimal("0")
        return _to_decimal(v)


class V2CreateOrderRequest(BaseModel):
    """Kalshi V2 order creation schema (POST /trade-api/v2/portfolio/events/orders)."""

    ticker: str
    side: Literal["bid", "ask"]
    count: str
    price: str
    time_in_force: TimeInForce = TimeInForce.IMMEDIATE_OR_CANCEL
    self_trade_prevention_type: SelfTradePrevention = SelfTradePrevention.TAKER_AT_CROSS
    client_order_id: str = Field(default_factory=new_client_order_id)
    exchange_index: int | None = None
    # Exchange-enforced: may only shrink a position (Kalshi requires immediate_or_cancel)
    reduce_only: bool | None = None

    @classmethod
    def from_intent(cls, intent: TradeIntent) -> V2CreateOrderRequest:
        """Serialize a TradeIntent into Kalshi's V2 fixed-point order schema."""
        count_str = (
            str(int(intent.count))
            if intent.count == intent.count.to_integral_value()
            else f"{intent.count:.2f}"
        )
        price_str = f"{intent.limit_price:.4f}"
        return cls(
            ticker=intent.ticker,
            side=intent.side.value,
            count=count_str,
            price=price_str,
            time_in_force=intent.time_in_force,
            self_trade_prevention_type=SelfTradePrevention.TAKER_AT_CROSS,
            client_order_id=new_client_order_id(),
            reduce_only=(
                True
                if intent.reduce_only and intent.time_in_force == TimeInForce.IMMEDIATE_OR_CANCEL
                else None
            ),
        )


class TradeIntent(BaseModel):
    """Structured trade proposal emitted by the AI strategy engine.

    `expected_edge_dollars` is the total expected gross edge in dollars across `count` contracts.
    """

    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ms: int
    ticker: str
    side: OrderSide
    count: Decimal = Field(gt=Decimal("0"))
    limit_price: Decimal = Field(gt=Decimal("0"), lt=Decimal("1"))
    expected_edge_dollars: Decimal = Field(ge=Decimal("0"))
    expected_edge_per_contract: Decimal | None = Field(default=None, ge=Decimal("0"))
    hold_to_expiry: bool = False
    confidence: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    inference_latency_ms: int = Field(ge=0)
    rationale: str
    provider: str
    time_in_force: TimeInForce = TimeInForce.IMMEDIATE_OR_CANCEL
    reduce_only: bool = False

    @field_validator(
        "count",
        "limit_price",
        "expected_edge_dollars",
        "expected_edge_per_contract",
        "confidence",
        mode="before",
    )
    @classmethod
    def parse_decimal(cls, v: object) -> Decimal | None:
        if v is None:
            return None
        return _to_decimal(v)


class RiskEvaluation(BaseModel):
    """Deterministic output from RiskGate.evaluate()."""

    intent_id: str
    timestamp_ms: int
    ticker: str
    approved: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    estimated_fill_price: Decimal = Field(default=Decimal("0.0000"))
    estimated_slippage_dollars: Decimal = Field(default=Decimal("0.00"))
    estimated_entry_fee_dollars: Decimal = Field(default=Decimal("0.00"))
    estimated_round_trip_fee_dollars: Decimal = Field(default=Decimal("0.00"))
    worst_case_equity_after: Decimal = Field(default=Decimal("0.00"))


class ExecutionRecord(BaseModel):
    """Unified execution/fill record for shadow, demo, and live modes."""

    order_id: str
    client_order_id: str
    intent_id: str
    ticker: str
    mode: ExecutionMode
    side: OrderSide
    count: Decimal
    requested_price: Decimal
    fill_price: Decimal
    slippage_dollars: Decimal
    fee_dollars: Decimal
    is_maker: bool = False
    status: str
    timestamp_ms: int
    shadow_fill_price: Decimal | None = None
    shadow_fee_dollars: Decimal | None = None
    error_message: str | None = None


class Position(BaseModel):
    """Tracked position in a single market (YES-leg normalized)."""

    ticker: str
    net_contracts: Decimal = Field(default=Decimal("0"))  # +YES, -NO
    avg_entry_price: Decimal = Field(default=Decimal("0.0000"))
    cost_basis_dollars: Decimal = Field(default=Decimal("0.00"))
    realized_pnl_dollars: Decimal = Field(default=Decimal("0.00"))


class PortfolioState(BaseModel):
    """Live portfolio state used by RiskGate and telemetry."""

    cash_balance_dollars: Decimal
    starting_capital_dollars: Decimal
    daily_start_equity_dollars: Decimal
    positions: dict[str, Position] = Field(default_factory=dict)
    resting_order_collateral_dollars: Decimal = Field(default=Decimal("0.00"))
    resting_order_tickers: set[str] = Field(default_factory=set)
    recent_order_timestamps_ms: list[int] = Field(default_factory=list)
    kill_switch_tripped: bool = False

    @property
    def total_position_value_dollars(self) -> Decimal:
        return sum(
            (
                p.cost_basis_dollars
                if p.cost_basis_dollars > Decimal("0")
                else abs(p.net_contracts) * p.avg_entry_price
                for p in self.positions.values()
                if p.net_contracts != Decimal("0")
            ),
            Decimal("0.00"),
        )

    @property
    def current_equity_dollars(self) -> Decimal:
        return (
            self.cash_balance_dollars
            + self.resting_order_collateral_dollars
            + self.total_position_value_dollars
        )

    @property
    def session_net_pnl_dollars(self) -> Decimal:
        return self.current_equity_dollars - self.starting_capital_dollars

    @property
    def daily_pnl_dollars(self) -> Decimal:
        return self.current_equity_dollars - self.daily_start_equity_dollars

    @property
    def open_risk_collateral_dollars(self) -> Decimal:
        position_collateral = sum(
            (p.cost_basis_dollars for p in self.positions.values() if p.net_contracts != 0),
            Decimal("0.00"),
        )
        return position_collateral + self.resting_order_collateral_dollars
