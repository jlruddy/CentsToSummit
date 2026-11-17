"""Async httpx REST client and WebSocket v2 market-data/orderbook/fill streamer."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

import httpx
import websockets

from kalshi_poc.auth import KalshiSigner
from kalshi_poc.models import (
    BalanceResponse,
    MarketTicker,
    OrderbookSnapshot,
    PriceLevel,
    V2CreateOrderRequest,
)

logger = logging.getLogger(__name__)


class KalshiApiError(Exception):
    """Raised on Kalshi REST API errors with diagnostic status and detail."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class KalshiRestClient:
    """Async REST client for Kalshi v2 API with strict rate-limit & 429 backoff."""

    def __init__(
        self,
        base_url: str,
        signer: KalshiSigner | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._balance_lock = asyncio.Lock()
        self._last_balance_call_monotonic: float = 0.0
        self.market_shards: dict[str, int] = {}
        self.shard_balances: dict[int, Decimal] = {}
        self._last_markets_cursor: str | None = None
        self._tier_upgraded: bool = False

    async def close(self) -> None:
        await self._client.aclose()

    def _full_sign_path(self, endpoint: str) -> str:
        # base_url ends with /trade-api/v2
        ep = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return f"/trade-api/v2{ep}"

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        private: bool = True,
        max_retries: int = 3,
    ) -> tuple[dict[str, Any], float]:
        ep = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url = f"{self.base_url}{ep}"
        sign_path = self._full_sign_path(ep)

        for attempt in range(max_retries + 1):
            headers: dict[str, str] = {"Accept": "application/json"}
            if json_body is not None:
                headers["Content-Type"] = "application/json"
            if private:
                if self.signer is None:
                    raise KalshiApiError(401, "Kalshi credentials are not configured.")
                headers.update(self.signer.headers(method, sign_path))

            t0 = time.perf_counter()
            resp = await self._client.request(
                method, url, params=params, json=json_body, headers=headers
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code == 429 and attempt < max_retries:
                retry_after = float(resp.headers.get("Retry-After", 2**attempt))
                logger.warning("Kalshi 429 rate limit on %s, backing off %.2fs", ep, retry_after)
                await asyncio.sleep(retry_after)
                continue

            if resp.status_code >= 400:
                try:
                    err_data = resp.json()
                    detail = err_data.get("error", {}).get("message") or err_data.get(
                        "message", resp.text
                    )
                except Exception:
                    detail = resp.text
                raise KalshiApiError(resp.status_code, str(detail))

            return (resp.json() if resp.content else {}), latency_ms

        raise KalshiApiError(429, "Max retries exceeded on HTTP 429")

    async def get_balance(self) -> tuple[BalanceResponse, float]:
        """Call GET /trade-api/v2/portfolio/balance respecting 1 req/sec no-burst bucket."""
        async with self._balance_lock:
            now = time.monotonic()
            elapsed = now - self._last_balance_call_monotonic
            if elapsed < 1.05:
                await asyncio.sleep(1.05 - elapsed)
            self._last_balance_call_monotonic = time.monotonic()
            data, latency_ms = await self._request("GET", "/portfolio/balance", private=True)
            bal = BalanceResponse.from_api_payload(data)
            if bal.balance_breakdown:
                self.shard_balances = dict(bal.balance_breakdown)
            return bal, latency_ms

    async def get_markets(
        self,
        series_ticker: str | None = None,
        status: str = "open",
        limit: int = 200,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        mve_filter: str | None = None,
        cursor: str | None = None,
    ) -> list[MarketTicker]:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if max_close_ts is not None:
            params["max_close_ts"] = max_close_ts
        if mve_filter:
            params["mve_filter"] = mve_filter
        if cursor:
            params["cursor"] = cursor
        data, _ = await self._request("GET", "/markets", params=params, private=False)
        self._last_markets_cursor = data.get("cursor") or None
        markets = [MarketTicker.model_validate(m) for m in data.get("markets", [])]
        for m in markets:
            if m.exchange_index is not None:
                self.market_shards[m.ticker] = m.exchange_index
        return markets

    async def get_market_exchange_index(self, ticker: str) -> int:
        """Return cached exchange_index for ticker, or fetch GET /markets/{ticker} if unknown."""
        if ticker in self.market_shards:
            return self.market_shards[ticker]
        try:
            data, _ = await self._request("GET", f"/markets/{ticker}", private=False)
            mkt = data.get("market", data)
            idx = mkt.get("exchange_index")
            if idx is not None:
                shard = int(idx)
                self.market_shards[ticker] = shard
                return shard
        except Exception as exc:
            logger.debug("Failed to fetch exchange_index for %s: %s", ticker, exc)
        return 0

    async def discover_active_markets(
        self,
        prefixes: list[str],
        max_tickers: int = 6,
        environment: str = "demo",
        now_utc: datetime | None = None,
        max_per_series: int = 1,
    ) -> list[MarketTicker]:
        """Direct per-series intraday discovery (limit=200, >= 125s life, 0.18<=mid<=0.82, hourly fallbacks)."""
        now = now_utc or datetime.now(timezone.utc)
        now_et = now.astimezone(ZoneInfo("America/New_York"))
        in_us_market_hours = (
            now_et.weekday() < 5
            and (now_et.hour, now_et.minute) >= (9, 30)
            and (now_et.hour, now_et.minute) < (16, 0)
        )

        target_series = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M"]
        if in_us_market_hours:
            target_series.extend(["KXINXU", "KXNASDAQ100U"])

        crypto_hourly_fallback: dict[str, str] = {
            "KXBTC15M": "KXBTCD",
            "KXETH15M": "KXETHD",
            "KXSOL15M": "KXSOLD",
            "KXXRP15M": "KXXRPD",
        }

        pref_tuple = tuple(p.upper() for p in prefixes)
        if pref_tuple:
            matched_series = [s for s in target_series if s.startswith(pref_tuple)]
            for p in prefixes:
                p_up = p.upper()
                if p_up not in matched_series and not any(
                    s.startswith(p_up) for s in matched_series
                ):
                    matched_series.append(p_up)
            query_series = matched_series[:6]
        else:
            query_series = target_series

        def _series_key(m: MarketTicker) -> str:
            raw = (m.series_ticker or m.ticker.split("-", 1)[0]).upper()
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

        def _norm_dollar(v: Decimal | None) -> Decimal | None:
            if v is None:
                return None
            return (v / Decimal("100")) if v > Decimal("1.00") else v

        def _derived_quotes(m: MarketTicker) -> tuple[Decimal | None, Decimal | None]:
            bid = _norm_dollar(m.yes_bid_dollars)
            no_bid = _norm_dollar(m.no_bid_dollars)
            ask = _norm_dollar(m.yes_ask_dollars)
            if ask is None and no_bid is not None and Decimal("0") < no_bid < Decimal("1.00"):
                ask = Decimal("1.00") - no_bid
            return bid, ask

        def _expiry_dt_utc(m: MarketTicker) -> datetime | None:
            dt = m.expiry_dt
            if dt is None:
                return None
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

        def _remaining_secs(m: MarketTicker) -> float:
            dt = _expiry_dt_utc(m)
            if dt is None:
                return 86400.0
            return (dt - now).total_seconds()

        def _qualifies(m: MarketTicker) -> bool:
            bid, ask = _derived_quotes(m)
            if bid is None or ask is None:
                return False
            if bid <= Decimal("0.00") or ask >= Decimal("1.00"):
                return False
            if bid >= Decimal("0.90") or ask <= Decimal("0.10"):
                return False
            spread = ask - bid
            mid = (bid + ask) / Decimal("2.0")
            return (
                Decimal("0.00") <= spread <= Decimal("0.06")
                and Decimal("0.18") <= mid <= Decimal("0.82")
            )

        def _score_contract(m: MarketTicker) -> float:
            bid_d, ask_d = _derived_quotes(m)
            bid = float(bid_d) if bid_d is not None else 0.0
            ask = float(ask_d) if ask_d is not None else 1.0
            mid = (bid + ask) / 2.0
            spread = max(0.0, ask - bid)
            close_in = max(0.0, _remaining_secs(m))
            return abs(mid - 0.50) + (spread * 2.0) + (close_in / 86400.0) * 0.1

        def _spread_val(m: MarketTicker) -> Decimal:
            bid, ask = _derived_quotes(m)
            if bid is None or ask is None:
                return Decimal("1.00")
            return max(Decimal("0"), ask - bid)

        async def _fetch_series(series: str) -> list[MarketTicker]:
            for attempt in range(2):
                try:
                    return await self.get_markets(
                        series_ticker=series,
                        status="open",
                        limit=200,
                    )
                except KalshiApiError as exc:
                    if exc.status_code == 429 and attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    logger.warning("[DISCOVERY] series=%s error=HTTP %s", series, exc.status_code)
                    return []
                except Exception as exc:
                    logger.warning("[DISCOVERY] series=%s error=%r", series, exc)
                    return []
            return []

        def _pick_for_series(series: str, mkts: list[MarketTicker]) -> list[MarketTicker]:
            s_up = series.upper()
            series_mkts = [
                m
                for m in mkts
                if (
                    m.series_ticker
                    and (
                        m.series_ticker.upper() == s_up
                        or s_up.startswith(m.series_ticker.upper())
                    )
                )
                or (not m.series_ticker and m.ticker.upper().startswith(s_up))
            ]
            eligible = [m for m in series_mkts if _remaining_secs(m) >= 125.0]
            if not eligible:
                logger.info(
                    "[DISCOVERY] series=%s open_count=%d selected=NONE reason=no_contract_ge_125s",
                    series,
                    len(series_mkts),
                )
                return []

            qualified = [m for m in eligible if _qualifies(m)]
            if not qualified:
                sample = eligible[0]
                s_bid, s_ask = _derived_quotes(sample)
                logger.info(
                    "[DISCOVERY] series=%s open_count=%d selected=NONE reason=no_contract_in_mid_0.18_0.82_spread_0.06 (sample=%s close_in=%.0fs rest_bid=%s rest_ask=%s)",
                    series,
                    len(series_mkts),
                    sample.ticker,
                    _remaining_secs(sample),
                    s_bid,
                    s_ask,
                )
                return []

            qualified.sort(key=lambda m: (round(_score_contract(m), 9), -float(m.volume_fp)))
            chosen = qualified[0]
            bid, ask = _derived_quotes(chosen)
            logger.info(
                "[DISCOVERY] series=%s open_count=%d selected=%s close_in=%.0fs rest_bid=%s rest_ask=%s score=%.4f",
                series,
                len(series_mkts),
                chosen.ticker,
                _remaining_secs(chosen),
                bid,
                ask,
                _score_contract(chosen),
            )
            return qualified[:max_per_series]

        selected: list[MarketTicker] = []
        seen_tickers: set[str] = set()
        series_counts: dict[str, int] = {}

        for idx, series in enumerate(query_series):
            if idx > 0:
                await asyncio.sleep(0.20)
            mkts = await _fetch_series(series)
            picks = _pick_for_series(series, mkts)

            # Fallback to hourly multi-strike crypto series during 2-minute 15M rollover windows
            if not picks and series in crypto_hourly_fallback:
                fb_series = crypto_hourly_fallback[series]
                await asyncio.sleep(0.20)
                fb_mkts = await _fetch_series(fb_series)
                picks = _pick_for_series(fb_series, fb_mkts)

            # Fallback to KXNASDAQDUD if KXNASDAQ100U has no qualifying active contracts
            if not picks and series == "KXNASDAQ100U":
                await asyncio.sleep(0.20)
                dud_mkts = await _fetch_series("KXNASDAQDUD")
                picks = _pick_for_series("KXNASDAQDUD", dud_mkts)

            for chosen in picks:
                if chosen.ticker not in seen_tickers:
                    sk = _series_key(chosen)
                    if series_counts.get(sk, 0) < max_per_series:
                        series_counts[sk] = series_counts.get(sk, 0) + 1
                        seen_tickers.add(chosen.ticker)
                        selected.append(chosen)
                        if len(selected) >= max_tickers:
                            break
            if len(selected) >= max_tickers:
                break

        # Demo sandbox fallback if per-series queries returned nothing
        if not selected and environment == "demo":
            try:
                demo_mkts = await self.get_markets(
                    status="open", limit=50, mve_filter="exclude"
                )
                eligible_demo = [
                    m for m in demo_mkts if _remaining_secs(m) >= 125.0 and _qualifies(m)
                ]
                eligible_demo.sort(key=lambda m: (_spread_val(m), -m.volume_fp))
                for m in eligible_demo:
                    sk = _series_key(m)
                    if m.ticker not in seen_tickers and series_counts.get(sk, 0) < max_per_series:
                        series_counts[sk] = series_counts.get(sk, 0) + 1
                        seen_tickers.add(m.ticker)
                        selected.append(m)
                        if len(selected) >= max_tickers:
                            break
            except Exception as exc:
                logger.debug("Demo fallback market discovery failed: %s", exc)

        return selected[:max_tickers]

    async def get_orderbook(self, ticker: str) -> OrderbookSnapshot:
        data, _ = await self._request("GET", f"/markets/{ticker}/orderbook", private=False)
        ts_ms = int(time.time() * 1000)
        return OrderbookSnapshot.from_kalshi_orderbook_fp(ticker, data, ts_ms)

    async def create_order(self, order: V2CreateOrderRequest) -> dict[str, Any]:
        if order.exchange_index is None:
            shard = self.market_shards.get(order.ticker, 0)
            if shard != 0:
                order = order.model_copy(update={"exchange_index": shard})
        data, _ = await self._request(
            "POST",
            "/portfolio/events/orders",
            json_body=order.model_dump(mode="json", exclude_none=True),
            private=True,
        )
        return data

    async def fund_shard(self, target_shard: int, amount_dollars: Decimal) -> dict[str, Any]:
        """Transfer collateral from Shard 0 to `target_shard` (amount in centicents)."""
        if target_shard == 0 or amount_dollars <= Decimal("0"):
            return {}
        centicents = int(amount_dollars * Decimal("10000"))
        if centicents <= 0:
            return {}
        payload = {
            "source": "event_contract",
            "destination": "event_contract",
            "source_exchange_shard": 0,
            "destination_exchange_shard": target_shard,
            "source_subaccount": 0,
            "destination_subaccount": 0,
            "amount": centicents,
        }
        try:
            data, _ = await self._request(
                "POST",
                "/portfolio/intra_exchange_instance_transfer",
                json_body=payload,
                private=True,
            )
            self.shard_balances[target_shard] = (
                self.shard_balances.get(target_shard, Decimal("0.00")) + amount_dollars
            )
            logger.info(
                "Transferred $%.2f from shard 0 -> shard %d", amount_dollars, target_shard
            )
            return data
        except KalshiApiError as exc:
            logger.error(
                "Shard transfer failed: HTTP %s %s", exc.status_code, exc.message
            )
            raise

    async def upgrade_api_tier_once(self) -> None:
        """One-time call to upgrade Kalshi API rate-limit tier after first live fill."""
        if self._tier_upgraded or self.signer is None:
            return
        self._tier_upgraded = True
        try:
            await self._request(
                "POST",
                "/account/api_usage_level/upgrade",
                private=True,
            )
            logger.info("Requested Kalshi API usage tier upgrade.")
        except KalshiApiError as exc:
            if exc.status_code in (400, 403):
                logger.debug("API tier upgrade already applied or unavailable (%s)", exc)
            else:
                logger.warning("API tier upgrade request failed: %s", exc)
        except Exception as exc:
            logger.debug("API tier upgrade error: %s", exc)

    async def cancel_order(
        self, order_id: str, market_ticker: str | None = None
    ) -> dict[str, Any]:
        params = {"market_ticker": market_ticker} if market_ticker else None
        data, _ = await self._request(
            "DELETE",
            f"/portfolio/events/orders/{order_id}",
            params=params,
            private=True,
        )
        return data

    async def get_orders(self, status: str = "resting") -> list[dict[str, Any]]:
        data, _ = await self._request(
            "GET", "/portfolio/events/orders", params={"status": status}, private=True
        )
        return data.get("orders", [])

    async def get_open_positions(self) -> list[dict[str, Any]]:
        """Fetch open positions via GET /trade-api/v2/portfolio/positions with non-zero position count."""
        data, _ = await self._request("GET", "/portfolio/positions", private=True)
        raw_positions = (
            data.get("market_positions")
            or data.get("positions")
            or data.get("event_positions")
            or []
        )
        open_positions: list[dict[str, Any]] = []
        for p in raw_positions:
            if not isinstance(p, dict):
                continue
            raw_cnt = (
                p.get("position_fp")
                if p.get("position_fp") is not None
                else p.get("position", 0)
            )
            try:
                cnt = Decimal(str(raw_cnt))
            except Exception:
                cnt = Decimal("0")
            if cnt != Decimal("0"):
                open_positions.append(p)
        return open_positions

    async def get_resting_orders(self) -> list[dict[str, Any]]:
        """Fetch resting orders via GET /trade-api/v2/portfolio/orders?status=resting."""
        try:
            data, _ = await self._request(
                "GET", "/portfolio/orders", params={"status": "resting"}, private=True
            )
            orders = data.get("orders", [])
            if isinstance(orders, list):
                return orders
        except KalshiApiError:
            pass
        return await self.get_orders(status="resting")

    async def log_open_positions_and_resting_orders(self) -> None:
        """Fetch and log non-zero positions and resting orders for portfolio visibility."""
        try:
            open_pos = await self.get_open_positions()
            if open_pos:
                for p in open_pos:
                    ticker = p.get("ticker") or p.get("market_ticker") or p.get("event_ticker")
                    cnt = p.get("position_fp") if p.get("position_fp") is not None else p.get("position")
                    exp = (
                        p.get("market_exposure_dollars")
                        or p.get("market_exposure")
                        or p.get("total_traded_dollars")
                        or p.get("total_traded")
                    )
                    rpnl = p.get("realized_pnl_dollars") or p.get("realized_pnl")
                    fees = p.get("fees_paid_dollars") or p.get("fees_paid")
                    logger.info(
                        "[PORTFOLIO] Open position: ticker=%s position=%s exposure=%s realized_pnl=%s fees=%s",
                        ticker,
                        cnt,
                        exp,
                        rpnl,
                        fees,
                    )
            else:
                logger.info("[PORTFOLIO] Open positions: NONE (0 non-zero positions)")
        except Exception as exc:
            logger.warning("[PORTFOLIO] Failed to fetch open positions: %s", exc)

        try:
            resting = await self.get_resting_orders()
            if resting:
                for o in resting:
                    ticker = o.get("ticker") or o.get("market_ticker")
                    oid = o.get("order_id")
                    side = o.get("side") or o.get("action")
                    rem = o.get("remaining_count_fp") or o.get("remaining_count") or o.get("count")
                    price = (
                        o.get("yes_price_dollars")
                        or o.get("price_dollars")
                        or o.get("price")
                        or o.get("yes_price")
                    )
                    logger.info(
                        "[PORTFOLIO] Resting order: order_id=%s ticker=%s side=%s remaining=%s price=%s",
                        oid,
                        ticker,
                        side,
                        rem,
                        price,
                    )
            else:
                logger.info("[PORTFOLIO] Resting orders: NONE (0 resting orders)")
        except Exception as exc:
            logger.warning("[PORTFOLIO] Failed to fetch resting orders: %s", exc)

    async def cancel_all_orders(self) -> list[dict[str, Any]]:
        """Batch-cancel all resting orders via V2 DELETE /portfolio/events/orders."""
        try:
            data, _ = await self._request(
                "DELETE", "/portfolio/events/orders", private=True
            )
            canceled = data.get("orders") or data.get("canceled_orders")
            if isinstance(canceled, list):
                return canceled
            return [data] if data else []
        except Exception as exc:
            logger.warning("Batch V2 cancel_all_orders failed (%s), falling back to per-order", exc)
            resting = await self.get_orders(status="resting")
            results: list[dict[str, Any]] = []
            for order in resting:
                oid = order.get("order_id")
                ticker = order.get("ticker") or order.get("market_ticker")
                if oid:
                    try:
                        results.append(await self.cancel_order(oid, market_ticker=ticker))
                    except Exception as inner_exc:
                        logger.error("Failed to cancel order %s: %s", oid, inner_exc)
            return results


class KalshiWebSocketClient:
    """Async WebSocket v2 streamer for orderbook_delta, ticker, and private fill events."""

    def __init__(
        self,
        ws_url: str,
        signer: KalshiSigner | None = None,
        on_orderbook: Callable[[OrderbookSnapshot], Awaitable[None]] | None = None,
        on_fill: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.ws_url = ws_url
        self.signer = signer
        self.on_orderbook = on_orderbook
        self.on_fill = on_fill

        self.orderbooks: dict[str, OrderbookSnapshot] = {}
        self._yes_books: dict[str, dict[Decimal, Decimal]] = {}
        self._no_books: dict[str, dict[Decimal, Decimal]] = {}
        self._sid_to_ticker: dict[int, str] = {}
        self.tickers: list[str] = []
        self._ws: Any = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._cmd_id = 1

        # Health metrics
        self.connected = False
        self.last_message_ms: int | None = None
        self.reconnect_count = 0
        self.last_error: str | None = None

    def health(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "last_message_ms": self.last_message_ms,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error,
            "subscribed_tickers": list(self.tickers),
        }

    def start(self, tickers: list[str]) -> None:
        self.tickers = list(dict.fromkeys(tickers))
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run_loop(), name="kalshi-ws")

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def update_subscriptions(self, tickers: list[str]) -> None:
        """Update ticker subscriptions without dropping the WebSocket connection."""
        new_list = list(dict.fromkeys(tickers))
        if set(new_list) == set(self.tickers):
            return
        self.tickers = new_list
        if self._ws is not None and self.connected and new_list:
            await self._send_subscribe(self._ws, new_list)

    async def unsubscribe_tickers(self, tickers_to_remove: list[str]) -> None:
        """Remove widened/off-band/expiring tickers from local books and WS subscription."""
        if not tickers_to_remove:
            return
        remove_set = set(tickers_to_remove)
        self.tickers = [t for t in self.tickers if t not in remove_set]
        sids_to_remove = [
            sid for sid, t in list(self._sid_to_ticker.items()) if t in remove_set
        ]
        for sid in sids_to_remove:
            self._sid_to_ticker.pop(sid, None)
        for t in tickers_to_remove:
            self.orderbooks.pop(t, None)
            self._yes_books.pop(t, None)
            self._no_books.pop(t, None)
        if self._ws is not None and self.connected and sids_to_remove:
            msg = {
                "id": self._cmd_id,
                "cmd": "unsubscribe",
                "params": {"sids": sids_to_remove},
            }
            self._cmd_id += 1
            try:
                await self._ws.send(json.dumps(msg))
            except Exception as exc:
                logger.debug("WS unsubscribe send failed: %s", exc)

    async def _send_subscribe(self, ws: Any, tickers: list[str]) -> None:
        if not tickers:
            return
        channels = ["orderbook_delta", "ticker"]
        if self.signer is not None:
            channels.append("fill")
        msg = {
            "id": self._cmd_id,
            "cmd": "subscribe",
            "params": {"channels": channels, "market_tickers": tickers},
        }
        self._cmd_id += 1
        await ws.send(json.dumps(msg))

    async def _run_loop(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                headers = (
                    self.signer.headers("GET", "/trade-api/ws/v2")
                    if self.signer is not None
                    else {}
                )
                async with websockets.connect(
                    self.ws_url, additional_headers=headers, ping_interval=20, ping_timeout=10
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    self.last_error = None
                    backoff = 1.0
                    if self.tickers:
                        await self._send_subscribe(ws, self.tickers)

                    async for raw_msg in ws:
                        self.last_message_ms = int(time.time() * 1000)
                        await self._handle_message(raw_msg)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.connected = False
                self.last_error = str(exc)
                self.reconnect_count += 1
                logger.warning("Kalshi WS disconnected (%s); reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
            finally:
                self.connected = False
                self._ws = None

    def set_snapshot(self, snap: OrderbookSnapshot) -> None:
        """Replace isolated per-ticker YES/NO bid maps from an OrderbookSnapshot."""
        self._yes_books[snap.ticker] = {lvl.price: lvl.count for lvl in snap.yes_bids}
        self._no_books[snap.ticker] = {lvl.price: lvl.count for lvl in snap.no_bids}
        self.orderbooks[snap.ticker] = snap

    def _build_snapshot_from_books(self, ticker: str, ts_ms: int) -> OrderbookSnapshot:
        yes_map = self._yes_books.get(ticker, {})
        no_map = self._no_books.get(ticker, {})
        yes_bids = [
            PriceLevel(price=p, count=c) for p, c in sorted(yes_map.items(), key=lambda x: x[0])
        ]
        no_bids = [
            PriceLevel(price=p, count=c) for p, c in sorted(no_map.items(), key=lambda x: x[0])
        ]
        snap = OrderbookSnapshot(
            ticker=ticker, timestamp_ms=ts_ms, yes_bids=yes_bids, no_bids=no_bids
        )
        self.orderbooks[ticker] = snap
        return snap

    async def _handle_message(self, raw_msg: str | bytes) -> None:
        try:
            data = json.loads(raw_msg)
        except Exception:
            return

        msg_type = data.get("type")
        msg = data.get("msg") or {}
        sid = data.get("sid")
        ts_ms = int(time.time() * 1000)

        if msg_type == "error":
            self.last_error = str(msg.get("msg") or msg)
            logger.warning("Kalshi WS error message: %s", data)
            return

        if msg_type == "subscribed":
            if sid is not None and msg.get("market_ticker"):
                self._sid_to_ticker[int(sid)] = str(msg["market_ticker"])
            return

        ticker = msg.get("market_ticker") or msg.get("ticker")
        if not ticker and sid is not None:
            ticker = self._sid_to_ticker.get(int(sid))

        if msg_type in ("orderbook_snapshot", "orderbook_delta", "ticker"):
            if not ticker or (self.tickers and ticker not in self.tickers):
                return

        if msg_type == "orderbook_snapshot":
            if not ticker:
                return
            if sid is not None:
                self._sid_to_ticker[int(sid)] = str(ticker)
            snap = OrderbookSnapshot.from_kalshi_orderbook_fp(ticker, msg, ts_ms)
            self.set_snapshot(snap)
            if self.on_orderbook:
                await self.on_orderbook(snap)

        elif msg_type == "orderbook_delta":
            if not ticker:
                return
            side = msg.get("side")
            price_raw = msg.get("price_dollars") or msg.get("price")
            delta_raw = msg.get("delta_fp") or msg.get("delta")
            if price_raw is None or delta_raw is None or side not in ("yes", "no"):
                return

            price = Decimal(str(price_raw))
            if price > Decimal("1"):
                price = price / Decimal("100")
            price = price.quantize(Decimal("0.0001"))
            delta = Decimal(str(delta_raw))

            book_dict = (
                self._yes_books.setdefault(ticker, {})
                if side == "yes"
                else self._no_books.setdefault(ticker, {})
            )
            new_qty = book_dict.get(price, Decimal("0")) + delta
            if new_qty <= Decimal("0"):
                book_dict.pop(price, None)
            else:
                book_dict[price] = new_qty

            snap = self._build_snapshot_from_books(ticker, ts_ms)
            if self.on_orderbook:
                await self.on_orderbook(snap)

        elif msg_type == "ticker":
            if not ticker or ticker in self.orderbooks:
                return
            yes_bid = msg.get("yes_bid_dollars")
            no_bid = msg.get("no_bid_dollars")
            yes_ask = msg.get("yes_ask_dollars")
            if yes_bid is None and msg.get("yes_bid") is not None:
                yes_bid = Decimal(str(msg["yes_bid"])) / Decimal("100")
            if no_bid is None and msg.get("no_bid") is not None:
                no_bid = Decimal(str(msg["no_bid"])) / Decimal("100")
            if no_bid is None and yes_ask is not None:
                no_bid = Decimal("1.00") - Decimal(str(yes_ask))
            elif no_bid is None and msg.get("yes_ask") is not None:
                no_bid = Decimal("1.00") - (Decimal(str(msg["yes_ask"])) / Decimal("100"))
            yes_levels = (
                [PriceLevel(price=Decimal(str(yes_bid)), count=Decimal("1"))]
                if yes_bid is not None and Decimal(str(yes_bid)) > Decimal("0")
                else []
            )
            no_levels = (
                [PriceLevel(price=Decimal(str(no_bid)), count=Decimal("1"))]
                if no_bid is not None and Decimal(str(no_bid)) > Decimal("0")
                else []
            )
            if yes_levels or no_levels:
                snap = OrderbookSnapshot(
                    ticker=ticker,
                    timestamp_ms=ts_ms,
                    yes_bids=yes_levels,
                    no_bids=no_levels,
                )
                self.set_snapshot(snap)
                if self.on_orderbook:
                    await self.on_orderbook(snap)

        elif msg_type == "fill":
            if self.on_fill:
                await self.on_fill(msg)

    @staticmethod
    def _apply_delta(
        prev: OrderbookSnapshot, delta_msg: dict[str, Any], ts_ms: int
    ) -> OrderbookSnapshot:
        side = delta_msg.get("side")
        price_raw = delta_msg.get("price_dollars") or delta_msg.get("price")
        delta_raw = delta_msg.get("delta_fp") or delta_msg.get("delta")
        if price_raw is None or delta_raw is None or side not in ("yes", "no"):
            return prev

        price = Decimal(str(price_raw))
        if price > Decimal("1"):
            price = price / Decimal("100")
        delta = Decimal(str(delta_raw))

        target_levels = list(prev.yes_bids if side == "yes" else prev.no_bids)
        book_map = {lvl.price: lvl.count for lvl in target_levels}
        new_qty = book_map.get(price, Decimal("0")) + delta
        if new_qty <= Decimal("0"):
            book_map.pop(price, None)
        else:
            book_map[price] = new_qty

        updated_levels = [
            PriceLevel(price=p, count=c) for p, c in sorted(book_map.items(), key=lambda x: x[0])
        ]
        return OrderbookSnapshot(
            ticker=prev.ticker,
            timestamp_ms=ts_ms,
            yes_bids=updated_levels if side == "yes" else prev.yes_bids,
            no_bids=updated_levels if side == "no" else prev.no_bids,
        )
