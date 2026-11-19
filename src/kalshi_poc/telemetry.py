"""Async SQLite (WAL mode) event logger, live event bus, and Parquet exporter."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiosqlite
import duckdb

from kalshi_poc.models import (
    ExecutionRecord,
    OrderbookSnapshot,
    PortfolioState,
    RiskEvaluation,
    TradeIntent,
)

TABLES = (
    "orderbook_snapshots",
    "ai_decisions",
    "llm_calls",
    "risk_evaluations",
    "orders_and_fills",
    "equity_curve",
    "exchange_events",
)
EXPORT_CHUNK_ROWS = 20_000

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    timestamp_ms INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    best_yes_bid TEXT,
    best_yes_ask TEXT,
    best_no_bid TEXT,
    best_no_ask TEXT,
    spread TEXT,
    yes_bids_json TEXT NOT NULL,
    no_bids_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    timestamp_ms INTEGER NOT NULL,
    intent_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    count TEXT NOT NULL,
    limit_price TEXT NOT NULL,
    expected_edge_dollars TEXT NOT NULL,
    confidence TEXT NOT NULL,
    inference_latency_ms INTEGER NOT NULL,
    provider TEXT NOT NULL,
    rationale TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    estimated_cost_usd TEXT DEFAULT '0.000000',
    expected_edge_per_contract TEXT DEFAULT '',
    hold_to_expiry INTEGER DEFAULT 0,
    decision_status TEXT DEFAULT '',
    llm_call_id TEXT DEFAULT '',
    raw_proposal_json TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    timestamp_ms INTEGER NOT NULL,
    llm_call_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT DEFAULT '',
    trigger_reason TEXT DEFAULT '',
    system_prompt TEXT DEFAULT '',
    user_prompt TEXT DEFAULT '',
    raw_response TEXT DEFAULT '',
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    estimated_cost_usd TEXT DEFAULT '0.000000',
    latency_ms INTEGER DEFAULT 0,
    error TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS risk_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    timestamp_ms INTEGER NOT NULL,
    intent_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    approved INTEGER NOT NULL,
    rejection_reasons_json TEXT NOT NULL,
    estimated_fill_price TEXT NOT NULL,
    estimated_slippage_dollars TEXT NOT NULL,
    estimated_entry_fee_dollars TEXT NOT NULL,
    estimated_round_trip_fee_dollars TEXT NOT NULL,
    worst_case_equity_after TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders_and_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    timestamp_ms INTEGER NOT NULL,
    order_id TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    mode TEXT NOT NULL,
    side TEXT NOT NULL,
    count TEXT NOT NULL,
    requested_price TEXT NOT NULL,
    fill_price TEXT NOT NULL,
    slippage_dollars TEXT NOT NULL,
    fee_dollars TEXT NOT NULL,
    is_maker INTEGER NOT NULL,
    status TEXT NOT NULL,
    shadow_fill_price TEXT,
    shadow_fee_dollars TEXT
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    timestamp_ms INTEGER NOT NULL,
    cash_balance_dollars TEXT NOT NULL,
    current_equity_dollars TEXT NOT NULL,
    session_net_pnl_dollars TEXT NOT NULL,
    daily_pnl_dollars TEXT NOT NULL,
    open_risk_collateral_dollars TEXT NOT NULL,
    equity_floor_dollars TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exchange_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    timestamp_ms INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    ticker TEXT DEFAULT '',
    order_id TEXT DEFAULT '',
    trade_id TEXT DEFAULT '',
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS engine_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ai_decisions_ts ON ai_decisions(timestamp_ms DESC);
CREATE INDEX IF NOT EXISTS idx_risk_evaluations_ts ON risk_evaluations(timestamp_ms DESC);
CREATE INDEX IF NOT EXISTS idx_orders_and_fills_ts ON orders_and_fills(timestamp_ms DESC);
CREATE INDEX IF NOT EXISTS idx_equity_curve_ts ON equity_curve(timestamp_ms DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls(timestamp_ms DESC);
"""


class TelemetryManager:
    """SQLite WAL telemetry store + asyncio live event bus + Parquet exporter."""

    def __init__(self, db_path: str | Path = "data/kalshi_telemetry.db") -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._write_lock = asyncio.Lock()
        self._last_ob_db_write_ms: dict[str, int] = {}
        self.session_id: str = uuid.uuid4().hex[:12]
        self.session_start_ts: int = int(time.time() * 1000)
        self.session_llm_spend_usd: Decimal = Decimal("0.000000")

    def new_session(self, session_start_ts: int | None = None) -> str:
        """Rotate session_id and session_start_ts so new rows are isolated from prior runs."""
        self.session_id = uuid.uuid4().hex[:12]
        self.session_start_ts = (
            session_start_ts if session_start_ts is not None else int(time.time() * 1000)
        )
        self.session_llm_spend_usd = Decimal("0.000000")
        return self.session_id

    async def get_session_equity_history(self, limit: int = 200) -> list[dict[str, Any]]:
        """Query equity_curve strictly for the active session (timestamp_ms >= session_start_ts)."""
        if self._db is None:
            return []
        async with self._db.execute(
            """
            SELECT timestamp_ms, current_equity_dollars, equity_floor_dollars
            FROM equity_curve
            WHERE timestamp_ms >= ?
            ORDER BY timestamp_ms ASC
            LIMIT ?
            """,
            (self.session_start_ts, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {"timestamp_ms": r[0], "equity": r[1], "floor": r[2]} for r in rows
        ]

    async def get_recent_activity(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return up to `limit` most recent AI_DECISION, RISK_GATE, and EXECUTION events (oldest-first).

        Scoped to the active session: a restart or shadow reset rotates session_id, so prior runs'
        events (e.g. a stale "Max open markets reached") are never replayed as current activity.
        History stays in SQLite / Parquet exports.
        """
        if self._db is None:
            return []
        events: list[dict[str, Any]] = []

        async with self._db.execute(
            """
            SELECT timestamp_ms, intent_id, ticker, side, count, limit_price,
                   expected_edge_dollars, confidence, inference_latency_ms, provider,
                   rationale, prompt_tokens, completion_tokens, estimated_cost_usd,
                   decision_status
            FROM ai_decisions
            WHERE session_id = ?
            ORDER BY timestamp_ms DESC, id DESC
            LIMIT ?
            """,
            (self.session_id, limit),
        ) as cur:
            for r in await cur.fetchall():
                side_val = r[3]
                action = "HOLD" if side_val == "hold" else ("BUY" if side_val == "bid" else "SELL")
                if r[14] in ("FILTERED", "PARSE_ERROR"):
                    action = "REJECTED"
                events.append(
                    {
                        "type": "AI_DECISION",
                        "data": {
                            "timestamp_ms": r[0],
                            "intent_id": r[1],
                            "ticker": r[2],
                            "side": side_val,
                            "action": action,
                            "count": r[4],
                            "limit_price": r[5],
                            "expected_edge_dollars": r[6],
                            "confidence": r[7],
                            "inference_latency_ms": r[8],
                            "provider": r[9],
                            "rationale": r[10],
                            "prompt_tokens": r[11] or 0,
                            "completion_tokens": r[12] or 0,
                            "estimated_cost_usd": r[13] or "0.000000",
                            "decision_status": r[14] or "",
                        },
                    }
                )

        async with self._db.execute(
            """
            SELECT timestamp_ms, intent_id, ticker, approved, rejection_reasons_json,
                   estimated_fill_price, estimated_slippage_dollars,
                   estimated_entry_fee_dollars, estimated_round_trip_fee_dollars,
                   worst_case_equity_after
            FROM risk_evaluations
            WHERE session_id = ?
            ORDER BY timestamp_ms DESC, id DESC
            LIMIT ?
            """,
            (self.session_id, limit),
        ) as cur:
            for r in await cur.fetchall():
                events.append(
                    {
                        "type": "RISK_GATE",
                        "data": {
                            "timestamp_ms": r[0],
                            "intent_id": r[1],
                            "ticker": r[2],
                            "approved": bool(r[3]),
                            "rejection_reasons": json.loads(r[4] or "[]"),
                            "estimated_fill_price": r[5],
                            "estimated_slippage_dollars": r[6],
                            "estimated_entry_fee_dollars": r[7],
                            "estimated_round_trip_fee_dollars": r[8],
                            "worst_case_equity_after": r[9],
                        },
                    }
                )

        async with self._db.execute(
            """
            SELECT timestamp_ms, order_id, client_order_id, intent_id, ticker, mode, side,
                   count, requested_price, fill_price, slippage_dollars, fee_dollars,
                   is_maker, status, shadow_fill_price, shadow_fee_dollars
            FROM orders_and_fills
            WHERE session_id = ?
            ORDER BY timestamp_ms DESC, id DESC
            LIMIT ?
            """,
            (self.session_id, limit),
        ) as cur:
            for r in await cur.fetchall():
                events.append(
                    {
                        "type": "EXECUTION",
                        "data": {
                            "timestamp_ms": r[0],
                            "order_id": r[1],
                            "client_order_id": r[2],
                            "intent_id": r[3],
                            "ticker": r[4],
                            "mode": r[5],
                            "side": r[6],
                            "count": r[7],
                            "requested_price": r[8],
                            "fill_price": r[9],
                            "slippage_dollars": r[10],
                            "fee_dollars": r[11],
                            "is_maker": bool(r[12]),
                            "status": r[13],
                            "shadow_fill_price": r[14],
                            "shadow_fee_dollars": r[15],
                        },
                    }
                )

        events.sort(key=lambda e: int(e["data"]["timestamp_ms"]), reverse=True)
        recent = events[:limit]
        recent.reverse()  # oldest-first so client unshift() ends up newest-first
        return recent

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA busy_timeout=5000;")
        await self._db.execute("PRAGMA mmap_size=0;")
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.executescript(SCHEMA_SQL)
        for tbl in (
            "orders_and_fills",
            "equity_curve",
            "orderbook_snapshots",
            "ai_decisions",
            "risk_evaluations",
        ):
            try:
                await self._db.execute(f"ALTER TABLE {tbl} ADD COLUMN session_id TEXT DEFAULT ''")
            except Exception:
                pass
        for col_def in (
            "prompt_tokens INTEGER DEFAULT 0",
            "completion_tokens INTEGER DEFAULT 0",
            "estimated_cost_usd TEXT DEFAULT '0.000000'",
            "expected_edge_per_contract TEXT DEFAULT ''",
            "hold_to_expiry INTEGER DEFAULT 0",
            "decision_status TEXT DEFAULT ''",
            "llm_call_id TEXT DEFAULT ''",
            "raw_proposal_json TEXT DEFAULT ''",
        ):
            try:
                await self._db.execute(f"ALTER TABLE ai_decisions ADD COLUMN {col_def}")
            except Exception:
                pass
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    async def broadcast(self, event_type: str, payload: dict[str, Any]) -> None:
        msg = {"type": event_type, "data": payload}
        for q in list(self._subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    async def log_orderbook(self, snap: OrderbookSnapshot) -> None:
        yes_bids = [[str(l.price), str(l.count)] for l in snap.yes_bids[-5:]]
        no_bids = [[str(l.price), str(l.count)] for l in snap.no_bids[-5:]]
        last_write = self._last_ob_db_write_ms.get(snap.ticker, 0)
        if self._db is not None and (snap.timestamp_ms - last_write) >= 10000:
            self._last_ob_db_write_ms[snap.ticker] = snap.timestamp_ms
            async with self._write_lock:
                await self._db.execute(
                    """
                    INSERT INTO orderbook_snapshots (
                        session_id, timestamp_ms, ticker, best_yes_bid, best_yes_ask,
                        best_no_bid, best_no_ask, spread, yes_bids_json, no_bids_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.session_id,
                        snap.timestamp_ms,
                        snap.ticker,
                        str(snap.best_yes_bid) if snap.best_yes_bid is not None else None,
                        str(snap.best_yes_ask) if snap.best_yes_ask is not None else None,
                        str(snap.best_no_bid) if snap.best_no_bid is not None else None,
                        str(snap.best_no_ask) if snap.best_no_ask is not None else None,
                        str(snap.yes_spread) if snap.yes_spread is not None else None,
                        json.dumps(yes_bids),
                        json.dumps(no_bids),
                    ),
                )
                await self._db.commit()

        await self.broadcast(
            "ORDERBOOK",
            {
                "timestamp_ms": snap.timestamp_ms,
                "ticker": snap.ticker,
                "best_yes_bid": str(snap.best_yes_bid) if snap.best_yes_bid else None,
                "best_yes_ask": str(snap.best_yes_ask) if snap.best_yes_ask else None,
                "best_no_bid": str(snap.best_no_bid) if snap.best_no_bid else None,
                "best_no_ask": str(snap.best_no_ask) if snap.best_no_ask else None,
                "spread": str(snap.yes_spread) if snap.yes_spread else None,
                "top5_yes_bids": [[str(l.price), str(l.count)] for l in reversed(snap.yes_bids[-5:])],
                "top5_yes_asks": [[str(l.price), str(l.count)] for l in snap.effective_yes_asks[:5]],
            },
        )

    async def log_ai_decision(self, decision: TradeIntent | dict[str, Any]) -> None:
        if isinstance(decision, TradeIntent):
            payload = decision.model_dump(mode="json")
            payload["action"] = "BUY" if decision.side.value == "bid" else "SELL"
        else:
            payload = dict(decision)

        prompt_tokens = int(payload.get("prompt_tokens", 0) or 0)
        completion_tokens = int(payload.get("completion_tokens", 0) or 0)
        cost_dec = Decimal(str(payload.get("estimated_cost_usd", "0.000000") or "0.000000"))
        self.session_llm_spend_usd += cost_dec
        payload["prompt_tokens"] = prompt_tokens
        payload["completion_tokens"] = completion_tokens
        payload["estimated_cost_usd"] = f"{cost_dec:.6f}"
        side_val = str(payload.get("side", "hold"))
        payload.setdefault("decision_status", "HOLD" if side_val == "hold" else "PROPOSED")
        raw_prop = payload.pop("raw_proposal", None)
        raw_prop_json = json.dumps(raw_prop, default=str) if raw_prop is not None else ""
        edge_pc = payload.get("expected_edge_per_contract")

        if self._db is not None:
            async with self._write_lock:
                await self._db.execute(
                    """
                    INSERT INTO ai_decisions (
                        session_id, timestamp_ms, intent_id, ticker, side, count, limit_price,
                        expected_edge_dollars, confidence, inference_latency_ms, provider,
                        rationale, prompt_tokens, completion_tokens, estimated_cost_usd,
                        expected_edge_per_contract, hold_to_expiry, decision_status,
                        llm_call_id, raw_proposal_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.session_id,
                        int(payload["timestamp_ms"]),
                        str(payload["intent_id"]),
                        str(payload["ticker"]),
                        str(payload.get("side", "hold")),
                        str(payload.get("count", "0")),
                        str(payload.get("limit_price", "0.0000")),
                        str(payload.get("expected_edge_dollars", "0.00")),
                        str(payload.get("confidence", "0.00")),
                        int(payload.get("inference_latency_ms", 0)),
                        str(payload.get("provider", "unknown")),
                        str(payload.get("rationale", "")),
                        prompt_tokens,
                        completion_tokens,
                        f"{cost_dec:.6f}",
                        str(edge_pc) if edge_pc is not None else "",
                        1 if payload.get("hold_to_expiry") else 0,
                        str(payload.get("decision_status", "")),
                        str(payload.get("llm_call_id", "") or ""),
                        raw_prop_json,
                    ),
                )
                await self._db.commit()

        await self.broadcast("AI_DECISION", payload)

    async def log_llm_call(self, call: dict[str, Any]) -> None:
        """Persist one LLM request/response verbatim (prompts, raw text, tokens, cost, error)."""
        if self._db is None:
            return
        async with self._write_lock:
            await self._db.execute(
                """
                INSERT INTO llm_calls (
                    session_id, timestamp_ms, llm_call_id, provider, model, trigger_reason,
                    system_prompt, user_prompt, raw_response, prompt_tokens, completion_tokens,
                    estimated_cost_usd, latency_ms, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    int(call.get("timestamp_ms") or time.time() * 1000),
                    str(call.get("llm_call_id", "")),
                    str(call.get("provider", "")),
                    str(call.get("model", "")),
                    str(call.get("trigger_reason", "")),
                    str(call.get("system_prompt", "")),
                    str(call.get("user_prompt", "")),
                    str(call.get("raw_response", "")),
                    int(call.get("prompt_tokens", 0) or 0),
                    int(call.get("completion_tokens", 0) or 0),
                    str(call.get("estimated_cost_usd", "0.000000")),
                    int(call.get("latency_ms", 0) or 0),
                    str(call.get("error", "")),
                ),
            )
            await self._db.commit()

    async def log_risk_evaluation(self, evaluation: RiskEvaluation) -> None:
        if self._db is not None:
            async with self._write_lock:
                await self._db.execute(
                    """
                    INSERT INTO risk_evaluations (
                        session_id, timestamp_ms, intent_id, ticker, approved,
                        rejection_reasons_json, estimated_fill_price, estimated_slippage_dollars,
                        estimated_entry_fee_dollars, estimated_round_trip_fee_dollars,
                        worst_case_equity_after
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.session_id,
                        evaluation.timestamp_ms,
                        evaluation.intent_id,
                        evaluation.ticker,
                        1 if evaluation.approved else 0,
                        json.dumps(evaluation.rejection_reasons),
                        str(evaluation.estimated_fill_price),
                        str(evaluation.estimated_slippage_dollars),
                        str(evaluation.estimated_entry_fee_dollars),
                        str(evaluation.estimated_round_trip_fee_dollars),
                        str(evaluation.worst_case_equity_after),
                    ),
                )
                await self._db.commit()

        await self.broadcast("RISK_GATE", evaluation.model_dump(mode="json"))

    async def log_execution(self, record: ExecutionRecord) -> None:
        if self._db is not None:
            async with self._write_lock:
                await self._db.execute(
                    """
                    INSERT INTO orders_and_fills (
                        session_id, timestamp_ms, order_id, client_order_id, intent_id, ticker, mode, side,
                        count, requested_price, fill_price, slippage_dollars, fee_dollars,
                        is_maker, status, shadow_fill_price, shadow_fee_dollars
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.session_id,
                        record.timestamp_ms,
                        record.order_id,
                        record.client_order_id,
                        record.intent_id,
                        record.ticker,
                        record.mode.value,
                        record.side.value,
                        str(record.count),
                        str(record.requested_price),
                        str(record.fill_price),
                        str(record.slippage_dollars),
                        str(record.fee_dollars),
                        1 if record.is_maker else 0,
                        record.status,
                        str(record.shadow_fill_price) if record.shadow_fill_price is not None else None,
                        str(record.shadow_fee_dollars) if record.shadow_fee_dollars is not None else None,
                    ),
                )
                await self._db.commit()

        await self.broadcast("EXECUTION", record.model_dump(mode="json"))

    async def save_state(self, key: str, value: dict[str, Any]) -> None:
        """Durably upsert a small engine-state record (kill-switch, daily baseline, last mode)."""
        if self._db is None:
            return
        async with self._write_lock:
            await self._db.execute(
                """
                INSERT INTO engine_state (key, value_json, updated_ms) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json,
                                               updated_ms = excluded.updated_ms
                """,
                (key, json.dumps(value, default=str), int(time.time() * 1000)),
            )
            await self._db.commit()

    async def load_state(self, key: str) -> dict[str, Any] | None:
        if self._db is None:
            return None
        async with self._db.execute(
            "SELECT value_json FROM engine_state WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    async def log_exchange_event(
        self,
        event_type: str,
        payload: Any,
        *,
        ticker: str = "",
        order_id: str = "",
        trade_id: str = "",
        timestamp_ms: int | None = None,
    ) -> None:
        """Persist raw exchange payloads (fills, order responses) verbatim for later verification."""
        if self._db is None:
            return
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        async with self._write_lock:
            await self._db.execute(
                """
                INSERT INTO exchange_events (
                    session_id, timestamp_ms, event_type, ticker, order_id, trade_id, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    ts,
                    event_type,
                    ticker,
                    order_id,
                    trade_id,
                    json.dumps(payload, default=str),
                ),
            )
            await self._db.commit()

    async def log_equity(
        self, portfolio: PortfolioState, equity_floor: str, timestamp_ms: int
    ) -> None:
        payload = {
            "timestamp_ms": timestamp_ms,
            "cash_balance_dollars": str(portfolio.cash_balance_dollars),
            "current_equity_dollars": str(portfolio.current_equity_dollars),
            "session_net_pnl_dollars": str(portfolio.session_net_pnl_dollars),
            "daily_pnl_dollars": str(portfolio.daily_pnl_dollars),
            "open_risk_collateral_dollars": str(portfolio.open_risk_collateral_dollars),
            "equity_floor_dollars": equity_floor,
        }
        if self._db is not None:
            async with self._write_lock:
                await self._db.execute(
                    """
                    INSERT INTO equity_curve (
                        session_id, timestamp_ms, cash_balance_dollars, current_equity_dollars,
                        session_net_pnl_dollars, daily_pnl_dollars,
                        open_risk_collateral_dollars, equity_floor_dollars
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (self.session_id, *payload.values()),
                )
                await self._db.commit()

        await self.broadcast("EQUITY", payload)


    async def export_parquet_zip_async(self, output_dir: str | Path) -> Path:
        """Flush SQLite connection and export to Parquet without holding the live write lock."""
        async with self._write_lock:
            if self._db is not None:
                await self._db.commit()
        return await asyncio.to_thread(export_parquet_zip, self.db_path, output_dir)


def _arrow_schema_for_table(conn: sqlite3.Connection, table: str) -> Any:
    """Build a stable Arrow schema from SQLite declared column types (INTEGER -> int64, else str)."""
    import pyarrow as pa

    fields = []
    for _, name, decl_type, *_ in conn.execute(f"PRAGMA table_info({table})"):
        arrow_type = pa.int64() if "INT" in str(decl_type).upper() else pa.string()
        fields.append(pa.field(name, arrow_type))
    return pa.schema(fields)


def export_sqlite_to_parquet(db_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Export every telemetry table in full (oldest first) to Parquet, streaming in chunks."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    db_file = Path(db_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not db_file.exists():
        conn = sqlite3.connect(db_file, timeout=10.0)
        conn.execute("PRAGMA mmap_size=0;")
        conn.executescript(SCHEMA_SQL)
        conn.close()

    exported: list[Path] = []
    sconn = sqlite3.connect(db_file, timeout=10.0)
    sconn.execute("PRAGMA busy_timeout=5000;")
    sconn.execute("PRAGMA mmap_size=0;")
    try:
        existing = {
            r[0] for r in sconn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in TABLES:
            if table not in existing:
                sconn.executescript(SCHEMA_SQL)
            schema = _arrow_schema_for_table(sconn, table)
            cols = schema.names
            out_file = out_dir / f"{table}.parquet"
            cur = sconn.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY id ASC")
            with pq.ParquetWriter(out_file, schema) as writer:
                wrote_any = False
                while True:
                    rows = cur.fetchmany(EXPORT_CHUNK_ROWS)
                    if not rows:
                        break
                    columns = list(zip(*rows))
                    data = {
                        name: [
                            (None if v is None else (int(v) if pa.types.is_integer(f.type) else str(v)))
                            for v in columns[i]
                        ]
                        for i, (name, f) in enumerate(zip(cols, schema))
                    }
                    writer.write_table(pa.table(data, schema=schema))
                    wrote_any = True
                if not wrote_any:
                    writer.write_table(schema.empty_table())
            exported.append(out_file)
    finally:
        sconn.close()

    return exported


def export_parquet_zip(db_path: str | Path, output_dir: str | Path) -> Path:
    """Export all Parquet files and bundle them into a single zip archive."""
    files = export_sqlite_to_parquet(db_path, output_dir)
    zip_path = Path(output_dir) / "kalshi_telemetry_parquet.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    return zip_path
