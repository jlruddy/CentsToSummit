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
    "risk_evaluations",
    "orders_and_fills",
    "equity_curve",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    estimated_cost_usd TEXT DEFAULT '0.000000'
);

CREATE TABLE IF NOT EXISTS risk_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
"""


class TelemetryManager:
    """SQLite WAL telemetry store + asyncio live event bus + Parquet exporter."""

    def __init__(self, db_path: str | Path = "data/kalshi_telemetry.db") -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._write_lock = asyncio.Lock()
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
        """Return up to `limit` most recent AI_DECISION, RISK_GATE, and EXECUTION events (oldest-first)."""
        if self._db is None:
            return []
        events: list[dict[str, Any]] = []

        async with self._db.execute(
            """
            SELECT timestamp_ms, intent_id, ticker, side, count, limit_price,
                   expected_edge_dollars, confidence, inference_latency_ms, provider,
                   rationale, prompt_tokens, completion_tokens, estimated_cost_usd
            FROM ai_decisions
            ORDER BY timestamp_ms DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ) as cur:
            for r in await cur.fetchall():
                side_val = r[3]
                action = "HOLD" if side_val == "hold" else ("BUY" if side_val == "bid" else "SELL")
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
            ORDER BY timestamp_ms DESC, id DESC
            LIMIT ?
            """,
            (limit,),
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
            ORDER BY timestamp_ms DESC, id DESC
            LIMIT ?
            """,
            (limit,),
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
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.executescript(SCHEMA_SQL)
        for tbl in ("orders_and_fills", "equity_curve"):
            try:
                await self._db.execute(f"ALTER TABLE {tbl} ADD COLUMN session_id TEXT DEFAULT ''")
            except Exception:
                pass
        for col_def in (
            "prompt_tokens INTEGER DEFAULT 0",
            "completion_tokens INTEGER DEFAULT 0",
            "estimated_cost_usd TEXT DEFAULT '0.000000'",
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
        if self._db is not None:
            async with self._write_lock:
                await self._db.execute(
                    """
                    INSERT INTO orderbook_snapshots (
                        timestamp_ms, ticker, best_yes_bid, best_yes_ask,
                        best_no_bid, best_no_ask, spread, yes_bids_json, no_bids_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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

        if self._db is not None:
            async with self._write_lock:
                await self._db.execute(
                    """
                    INSERT INTO ai_decisions (
                        timestamp_ms, intent_id, ticker, side, count, limit_price,
                        expected_edge_dollars, confidence, inference_latency_ms, provider,
                        rationale, prompt_tokens, completion_tokens, estimated_cost_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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
                    ),
                )
                await self._db.commit()

        await self.broadcast("AI_DECISION", payload)

    async def log_risk_evaluation(self, evaluation: RiskEvaluation) -> None:
        if self._db is not None:
            async with self._write_lock:
                await self._db.execute(
                    """
                    INSERT INTO risk_evaluations (
                        timestamp_ms, intent_id, ticker, approved, rejection_reasons_json,
                        estimated_fill_price, estimated_slippage_dollars,
                        estimated_entry_fee_dollars, estimated_round_trip_fee_dollars,
                        worst_case_equity_after
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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


def export_sqlite_to_parquet(db_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Export all 5 SQLite WAL telemetry tables to Parquet files using DuckDB + PyArrow."""
    db_file = Path(db_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not db_file.exists():
        # Ensure empty tables exist before exporting
        conn = sqlite3.connect(db_file)
        conn.executescript(SCHEMA_SQL)
        conn.close()

    exported: list[Path] = []
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("INSTALL sqlite; LOAD sqlite;")
    except Exception:
        pass

    # Fallback-safe: read via sqlite3 + pyarrow if duckdb sqlite extension is unavailable offline
    try:
        con.execute(f"ATTACH '{db_file.as_posix()}' AS src (TYPE sqlite);")
        for table in TABLES:
            out_file = out_dir / f"{table}.parquet"
            con.execute(f"COPY src.{table} TO '{out_file.as_posix()}' (FORMAT PARQUET);")
            exported.append(out_file)
    except Exception:
        import pyarrow as pa
        import pyarrow.parquet as pq

        sconn = sqlite3.connect(db_file)
        sconn.row_factory = sqlite3.Row
        try:
            for table in TABLES:
                rows = sconn.execute(f"SELECT * FROM {table}").fetchall()
                cols = [d[0] for d in sconn.execute(f"SELECT * FROM {table} LIMIT 0").description]
                data = {col: [r[col] for r in rows] for col in cols}
                arrow_tbl = pa.table(data)
                out_file = out_dir / f"{table}.parquet"
                pq.write_table(arrow_tbl, out_file)
                exported.append(out_file)
        finally:
            sconn.close()
    finally:
        con.close()

    return exported


def export_parquet_zip(db_path: str | Path, output_dir: str | Path) -> Path:
    """Export all Parquet files and bundle them into a single zip archive."""
    files = export_sqlite_to_parquet(db_path, output_dir)
    zip_path = Path(output_dir) / "kalshi_telemetry_parquet.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    return zip_path
