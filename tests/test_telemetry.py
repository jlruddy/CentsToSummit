"""M1: research telemetry keeps raw LLM I/O, every AI proposal, session ids, and full history."""

import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pyarrow.parquet as pq

from kalshi_poc.config import AppSettings
from kalshi_poc.models import OrderbookSnapshot, PortfolioState, PriceLevel, StrategyProvider
from kalshi_poc.strategy_ai import GeminiStrategy
from kalshi_poc.telemetry import TelemetryManager, export_sqlite_to_parquet


def _snap(ticker: str = "KXBTC-1") -> OrderbookSnapshot:
    return OrderbookSnapshot(
        ticker=ticker,
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("50"))],
        no_bids=[PriceLevel(price=Decimal("0.49"), count=Decimal("10"))],  # ask 0.51
    )


def _portfolio() -> PortfolioState:
    return PortfolioState(
        cash_balance_dollars=Decimal("100"),
        starting_capital_dollars=Decimal("100"),
        daily_start_equity_dollars=Decimal("100"),
    )


def _strat() -> GeminiStrategy:
    return GeminiStrategy(
        AppSettings(_env_file=None, strategy_provider=StrategyProvider.GEMINI, llm_api_key="k")
    )


def test_one_malformed_item_does_not_drop_the_batch() -> None:
    strat = _strat()
    raw = json.dumps(
        [
            {"ticker": "KXBTC-1", "side": "buy", "count": "2", "limit_price": "0.51"},  # bad side
            {"ticker": "KXETH-1", "side": "bid", "count": "2", "limit_price": "0.51",
             "expected_edge_per_contract": "0.06", "confidence": "1.7"},  # confidence > 1
            {"ticker": "KXBTC-1", "side": "bid", "count": "2", "limit_price": "0.51",
             "expected_edge_per_contract": "0.06"},  # valid
            {"ticker": "KXSOL-1", "side": "bid", "count": "1", "limit_price": "0.51"},  # no book
        ]
    )
    intents, rejections = strat._parse_batch_intents_detailed(
        raw, [_snap("KXBTC-1"), _snap("KXETH-1")], 10, positions_by_ticker={}
    )
    assert [i.ticker for i in intents] == ["KXBTC-1"]
    statuses = sorted((r["status"], r["proposal"].get("ticker")) for r in rejections)
    assert statuses == [("FILTERED", "KXSOL-1"), ("PARSE_ERROR", "KXBTC-1"), ("PARSE_ERROR", "KXETH-1")]
    assert all(r["reason"] for r in rejections)


def test_unparseable_response_is_recorded_not_silently_dropped() -> None:
    intents, rejections = _strat()._parse_batch_intents_detailed(
        "Sure! Here are my trades: [oops", [_snap()], 10, positions_by_ticker={}
    )
    assert intents == []
    assert len(rejections) == 1 and rejections[0]["status"] == "PARSE_ERROR"
    assert "oops" in rejections[0]["proposal"]["raw_text"]


async def test_llm_call_and_proposals_persisted_with_raw_io(tmp_path: Path) -> None:
    strat = _strat()
    raw = json.dumps(
        [
            {"ticker": "KXBTC-1", "side": "bid", "count": "2", "limit_price": "0.51",
             "expected_edge_per_contract": "0.06", "hold_to_expiry": True, "rationale": "imb"},
            {"ticker": "KXBTC-1", "side": "ask", "count": "1", "limit_price": "0.49"},  # duplicate
        ]
    )
    with patch.object(strat, "_call_llm_batch", new=AsyncMock(return_value=(raw, 900, 80))):
        batch = await strat.evaluate_batch([_snap()], _portfolio(), force=True)

    call = batch.llm_call
    assert call is not None
    assert call["raw_response"] == raw
    assert '"KXBTC-1"' in call["user_prompt"] and call["system_prompt"]
    assert call["prompt_tokens"] == 900 and call["error"] == ""
    assert {d["decision_status"] for d in batch.decisions} == {"PROPOSED", "FILTERED"}
    assert all(d["llm_call_id"] == call["llm_call_id"] for d in batch.decisions)

    telem = TelemetryManager(tmp_path / "t.db")
    await telem.initialize()
    try:
        await telem.log_llm_call(call)
        for d in batch.decisions:
            await telem.log_ai_decision(d)
    finally:
        await telem.close()

    con = sqlite3.connect(tmp_path / "t.db")
    row = con.execute("SELECT session_id, raw_response, prompt_tokens FROM llm_calls").fetchone()
    assert row[0] and row[1] == raw and row[2] == 900
    dec = con.execute(
        "SELECT session_id, decision_status, expected_edge_per_contract, hold_to_expiry, "
        "llm_call_id, raw_proposal_json FROM ai_decisions WHERE decision_status='PROPOSED'"
    ).fetchone()
    assert dec[0] and dec[2] == "0.06" and dec[3] == 1 and dec[4] == call["llm_call_id"]
    assert json.loads(dec[5])["rationale"] == "imb"
    con.close()


async def test_llm_errors_are_recorded_as_llm_calls() -> None:
    strat = _strat()
    with patch.object(strat, "_call_llm_batch", new=AsyncMock(side_effect=RuntimeError("boom"))):
        batch = await strat.evaluate_batch([_snap()], _portfolio(), force=True)
    assert batch.llm_call is not None
    assert "boom" in batch.llm_call["error"]
    assert batch.decisions[0]["llm_call_id"] == batch.llm_call["llm_call_id"]


async def test_orderbook_history_not_pruned_and_session_ids_everywhere(tmp_path: Path) -> None:
    telem = TelemetryManager(tmp_path / "t.db")
    await telem.initialize()
    try:
        assert telem._db is not None
        await telem._db.executemany(
            "INSERT INTO orderbook_snapshots (timestamp_ms, ticker, yes_bids_json, no_bids_json) "
            "VALUES (?, 'OLD', '[]', '[]')",
            [(i,) for i in range(5100)],
        )
        await telem._db.commit()
        snap = _snap()
        snap.timestamp_ms = 10**12
        await telem.log_orderbook(snap)
    finally:
        await telem.close()
    con = sqlite3.connect(tmp_path / "t.db")
    assert con.execute("SELECT COUNT(*) FROM orderbook_snapshots").fetchone()[0] == 5101
    for table in ("orderbook_snapshots", "ai_decisions", "risk_evaluations", "llm_calls",
                  "exchange_events", "orders_and_fills", "equity_curve"):
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        assert "session_id" in cols, table
    assert con.execute(
        "SELECT session_id FROM orderbook_snapshots WHERE ticker='KXBTC-1'"
    ).fetchone()[0]
    con.close()


async def test_parquet_export_is_complete_and_typed(tmp_path: Path) -> None:
    telem = TelemetryManager(tmp_path / "t.db")
    await telem.initialize()
    await telem.close()
    con = sqlite3.connect(tmp_path / "t.db")
    con.executemany(
        "INSERT INTO equity_curve (timestamp_ms, cash_balance_dollars, current_equity_dollars, "
        "session_net_pnl_dollars, daily_pnl_dollars, open_risk_collateral_dollars, equity_floor_dollars) "
        "VALUES (?, '1', '1', '0', '0', '0', '0')",
        [(i,) for i in range(60_123)],
    )
    con.commit()
    con.close()
    files = {f.name: f for f in export_sqlite_to_parquet(tmp_path / "t.db", tmp_path / "out")}
    assert {"llm_calls.parquet", "exchange_events.parquet"} <= set(files)
    tbl = pq.read_table(files["equity_curve.parquet"])
    assert tbl.num_rows == 60_123
    assert str(tbl.schema.field("timestamp_ms").type) == "int64"
    assert tbl.column("timestamp_ms")[0].as_py() == 0  # oldest first
    assert pq.read_table(files["llm_calls.parquet"]).num_rows == 0
