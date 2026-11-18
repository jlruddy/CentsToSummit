"""H7: WS order-book integrity (seq gaps, no synthetic depth, incremental subscriptions)."""

import json
from decimal import Decimal
from unittest.mock import AsyncMock

from kalshi_poc.client import KalshiWebSocketClient


def _snapshot(sid: int, seq: int, ticker: str, yes: str = "0.48", no: str = "0.50") -> str:
    return json.dumps(
        {
            "type": "orderbook_snapshot",
            "sid": sid,
            "seq": seq,
            "msg": {"market_ticker": ticker, "yes_dollars": [[yes, "10"]], "no_dollars": [[no, "10"]]},
        }
    )


def _delta(sid: int, seq: int, ticker: str, price: str, delta: str, side: str = "yes") -> str:
    return json.dumps(
        {
            "type": "orderbook_delta",
            "sid": sid,
            "seq": seq,
            "msg": {"market_ticker": ticker, "side": side, "price_dollars": price, "delta_fp": delta},
        }
    )


async def test_contiguous_seq_applies_and_gap_forces_resync() -> None:
    ws = KalshiWebSocketClient(ws_url="wss://example.invalid")
    fake_socket = AsyncMock()
    ws._ws = fake_socket
    ws.connected = True
    await ws._handle_message(_snapshot(1, 1, "T-A"))
    await ws._handle_message(_snapshot(1, 2, "T-B"))  # same sid, contiguous
    await ws._handle_message(_delta(1, 3, "T-A", "0.49", "5"))
    assert ws.orderbooks["T-A"].best_yes_bid == Decimal("0.4900")

    # seq 4 is lost -> seq 5 must not be applied; all books dropped and the socket recycled
    await ws._handle_message(_delta(1, 5, "T-A", "0.47", "-10"))
    assert ws.orderbooks == {}
    assert ws.seq_gap_count == 1
    fake_socket.close.assert_awaited_once()
    assert "seq gap" in (ws.last_error or "")

    # Fresh snapshot after resubscribe re-establishes the book on a new sid
    await ws._handle_message(_snapshot(7, 1, "T-A", yes="0.40"))
    assert ws.orderbooks["T-A"].best_yes_bid == Decimal("0.40")


async def test_ticker_messages_never_create_synthetic_depth() -> None:
    ws = KalshiWebSocketClient(ws_url="wss://example.invalid")
    await ws._handle_message(
        json.dumps(
            {"type": "ticker", "sid": 3, "msg": {"market_ticker": "T-C", "yes_bid_dollars": "0.45", "yes_ask_dollars": "0.47"}}
        )
    )
    assert "T-C" not in ws.orderbooks


async def test_update_subscriptions_only_subscribes_new_tickers() -> None:
    ws = KalshiWebSocketClient(ws_url="wss://example.invalid")
    fake_socket = AsyncMock()
    ws._ws = fake_socket
    ws.connected = True
    ws.tickers = ["T-A", "T-B"]
    await ws.update_subscriptions(["T-A", "T-B", "T-C"])
    sent = json.loads(fake_socket.send.await_args.args[0])
    assert sent["cmd"] == "subscribe"
    assert sent["params"]["market_tickers"] == ["T-C"]


async def test_unsubscribe_never_drops_a_sid_shared_with_kept_tickers() -> None:
    ws = KalshiWebSocketClient(ws_url="wss://example.invalid")
    fake_socket = AsyncMock()
    ws._ws = fake_socket
    ws.connected = True
    ws.tickers = ["T-A", "T-B", "T-C"]
    await ws._handle_message(_snapshot(1, 1, "T-A"))
    await ws._handle_message(_snapshot(1, 2, "T-B"))  # sid 1 shared by A and B
    await ws._handle_message(_snapshot(2, 1, "T-C"))  # sid 2 only C

    await ws.unsubscribe_tickers(["T-A", "T-C"])
    sent = json.loads(fake_socket.send.await_args.args[0])
    assert sent["cmd"] == "unsubscribe"
    assert sent["params"]["sids"] == [2]
    assert "T-B" in ws.orderbooks and "T-A" not in ws.orderbooks
    assert ws.tickers == ["T-B"]


async def test_disconnect_drops_books_built_from_the_stream() -> None:
    ws = KalshiWebSocketClient(ws_url="wss://example.invalid")
    await ws._handle_message(_snapshot(1, 1, "T-A"))
    ws._on_stream_lost()
    assert ws.orderbooks == {}
    # seq tracking restarts on the new connection
    await ws._handle_message(_snapshot(9, 1, "T-A"))
    await ws._handle_message(_delta(9, 2, "T-A", "0.49", "5"))
    assert ws.orderbooks["T-A"].best_yes_bid == Decimal("0.4900")
