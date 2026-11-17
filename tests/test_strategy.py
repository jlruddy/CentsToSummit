"""Test position-aware exit logic and liquidity/budget-aware sizing in StubMomentumStrategy."""

from decimal import Decimal

from kalshi_poc.config import AppSettings
from kalshi_poc.models import (
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    Position,
    PriceLevel,
)
from kalshi_poc.risk_gate import RiskGate
from kalshi_poc.strategy_ai import StubMomentumStrategy


def _book(
    ticker: str = "KXBTC-26NOV1622-T91000",
    yes_bid: str = "0.48",
    yes_count: str = "50",
    no_bid: str = "0.49",
    no_count: str = "10",
) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        ticker=ticker,
        timestamp_ms=1700000000000,
        yes_bids=[PriceLevel(price=Decimal(yes_bid), count=Decimal(yes_count))],
        no_bids=[PriceLevel(price=Decimal(no_bid), count=Decimal(no_count))],
    )


async def test_strategy_sizes_down_to_remaining_budget_and_passes_risk_gate() -> None:
    settings = AppSettings(
        _env_file=None,
        initial_capital=Decimal("12.00"),
        equity_floor=Decimal("10.50"),  # $1.50 deployable -> affords 2 contracts @ $0.51 + fee
        max_position_dollars_per_market=Decimal("2.50"),
    )
    strat = StubMomentumStrategy(settings)
    gate = RiskGate(settings)
    portfolio = PortfolioState(
        cash_balance_dollars=Decimal("12.00"),
        starting_capital_dollars=Decimal("12.00"),
        daily_start_equity_dollars=Decimal("12.00"),
    )
    snap = _book(yes_bid="0.48", yes_count="50", no_bid="0.49", no_count="10")
    intent = await strat.generate_intent(snap, portfolio, positions_by_ticker=portfolio.positions)

    assert intent is not None
    assert intent.side == OrderSide.BID
    assert intent.count == Decimal("2")
    eval_res = gate.evaluate(intent, snap, portfolio)
    assert eval_res.approved is True


async def test_strategy_exits_on_take_profit_and_avoids_stacking() -> None:
    settings = AppSettings(
        _env_file=None,
        initial_capital=Decimal("100.00"),
        equity_floor=Decimal("10.00"),
        max_position_dollars_per_market=Decimal("2.50"),
    )
    strat = StubMomentumStrategy(settings)
    ticker = "KXBTC-26NOV1622-T91000"

    # Case 1: Take-profit (+$0.05/contract >= +$0.04) triggers SELL exit
    pos = Position(
        ticker=ticker,
        net_contracts=Decimal("3"),
        avg_entry_price=Decimal("0.45"),
        cost_basis_dollars=Decimal("1.35"),
    )
    portfolio = PortfolioState(
        cash_balance_dollars=Decimal("98.65"),
        starting_capital_dollars=Decimal("100.00"),
        daily_start_equity_dollars=Decimal("100.00"),
        positions={ticker: pos},
    )
    tp_snap = _book(ticker=ticker, yes_bid="0.50", yes_count="20", no_bid="0.48", no_count="20")
    exit_intent = await strat.generate_intent(tp_snap, portfolio, positions_by_ticker=portfolio.positions)
    assert exit_intent is not None
    assert exit_intent.side == OrderSide.ASK
    assert exit_intent.count == Decimal("3")
    assert "take-profit" in exit_intent.rationale

    # Case 2: Near max_position_dollars_per_market with no exit trigger -> HOLD (no stacking)
    near_cap_pos = Position(
        ticker=ticker,
        net_contracts=Decimal("5"),
        avg_entry_price=Decimal("0.47"),
        cost_basis_dollars=Decimal("2.35"),  # remaining = $0.15 < 1 contract
    )
    portfolio.positions[ticker] = near_cap_pos
    hold_snap = _book(ticker=ticker, yes_bid="0.48", yes_count="50", no_bid="0.49", no_count="10")
    _, detail = await strat.evaluate_detail(hold_snap, portfolio, positions_by_ticker=portfolio.positions)
    assert detail["action"] == "HOLD"


async def test_llm_batch_event_gating_cooldown_and_token_cost(tmp_path) -> None:
    import json
    from unittest.mock import AsyncMock, patch
    from kalshi_poc.models import StrategyProvider
    from kalshi_poc.strategy_ai import GeminiStrategy
    from kalshi_poc.telemetry import TelemetryManager

    settings = AppSettings(
        _env_file=None,
        strategy_provider=StrategyProvider.GEMINI,
        llm_api_key="test-key",
        llm_model="gemini-2.5-flash",
        llm_min_cooldown_sec=15,
        llm_heartbeat_sec=60,
    )
    strat = GeminiStrategy(settings)
    portfolio = PortfolioState(
        cash_balance_dollars=Decimal("100.00"),
        starting_capital_dollars=Decimal("100.00"),
        daily_start_equity_dollars=Decimal("100.00"),
    )
    # Tight spread ($0.02) and high imbalance (0.67 >= 0.35) -> triggers LLM on first call
    snaps = [
        _book(ticker="KXBTC-1", yes_bid="0.49", yes_count="50", no_bid="0.49", no_count="10"),
        _book(ticker="KXETH-1", yes_bid="0.49", yes_count="20", no_bid="0.49", no_count="20"),
    ]
    mock_resp = json.dumps(
        [
            {
                "ticker": "KXBTC-1",
                "side": "bid",
                "count": "2",
                "limit_price": "0.5100",
                "expected_edge_dollars": "0.20",
                "confidence": "0.80",
                "rationale": "Batched imbalance trade",
            }
        ]
    )

    with patch.object(
        strat, "_call_llm_batch", new=AsyncMock(return_value=(mock_resp, 1000, 200))
    ) as mock_call:
        batch1 = await strat.evaluate_batch(snaps, portfolio)
        assert mock_call.call_count == 1
        assert batch1.invoked_llm is True
        assert len(batch1.intents) == 1
        assert batch1.intents[0].ticker == "KXBTC-1"
        assert batch1.prompt_tokens == 1000
        assert batch1.completion_tokens == 200
        assert batch1.estimated_cost_usd > Decimal("0")

        # Second call immediately within cooldown -> event-gated locally, 0 LLM calls
        batch2 = await strat.evaluate_batch(snaps, portfolio)
        assert mock_call.call_count == 1
        assert batch2.invoked_llm is False
        assert batch2.intents == []
        assert batch2.decisions[0]["action"] == "HOLD"
        assert "Event gate" in batch2.decisions[0]["rationale"]

    telem = TelemetryManager(tmp_path / "telem.db")
    await telem.initialize()
    try:
        for d in batch1.decisions:
            await telem.log_ai_decision(d)
        assert telem.session_llm_spend_usd == batch1.estimated_cost_usd
    finally:
        await telem.close()


async def test_gemini_thinking_config_400_fallback_and_timeout_repr() -> None:
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch
    from kalshi_poc.models import StrategyProvider
    from kalshi_poc.strategy_ai import GeminiStrategy

    settings = AppSettings(
        _env_file=None,
        strategy_provider=StrategyProvider.GEMINI,
        llm_api_key="test-key",
        llm_model="gemini-3.8-flash",
    )
    strat = GeminiStrategy(settings)

    # 1. HTTP 400 on thinkingConfig falls back cleanly without thinkingConfig
    resp_400 = MagicMock(status_code=400)
    resp_200 = MagicMock(status_code=200)
    resp_200.raise_for_status = MagicMock()
    resp_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "[]"}]}}],
        "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 5},
    }
    post_mock = AsyncMock(side_effect=[resp_400, resp_200])
    client_mock = AsyncMock()
    client_mock.__aenter__ = AsyncMock(return_value=MagicMock(post=post_mock))
    client_mock.__aexit__ = AsyncMock(return_value=False)

    with patch("kalshi_poc.strategy_ai.httpx.AsyncClient", return_value=client_mock):
        text, p_tok, c_tok = await strat._call_llm_batch("test prompt")

    assert text == "[]"
    assert (p_tok, c_tok) == (50, 5)
    assert post_mock.call_count == 2
    first_gen_cfg = post_mock.call_args_list[0].kwargs["json"]["generationConfig"]
    second_gen_cfg = post_mock.call_args_list[1].kwargs["json"]["generationConfig"]
    assert first_gen_cfg["thinkingConfig"] == {"thinkingLevel": "low"}
    assert "thinkingConfig" not in second_gen_cfg
    # API key must be in x-goog-api-key header, never in query params
    assert post_mock.call_args_list[0].kwargs["headers"]["x-goog-api-key"] == "test-key"
    assert "params" not in post_mock.call_args_list[0].kwargs

    # 2. HTTP 402 backs off _last_llm_call_monotonic by +60s
    import time as _time
    import httpx as _httpx

    req_obj = _httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent")
    err_402 = _httpx.HTTPStatusError(
        "402 Payment Required",
        request=req_obj,
        response=_httpx.Response(402, request=req_obj),
    )
    snaps = [_book(ticker="KXBTC-1", yes_bid="0.49", yes_count="50", no_bid="0.49", no_count="10")]
    portfolio = PortfolioState(
        cash_balance_dollars=Decimal("100.00"),
        starting_capital_dollars=Decimal("100.00"),
        daily_start_equity_dollars=Decimal("100.00"),
    )
    before_mono = _time.monotonic()
    with patch.object(strat, "_call_llm_batch", new=AsyncMock(side_effect=err_402)):
        res_402 = await strat.evaluate_batch(snaps, portfolio, force=True)
    assert "HTTP 402" in res_402.decisions[0]["rationale"]
    assert strat._last_llm_call_monotonic >= before_mono + 55.0

    # 3. TimeoutError produces explicit repr(exc) in rationale & releases lock
    with patch.object(strat, "_call_llm_batch", new=AsyncMock(side_effect=asyncio.TimeoutError())):
        res = await strat.evaluate_batch(snaps, portfolio, force=True)

    assert "TimeoutError" in res.decisions[0]["rationale"]
    assert not strat._inflight_lock.locked()
    assert strat._last_llm_call_monotonic > 0


async def test_level1_imbalance_ignores_deep_outliers_and_clamps_crossing_price() -> None:
    import json
    from unittest.mock import AsyncMock, patch
    from kalshi_poc.models import StrategyProvider
    from kalshi_poc.strategy_ai import GeminiStrategy

    settings = AppSettings(
        _env_file=None,
        strategy_provider=StrategyProvider.GEMINI,
        llm_api_key="test-key",
        max_slippage_cents=Decimal("3"),
    )
    strat = GeminiStrategy(settings)
    # Level 1: yes_bid=0.49 (10), no_bid=0.49 (10 -> yes_ask=0.51).
    # Deep level 2 ticks away has 100,000 contracts -> must NOT skew Level 1 imbalance!
    snap = OrderbookSnapshot(
        ticker="KXINXU-1",
        timestamp_ms=1700000000000,
        yes_bids=[
            PriceLevel(price=Decimal("0.40"), count=Decimal("100000")),
            PriceLevel(price=Decimal("0.49"), count=Decimal("10")),
        ],
        no_bids=[
            PriceLevel(price=Decimal("0.40"), count=Decimal("50")),
            PriceLevel(price=Decimal("0.49"), count=Decimal("10")),
        ],
    )
    assert strat._compute_imbalance(snap) == Decimal("0")

    portfolio = PortfolioState(
        cash_balance_dollars=Decimal("100.00"),
        starting_capital_dollars=Decimal("100.00"),
        daily_start_equity_dollars=Decimal("100.00"),
    )
    prompt = strat._build_batch_prompt([snap], portfolio, {})
    assert "100000" not in prompt
    assert '"500"' in prompt

    # BUY at 0.50 when best_yes_ask is 0.51 -> clamped up to 0.51 to cross spread
    llm_out = json.dumps(
        [
            {
                "ticker": "KXINXU-1",
                "side": "bid",
                "count": "2",
                "limit_price": "0.5000",
                "expected_edge_per_contract": "0.05",
                "hold_to_expiry": True,
                "confidence": "0.80",
                "rationale": "Cross ask",
            }
        ]
    )
    with patch.object(strat, "_call_llm_batch", new=AsyncMock(return_value=(llm_out, 100, 20))):
        res = await strat.evaluate_batch([snap], portfolio, force=True)
    assert len(res.intents) == 1
    assert res.intents[0].limit_price == Decimal("0.5100")
