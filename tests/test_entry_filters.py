"""Anti-chase / min-entry-price filters, minutes_to_close feature and the mid-move trigger switch."""

import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from kalshi_poc.config import AppSettings
from kalshi_poc.models import (
    OrderSide,
    OrderbookSnapshot,
    PortfolioState,
    PriceLevel,
    StrategyProvider,
)
from kalshi_poc.strategy_ai import DEFAULT_BATCH_SYSTEM_PROMPT, GeminiStrategy

TICKER = "KXETH15M-26NOV191545-45"


def _settings(**kw) -> AppSettings:
    return AppSettings(
        _env_file=None,
        strategy_provider=StrategyProvider.GEMINI,
        llm_api_key="test-key",
        llm_min_cooldown_sec=15,
        llm_heartbeat_sec=60,
        **kw,
    )


def _snap(yes_bid: str, no_bid: str, ts_ms: int, ticker: str = TICKER) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        ticker=ticker,
        timestamp_ms=ts_ms,
        yes_bids=[PriceLevel(price=Decimal(yes_bid), count=Decimal("100"))],
        no_bids=[PriceLevel(price=Decimal(no_bid), count=Decimal("100"))],
    )


def _portfolio() -> PortfolioState:
    return PortfolioState(
        cash_balance_dollars=Decimal("500.00"),
        starting_capital_dollars=Decimal("500.00"),
        daily_start_equity_dollars=Decimal("500.00"),
    )


def _proposal(side: str, limit: str) -> str:
    return json.dumps(
        [
            {
                "ticker": TICKER,
                "side": side,
                "count": "10",
                "limit_price": limit,
                "expected_edge_per_contract": "0.05",
                "confidence": "0.7",
            }
        ]
    )


def _strategy_after_rally(**kw) -> tuple[GeminiStrategy, OrderbookSnapshot]:
    """YES mid rallies 0.55 -> 0.65 over one minute; the prompt is built on the last book."""
    strat = GeminiStrategy(_settings(**kw))
    t0 = 1700000000000
    strat._record_mid_prices([_snap("0.54", "0.44", t0)])  # mid 0.55
    last = _snap("0.64", "0.34", t0 + 61_000)  # mid 0.65, ask 0.66
    strat._build_batch_prompt([last], _portfolio(), {})
    return strat, last


def test_anti_chase_rejects_entry_in_direction_of_1m_spike() -> None:
    strat, snap = _strategy_after_rally(anti_chase_max_move_1m=Decimal("0.06"))
    intents, rejections = strat._parse_batch_intents_detailed(
        _proposal("bid", "0.66"), [snap], 10, positions_by_ticker={}
    )
    assert intents == []
    assert rejections[0]["status"] == "FILTERED"
    assert "Anti-chase" in rejections[0]["reason"]


def test_anti_chase_allows_fading_the_spike() -> None:
    # Selling YES (long NO at 0.36) fades the rally; move in trade direction is -0.10
    strat, snap = _strategy_after_rally(anti_chase_max_move_1m=Decimal("0.06"))
    intents, _ = strat._parse_batch_intents_detailed(
        _proposal("ask", "0.64"), [snap], 10, positions_by_ticker={}
    )
    assert len(intents) == 1 and intents[0].side == OrderSide.ASK


def test_anti_chase_off_by_default() -> None:
    strat, snap = _strategy_after_rally()
    intents, _ = strat._parse_batch_intents_detailed(
        _proposal("bid", "0.66"), [snap], 10, positions_by_ticker={}
    )
    assert len(intents) == 1


def test_min_entry_price_own_side_filters_cheap_entries_on_both_sides() -> None:
    strat = GeminiStrategy(_settings(min_entry_price_own_side=Decimal("0.60")))
    snap = _snap("0.54", "0.44", 1700000000000)  # YES 0.54/0.56, NO 0.44/0.46
    # Buy YES at 0.56 -> own side 0.56 < 0.60
    intents, rej = strat._parse_batch_intents_detailed(
        _proposal("bid", "0.56"), [snap], 10, positions_by_ticker={}
    )
    assert intents == [] and "below min_entry_price_own_side" in rej[0]["reason"]
    # Sell YES at 0.54 -> own side (NO) 0.46 < 0.60
    intents, rej = strat._parse_batch_intents_detailed(
        _proposal("ask", "0.54"), [snap], 10, positions_by_ticker={}
    )
    assert intents == [] and "below min_entry_price_own_side" in rej[0]["reason"]

    fav = _snap("0.64", "0.34", 1700000000000)  # YES ask 0.66
    intents, _ = strat._parse_batch_intents_detailed(
        _proposal("bid", "0.66"), [fav], 10, positions_by_ticker={}
    )
    assert len(intents) == 1


def test_prompt_has_minutes_to_close_and_active_limits() -> None:
    strat = GeminiStrategy(
        _settings(min_entry_price_own_side=Decimal("0.60"), anti_chase_max_move_1m=Decimal("0.06"))
    )
    strat._market_expiries = {TICKER: datetime.now(timezone.utc) + timedelta(minutes=12)}
    prompt = strat._build_batch_prompt([_snap("0.54", "0.44", 1700000000000)], _portfolio(), {})
    payload = json.loads(prompt[prompt.index("{"):])
    assert 11.5 <= float(payload["markets"][0]["minutes_to_close"]) <= 12.0
    assert payload["min_entry_price_own_side"] == "0.60"
    assert payload["max_mid_delta_1m_in_trade_direction"] == "0.06"


def test_prompt_minutes_to_close_null_when_expiry_unknown_and_limits_omitted_when_off() -> None:
    strat = GeminiStrategy(_settings())
    prompt = strat._build_batch_prompt([_snap("0.54", "0.44", 1700000000000)], _portfolio(), {})
    payload = json.loads(prompt[prompt.index("{"):])
    assert payload["markets"][0]["minutes_to_close"] is None
    assert "min_entry_price_own_side" not in payload
    assert "max_mid_delta_1m_in_trade_direction" not in payload


def test_system_prompt_no_longer_steers_toward_momentum() -> None:
    assert "momentum" not in DEFAULT_BATCH_SYSTEM_PROMPT.lower()
    assert "mean-revert" in DEFAULT_BATCH_SYSTEM_PROMPT
    assert "minutes_to_close" in DEFAULT_BATCH_SYSTEM_PROMPT


def _trigger_after_mid_move(**kw) -> tuple[bool, str]:
    strat = GeminiStrategy(_settings(**kw))
    # Balanced, wide book so the spread/imbalance trigger does not fire
    first = _snap("0.40", "0.50", 1700000000000)
    strat._last_llm_mids[TICKER] = first.mid_yes_price
    strat._last_llm_call_monotonic = time.monotonic() - 20.0  # past cooldown, before heartbeat
    moved = _snap("0.45", "0.45", 1700000020000)  # mid +0.05
    return strat._should_trigger_llm([moved], {}, force=False)


def test_mid_move_trigger_disabled_by_default() -> None:
    should_call, reason = _trigger_after_mid_move()
    assert should_call is False and "Event gate" in reason


def test_mid_move_trigger_can_be_re_enabled() -> None:
    should_call, reason = _trigger_after_mid_move(llm_mid_move_trigger_enabled=True)
    assert should_call is True and reason.startswith("mid_move")


def test_new_settings_persist_to_env(tmp_path) -> None:
    from kalshi_poc.config import ConfigManager

    mgr = ConfigManager(env_path=tmp_path / ".env", config_json_path=tmp_path / "config.json")
    mgr.update_and_persist(
        {"min_entry_price_own_side": "0.60", "anti_chase_max_move_1m": "0.06"}
    )
    env = (tmp_path / ".env").read_text()
    assert "MIN_ENTRY_PRICE_OWN_SIDE=0.60" in env
    assert "ANTI_CHASE_MAX_MOVE_1M=0.06" in env
    assert "LLM_MID_MOVE_TRIGGER_ENABLED=false" in env
