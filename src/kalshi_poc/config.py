"""Pydantic BaseSettings and thread-safe atomic config persistence with hot-reload."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kalshi_poc.auth import load_private_key
from kalshi_poc.models import Environment, ExecutionMode, StrategyProvider


class AppSettings(BaseSettings):
    """Complete system configuration editable via .env, config.json, or Web UI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Kalshi environment & credentials
    kalshi_environment: Environment = Field(default=Environment.DEMO)
    kalshi_api_key_id: str = Field(default="")
    kalshi_private_key_path: str = Field(default="~/.kalshi/demo_private_key.pem")

    # Execution & Capital
    execution_mode: ExecutionMode = Field(default=ExecutionMode.PAUSED)
    initial_capital: Decimal = Field(default=Decimal("5000.00"), gt=Decimal("0"))
    equity_floor: Decimal = Field(default=Decimal("2000.00"), ge=Decimal("0"))
    daily_drawdown_kill_switch: Decimal = Field(default=Decimal("500.00"), gt=Decimal("0"))
    poll_interval_ms: int = Field(default=1000, ge=100, le=60000)

    # Risk & Microstructure Limits
    max_position_dollars_per_market: Decimal = Field(default=Decimal("250.00"), gt=Decimal("0"))
    dynamic_sizing_enabled: bool = Field(default=True)
    house_money_risk_pct: Decimal = Field(
        default=Decimal("0.20"), gt=Decimal("0"), le=Decimal("1")
    )
    max_open_markets: int = Field(default=5, ge=1, le=100)
    max_orders_per_minute: int = Field(default=10, ge=1, le=600)
    min_net_edge_dollars: Decimal = Field(default=Decimal("0.02"), ge=Decimal("0"))
    max_slippage_cents: Decimal = Field(default=Decimal("3"), ge=Decimal("0"))
    # Risk-reducing exits may cross wider books than entries (a wide spread must not trap a position)
    exit_max_slippage_cents: Decimal = Field(default=Decimal("6"), ge=Decimal("0"))

    # Deterministic exits (engine-side; Kalshi event contracts have no native stop orders).
    # Prices are on the position's own side: YES bid for YES positions, NO bid for NO positions.
    take_profit_min_gain: Decimal = Field(default=Decimal("0.18"), ge=Decimal("0"), le=Decimal("1"))
    take_profit_min_price: Decimal = Field(default=Decimal("0.78"), gt=Decimal("0"), lt=Decimal("1"))
    take_profit_lock_price: Decimal = Field(default=Decimal("0.85"), gt=Decimal("0"), lt=Decimal("1"))
    stop_loss_drop: Decimal = Field(default=Decimal("0.22"), gt=Decimal("0"), le=Decimal("1"))
    stop_loss_disable_secs: int = Field(default=180, ge=0, le=3600)
    trailing_stop_enabled: bool = Field(default=True)
    trailing_stop_activation_gain: Decimal = Field(
        default=Decimal("0.08"), ge=Decimal("0"), le=Decimal("1")
    )
    trailing_stop_distance: Decimal = Field(default=Decimal("0.06"), gt=Decimal("0"), le=Decimal("1"))
    # After any approved exit, block new entries in that same market for this many seconds (0 = off).
    reentry_cooldown_secs: int = Field(default=60, ge=0, le=3600)

    # Tax-Hygiene Allowlist & Discovery Priority
    allowed_series_prefixes: list[str] = Field(
        default_factory=lambda: ["KXBTC", "KXETH", "KXINX", "KXFED", "KXCPI"]
    )
    enforce_financial_only: bool = Field(default=True)
    prefer_15m_turnover: bool = Field(default=True)
    subscribed_tickers: list[str] = Field(default_factory=list)

    # AI Strategy Settings
    strategy_provider: StrategyProvider = Field(default=StrategyProvider.STUB_MOMENTUM)
    llm_api_key: str = Field(default="")
    llm_endpoint: str = Field(default="")
    llm_model: str = Field(default="claude-3-5-sonnet-latest")
    llm_temperature: Decimal = Field(default=Decimal("0.1"), ge=Decimal("0"), le=Decimal("2"))
    system_prompt_override: str = Field(default="")
    llm_min_cooldown_sec: int = Field(default=15, ge=0, le=3600)
    llm_heartbeat_sec: int = Field(default=60, ge=1, le=86400)

    # Telemetry & Web Server
    db_path: str = Field(default="data/kalshi_telemetry.db")
    web_host: str = Field(default="127.0.0.1")
    web_port: int = Field(default=8000, ge=1, le=65535)
    # Extra Host/Origin hostnames accepted by the control panel (127.0.0.1/localhost/::1 always are)
    web_allowed_hosts: list[str] = Field(default_factory=list)

    @field_validator("web_allowed_hosts", mode="before")
    @classmethod
    def parse_host_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            v = json.loads(v) if v.startswith("[") else v.split(",")
        if isinstance(v, list):
            return [str(x).strip().lower() for x in v if str(x).strip()]
        return []

    @field_validator("allowed_series_prefixes", "subscribed_tickers", mode="before")
    @classmethod
    def parse_str_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("["):
                return [str(x).strip().upper() for x in json.loads(v) if str(x).strip()]
            return [x.strip().upper() for x in v.split(",") if x.strip()]
        if isinstance(v, list):
            return [str(x).strip().upper() for x in v if str(x).strip()]
        return []

    @property
    def rest_base_url(self) -> str:
        if self.kalshi_environment == Environment.PRODUCTION:
            return "https://external-api.kalshi.com/trade-api/v2"
        return "https://external-api.demo.kalshi.co/trade-api/v2"

    @property
    def ws_url(self) -> str:
        if self.kalshi_environment == Environment.PRODUCTION:
            return "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
        return "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"

    @property
    def resolved_private_key_path(self) -> Path:
        return Path(self.kalshi_private_key_path).expanduser()


def _q(value: str) -> str:
    """Double-quote a .env value (dotenv decodes the escapes) so newlines/#/quotes cannot inject."""
    return json.dumps(value, ensure_ascii=False)


def _atomic_write(path: Path, content: str, mode: int | None = None) -> None:
    """Write content atomically using a temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


class ConfigManager:
    """Thread-safe configuration manager with atomic persistence and hot-reload hooks."""

    def __init__(
        self,
        env_path: Path | str = ".env",
        config_json_path: Path | str = "config.json",
        key_dir: Path | str = "~/.kalshi",
        allow_live: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        # C2: live execution is only reachable when the process was started with --confirm-live
        self.allow_live = allow_live
        self.env_path = Path(env_path)
        self.config_json_path = Path(config_json_path)
        self.key_dir = Path(key_dir).expanduser()
        self._callbacks: list[Callable[[AppSettings, AppSettings], None]] = []
        self._settings = self._load_initial()

    def _load_initial(self) -> AppSettings:
        base = AppSettings(_env_file=self.env_path)
        if self.config_json_path.exists():
            try:
                data = json.loads(self.config_json_path.read_text(encoding="utf-8"))
                merged = base.model_dump()
                merged.update(data)
                return AppSettings.model_validate(merged)
            except Exception:
                return base
        return base

    @property
    def settings(self) -> AppSettings:
        with self._lock:
            return self._settings

    def register_reload_callback(
        self, cb: Callable[[AppSettings, AppSettings], None]
    ) -> None:
        with self._lock:
            self._callbacks.append(cb)

    def save_private_key_pem(self, pem_text: str, env: Environment) -> tuple[Path, str]:
        """Validate PEM, save securely to ~/.kalshi/ with 0600 permissions, return (path, algo)."""
        _, algo = load_private_key(pem_text)
        self.key_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.key_dir, 0o700)
        key_path = self.key_dir / f"{env.value}_private_key.pem"
        normalized_pem = pem_text.strip() + "\n"
        _atomic_write(key_path, normalized_pem, mode=0o600)
        return key_path, algo

    def get_private_key_status(self, settings: AppSettings | None = None) -> dict[str, Any]:
        s = settings or self.settings
        key_path = s.resolved_private_key_path
        if not key_path.exists():
            return {"configured": False, "masked": "", "algorithm": None}
        try:
            _, algo = load_private_key(key_path.read_bytes())
            return {"configured": True, "masked": "••••configured", "algorithm": algo}
        except Exception:
            return {"configured": False, "masked": "invalid_key", "algorithm": None}

    def get_masked_config(self) -> dict[str, Any]:
        """Return full config dict with secrets masked (never returns raw private key or LLM key)."""
        with self._lock:
            s = self._settings
            data = s.model_dump(mode="json")
            key_status = self.get_private_key_status(s)
            data["private_key_configured"] = key_status["configured"]
            data["private_key_masked"] = key_status["masked"]
            data["private_key_algorithm"] = key_status["algorithm"]
            data["llm_api_key"] = "••••configured" if s.llm_api_key else ""
            data["llm_api_key_configured"] = bool(s.llm_api_key)
            data["rest_base_url"] = s.rest_base_url
            data["ws_url"] = s.ws_url
            return data

    def update_and_persist(self, updates: dict[str, Any]) -> AppSettings:
        """Validate updates via Pydantic, persist atomically, and trigger hot-reload."""
        with self._lock:
            old_settings = self._settings
            current = old_settings.model_dump()

            cleaned = dict(updates)
            # Never overwrite an existing LLM key with the masked placeholder
            if cleaned.get("llm_api_key") in ("••••configured", None):
                cleaned.pop("llm_api_key", None)

            # Optional inline PEM upload/paste inside config update
            raw_pem = cleaned.pop("private_key_pem", None)
            if raw_pem and isinstance(raw_pem, str) and raw_pem.strip() and "••••" not in raw_pem:
                target_env = Environment(
                    cleaned.get("kalshi_environment", old_settings.kalshi_environment.value)
                )
                saved_path, _ = self.save_private_key_pem(raw_pem, target_env)
                cleaned["kalshi_private_key_path"] = str(saved_path)

            current.update(cleaned)
            new_settings = AppSettings.model_validate(current)
            if new_settings.execution_mode == ExecutionMode.LIVE and not self.allow_live:
                raise ValueError(
                    "execution_mode=live is disabled: restart with `kalshi-poc run --confirm-live` "
                    "to allow real-money trading."
                )
            if (
                new_settings.execution_mode == ExecutionMode.LIVE
                and new_settings.strategy_provider == StrategyProvider.STUB_MOMENTUM
            ):
                raise ValueError(
                    "execution_mode=live is blocked with strategy_provider=stub_momentum: "
                    "the stub is a test strategy; select a model provider before going live."
                )

            self._persist_files(new_settings)
            self._settings = new_settings
            callbacks = list(self._callbacks)

        for cb in callbacks:
            cb(old_settings, new_settings)
        return new_settings

    def _persist_files(self, s: AppSettings) -> None:
        json_payload = json.dumps(s.model_dump(mode="json"), indent=2)
        _atomic_write(self.config_json_path, json_payload + "\n")

        env_lines = [
            f"KALSHI_ENVIRONMENT={s.kalshi_environment.value}",
            f"KALSHI_API_KEY_ID={_q(s.kalshi_api_key_id)}",
            f"KALSHI_PRIVATE_KEY_PATH={_q(s.kalshi_private_key_path)}",
            f"EXECUTION_MODE={s.execution_mode.value}",
            f"INITIAL_CAPITAL={s.initial_capital}",
            f"EQUITY_FLOOR={s.equity_floor}",
            f"DAILY_DRAWDOWN_KILL_SWITCH={s.daily_drawdown_kill_switch}",
            f"POLL_INTERVAL_MS={s.poll_interval_ms}",
            f"MAX_POSITION_DOLLARS_PER_MARKET={s.max_position_dollars_per_market}",
            f"DYNAMIC_SIZING_ENABLED={str(s.dynamic_sizing_enabled).lower()}",
            f"HOUSE_MONEY_RISK_PCT={s.house_money_risk_pct}",
            f"MAX_OPEN_MARKETS={s.max_open_markets}",
            f"MAX_ORDERS_PER_MINUTE={s.max_orders_per_minute}",
            f"MIN_NET_EDGE_DOLLARS={s.min_net_edge_dollars}",
            f"MAX_SLIPPAGE_CENTS={s.max_slippage_cents}",
            f"EXIT_MAX_SLIPPAGE_CENTS={s.exit_max_slippage_cents}",
            f"TAKE_PROFIT_MIN_GAIN={s.take_profit_min_gain}",
            f"TAKE_PROFIT_MIN_PRICE={s.take_profit_min_price}",
            f"TAKE_PROFIT_LOCK_PRICE={s.take_profit_lock_price}",
            f"STOP_LOSS_DROP={s.stop_loss_drop}",
            f"STOP_LOSS_DISABLE_SECS={s.stop_loss_disable_secs}",
            f"TRAILING_STOP_ENABLED={str(s.trailing_stop_enabled).lower()}",
            f"TRAILING_STOP_ACTIVATION_GAIN={s.trailing_stop_activation_gain}",
            f"TRAILING_STOP_DISTANCE={s.trailing_stop_distance}",
            f"REENTRY_COOLDOWN_SECS={s.reentry_cooldown_secs}",
            f"ALLOWED_SERIES_PREFIXES={json.dumps(s.allowed_series_prefixes)}",
            f"ENFORCE_FINANCIAL_ONLY={str(s.enforce_financial_only).lower()}",
            f"PREFER_15M_TURNOVER={str(s.prefer_15m_turnover).lower()}",
            f"SUBSCRIBED_TICKERS={json.dumps(s.subscribed_tickers)}",
            f"STRATEGY_PROVIDER={s.strategy_provider.value}",
            f"LLM_API_KEY={_q(s.llm_api_key)}",
            f"LLM_ENDPOINT={_q(s.llm_endpoint)}",
            f"LLM_MODEL={_q(s.llm_model)}",
            f"LLM_TEMPERATURE={s.llm_temperature}",
            f"SYSTEM_PROMPT_OVERRIDE={_q(s.system_prompt_override)}",
            f"LLM_MIN_COOLDOWN_SEC={s.llm_min_cooldown_sec}",
            f"LLM_HEARTBEAT_SEC={s.llm_heartbeat_sec}",
            f"DB_PATH={_q(s.db_path)}",
            f"WEB_HOST={_q(s.web_host)}",
            f"WEB_PORT={s.web_port}",
            f"WEB_ALLOWED_HOSTS={json.dumps(s.web_allowed_hosts)}",
        ]
        _atomic_write(self.env_path, "\n".join(env_lines) + "\n", mode=0o600)
