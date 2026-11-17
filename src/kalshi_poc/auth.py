"""Dual Ed25519 and RSA-PSS request signer for Kalshi API v2."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Literal

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

KeyAlgorithm = Literal["Ed25519", "RSA-PSS"]
PrivateKeyType = ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey


def load_private_key(pem_data: str | bytes) -> tuple[PrivateKeyType, KeyAlgorithm]:
    """Load a PEM-encoded private key and auto-detect Ed25519 vs RSA-PSS."""
    raw = pem_data.encode("utf-8") if isinstance(pem_data, str) else pem_data
    raw = raw.strip()
    if not raw:
        raise ValueError("Private key PEM is empty.")

    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except Exception as exc:
        raise ValueError(f"Invalid PEM private key format: {exc}") from exc

    if isinstance(key, ed25519.Ed25519PrivateKey):
        return key, "Ed25519"
    if isinstance(key, rsa.RSAPrivateKey):
        return key, "RSA-PSS"
    raise ValueError(
        f"Unsupported private key type: {type(key).__name__}. Expected Ed25519 or RSA."
    )


def load_private_key_from_path(path: str | Path) -> tuple[PrivateKeyType, KeyAlgorithm]:
    """Load and auto-detect a PEM private key from disk."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Private key file not found: {resolved}")
    return load_private_key(resolved.read_bytes())


class KalshiSigner:
    """Signs Kalshi REST and WebSocket v2 requests using Ed25519 or RSA-PSS."""

    def __init__(self, api_key_id: str, private_key: PrivateKeyType, algorithm: KeyAlgorithm) -> None:
        if not api_key_id.strip():
            raise ValueError("Kalshi API Key ID cannot be empty.")
        self.api_key_id = api_key_id.strip()
        self.private_key = private_key
        self.algorithm = algorithm

    @classmethod
    def from_pem(cls, api_key_id: str, pem_data: str | bytes) -> KalshiSigner:
        key, algo = load_private_key(pem_data)
        return cls(api_key_id=api_key_id, private_key=key, algorithm=algo)

    @classmethod
    def from_path(cls, api_key_id: str, path: str | Path) -> KalshiSigner:
        key, algo = load_private_key_from_path(path)
        return cls(api_key_id=api_key_id, private_key=key, algorithm=algo)

    @staticmethod
    def normalize_path(path: str) -> str:
        """Strip query string and ensure path starts with /trade-api/."""
        path_without_query = path.split("?", 1)[0].strip()
        if not path_without_query.startswith("/"):
            path_without_query = "/" + path_without_query
        if not (
            path_without_query.startswith("/trade-api/v2")
            or path_without_query.startswith("/trade-api/ws/v2")
        ):
            raise ValueError(
                f"Kalshi signature path must begin with /trade-api/v2 or /trade-api/ws/v2, got: {path_without_query}"
            )
        return path_without_query

    def build_presign_string(self, method: str, path: str, timestamp_ms: int) -> str:
        """Construct `{timestamp_ms}{METHOD_UPPER}{path_without_query}`."""
        normalized_path = self.normalize_path(path)
        return f"{timestamp_ms}{method.upper()}{normalized_path}"

    def sign_message(self, message: bytes) -> str:
        """Sign raw UTF-8 pre-sign bytes and return base64 signature."""
        if isinstance(self.private_key, ed25519.Ed25519PrivateKey):
            sig = self.private_key.sign(message)
        elif isinstance(self.private_key, rsa.RSAPrivateKey):
            sig = self.private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
        else:
            raise ValueError(f"Unsupported key type: {type(self.private_key).__name__}")
        return base64.b64encode(sig).decode("ascii")

    def headers(
        self, method: str, path: str, timestamp_ms: int | None = None
    ) -> dict[str, str]:
        """Generate KALSHI-ACCESS-* authentication headers."""
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        presign = self.build_presign_string(method=method, path=path, timestamp_ms=ts)
        signature = self.sign_message(presign.encode("utf-8"))
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": signature,
        }
