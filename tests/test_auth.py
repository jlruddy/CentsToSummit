"""Verify Ed25519 and RSA-PSS signature formatting against Kalshi pre-sign rules."""

import base64

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from kalshi_poc.auth import KalshiSigner, load_private_key


def _generate_ed25519_pem() -> tuple[ed25519.Ed25519PrivateKey, str]:
    priv = ed25519.Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    return priv, pem


def _generate_rsa_pem() -> tuple[rsa.RSAPrivateKey, str]:
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    return priv, pem


def test_ed25519_signing_and_verification() -> None:
    priv, pem = _generate_ed25519_pem()
    _, algo = load_private_key(pem)
    assert algo == "Ed25519"

    signer = KalshiSigner.from_pem("test-key-id", pem)
    ts = 1700000000000
    headers = signer.headers("get", "/trade-api/v2/portfolio/balance?foo=bar", timestamp_ms=ts)

    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == str(ts)

    expected_msg = f"{ts}GET/trade-api/v2/portfolio/balance".encode("utf-8")
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    priv.public_key().verify(sig, expected_msg)


def test_rsa_pss_signing_and_verification() -> None:
    priv, pem = _generate_rsa_pem()
    _, algo = load_private_key(pem)
    assert algo == "RSA-PSS"

    signer = KalshiSigner.from_pem("rsa-key-id", pem)
    ts = 1700000000123
    headers = signer.headers("POST", "/trade-api/v2/portfolio/events/orders", timestamp_ms=ts)

    expected_msg = f"{ts}POST/trade-api/v2/portfolio/events/orders".encode("utf-8")
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    priv.public_key().verify(
        sig,
        expected_msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_websocket_path_presign_and_invalid_path() -> None:
    _, pem = _generate_ed25519_pem()
    signer = KalshiSigner.from_pem("ws-key", pem)
    presign = signer.build_presign_string("GET", "/trade-api/ws/v2", 1700000000000)
    assert presign == "1700000000000GET/trade-api/ws/v2"

    with pytest.raises(ValueError, match="must begin with"):
        signer.normalize_path("/v1/invalid")
