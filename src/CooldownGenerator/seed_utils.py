# seed_utils.py
import os, time, hashlib, hmac, secrets, socket, struct

def _splitmix64(x: int) -> tuple[int, int]:
    """SplitMix64 step: returns (y, next_x)."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= (z >> 31)
    return z, x

def _u64_from_bytes(b: bytes) -> int:
    return struct.unpack(">Q", hashlib.sha256(b).digest()[:8])[0]

def derive_seed(session_tag: str | None = None,
                *,
                deterministic_key: bytes | None = None) -> int:
    """
    Returns a 64-bit integer seed.
    - Deterministic path: if deterministic_key is provided, seed is derived
      reproducibly from (key, session_tag).
    - Nondeterministic path: mixes multiple entropy sources for high-entropy seed.
    """
    if deterministic_key is not None:
        salt = (session_tag or "default").encode()
        prk = hmac.new(salt, deterministic_key, hashlib.sha256).digest()
        okm = hmac.new(prk, b"seed\x01", hashlib.sha256).digest()
        return _u64_from_bytes(okm)

    raw = b"".join([
        os.urandom(16),
        struct.pack(">Q", time.time_ns()),
        struct.pack(">I", os.getpid()),
        socket.gethostname().encode(errors="ignore"),
        secrets.token_bytes(16),
    ])
    return _u64_from_bytes(raw)

def bump_seed(seed64: int, n: int = 1) -> int:
    """Generate a related seed using n SplitMix64 bumps."""
    x = seed64
    y = seed64
    for _ in range(n):
        y, x = _splitmix64(x)
    return y
