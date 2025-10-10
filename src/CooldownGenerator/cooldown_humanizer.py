# src/AutoFisher/CooldownGenerator/cooldown_humanizer.py
from __future__ import annotations

import random
from math import exp

from .seed_utils import derive_seed, bump_seed


class HumanCooldown:
    """
    Stateful human-like cooldown generator around a target base interval.

    Models:
      - Short-term jitter (Gaussian)
      - Slow drift (Ornstein–Uhlenbeck mean reversion)
      - Micro-pauses (brief attention hiccups)
      - Slips / outliers (lognormal tail)
      - Real breaks (short & long)
      - Warm-up after breaks (decaying bias + higher variance)
      - Periodic rekey + rekey after long breaks to decorrelate long streams

    The 'routine' component is **truncated** (rejection-sampled) to [min_clip, max_clip]
    instead of hard-clamped, eliminating histogram pile-ups at the boundaries.

    Usage:
      hc = HumanCooldown(base=3.0)
      while True:
          delay = hc.next()
          # sleep(delay); then perform your action
    """

    def __init__(
        self,
        base: float = 3.0,                   # intended cadence (seconds)
        seed: int | None = None,             # optional; None -> per-session strong seed
        # Routine bounds for truncated sampling (micro/slips may exceed)
        min_clip: float = 2.4,
        max_clip: float = 4.6,
        # Short-term jitter (fast noise)
        sigma_short: float = 0.18,
        # Slow drift (OU process per click)
        ou_theta: float = 0.08,              # mean reversion strength
        ou_sigma: float = 0.08,              # diffusion per step
        # Micro pauses (small bumps)
        p_micro: float = 0.06,
        micro_min: float = 0.15,
        micro_max: float = 0.55,
        # Slips / outliers (fat tail)
        p_slip: float = 0.015,
        slip_mu_ln: float = -1.0,            # lognormal median ~ exp(mu)
        slip_sigma_ln: float = 0.6,
        # Real breaks (short & long)
        p_break: float = 0.004,
        short_break_s: tuple[float, float] = (7.0, 25.0),
        p_long_break: float = 0.10,
        long_break_s: tuple[float, float] = (60.0, 210.0),
        # Warm-up after any break
        warmup_clicks: int = 12,
        warmup_bias: float = 0.22,
        warmup_sigma_boost: float = 0.55,
        # Rekeying (decorrelate long-range PRNG streams)
        rekey_every: int = 300,              # rekey every N intervals
        session_tag: str | None = "hc_session",
        deterministic_key: bytes | None = None,  # set for reproducible sessions
        # Truncation controls
        max_truncation_resamples: int = 12,  # safety cap for rejection sampling
    ):
        self.base = base
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.sigma_short = sigma_short
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.p_micro = p_micro
        self.micro_min = micro_min
        self.micro_max = micro_max
        self.p_slip = p_slip
        self.slip_mu_ln = slip_mu_ln
        self.slip_sigma_ln = slip_sigma_ln
        self.p_break = p_break
        self.short_break_s = short_break_s
        self.p_long_break = p_long_break
        self.long_break_s = long_break_s
        self.warmup_clicks = warmup_clicks
        self.warmup_bias = warmup_bias
        self.warmup_sigma_boost = warmup_sigma_boost
        self.max_truncation_resamples = max_truncation_resamples

        # PRNG seeding / rekey
        self._seed0 = seed if seed is not None else derive_seed(
            session_tag, deterministic_key=deterministic_key
        )
        self._rng = random.Random(self._seed0)
        self._rekey_every = max(1, rekey_every)
        self._ticks = 0

        # State
        self._ou = 0.0
        self._warmup_left = 0

    # --- internals ---

    def _rekey(self, n: int):
        """Re-seed PRNG using a derived SplitMix64 bump to decorrelate streams."""
        new_seed = bump_seed(self._seed0, n=n)
        self._rng.seed(new_seed)

    def _ou_step(self) -> float:
        # Discrete OU: x_{t+1} = x_t + theta*(0 - x_t) + sigma*N(0,1)
        x = self._ou
        x += self.ou_theta * (-x) + self.ou_sigma * self._rng.gauss(0.0, 1.0)
        self._ou = x
        return x

    def _maybe_break(self) -> float:
        if self._rng.random() < self.p_break:
            # choose short or long break
            if self._rng.random() < self.p_long_break:
                a, b = self.long_break_s
            else:
                a, b = self.short_break_s
            br = self._rng.uniform(a, b)
            # warm-up on resume
            self._warmup_left = self.warmup_clicks
            # rekey on sufficiently long breaks (>= 60s default)
            if br >= 60.0:
                n = (self._ticks // self._rekey_every) + 1
                self._rekey(n)
            return br
        return 0.0

    # --- public API ---

    def next(self) -> float:
        """Return the next cooldown (seconds)."""
        self._ticks += 1
        if (self._ticks % self._rekey_every) == 0:
            n = self._ticks // self._rekey_every
            self._rekey(n)

        # 1) Real break?
        br = self._maybe_break()
        if br > 0.0:
            return br

        # 2) Slow drift
        drift = self._ou_step()

        # 3) Warm-up shaping (decaying bias + variance boost)
        sigma = self.sigma_short
        warm_bias = 0.0
        if self._warmup_left > 0:
            k = max(1, self.warmup_clicks)
            i = self._warmup_left
            decay = exp(-2.0 * (1.0 - i / k))  # ~0.135 .. 1 ramp
            warm_bias = self.warmup_bias * decay
            sigma *= (1.0 + self.warmup_sigma_boost * decay)
            self._warmup_left -= 1

        # 4) Short-term jitter (with **truncated sampling** for routine)
        #    Re-sample jitter until routine is within [min_clip, max_clip]
        #    to avoid boundary pile-ups in the histogram.
        attempts = 0
        while True:
            jitter = self._rng.gauss(0.0, sigma)
            routine = self.base + drift + jitter + warm_bias
            if self.min_clip <= routine <= self.max_clip:
                break
            attempts += 1
            if attempts >= self.max_truncation_resamples:
                # safety fallback: clamp if rejection failed too many times
                routine = max(self.min_clip, min(self.max_clip, routine))
                break

        # 5) Micro pause (small hiccup)
        micro = 0.0
        if self._rng.random() < self.p_micro:
            micro = self._rng.uniform(self.micro_min, self.micro_max)

        # 6) Slip / outlier (fat tail)
        slip = 0.0
        if self._rng.random() < self.p_slip:
            slip = self._rng.lognormvariate(self.slip_mu_ln, self.slip_sigma_ln)

        # 7) Compose: allow micro/slip to exceed the routine bounds naturally
        total = routine + micro + slip

        # 8) Guard
        return max(0.8, total)


# Optional convenience singleton-style API
_default_hc: HumanCooldown | None = None

def human_cooldown() -> float:
    """
    Stateless call facade:
        from AutoFisher.CooldownGenerator.cooldown_humanizer import human_cooldown
        time.sleep(human_cooldown())
    """
    global _default_hc
    if _default_hc is None:
        _default_hc = HumanCooldown()
    return _default_hc.next()
