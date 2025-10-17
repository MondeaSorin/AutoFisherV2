# -*- coding: utf-8 -*-
"""
Competitive human-like cooldown generator (class-only).
Usage:
    from cooldown_humanizer import HumanizedCooldown
    hc = HumanizedCooldown(base=3.0, seed=1337)
    lag = hc.next()
Design goals:
- Metronome-locked human tapping with negative lag-1 autocorrelation (error correction)
- Asymmetric distribution: short left tail, natural right tail
- Competitive spread (tight), tunable via parameters
- No automatic long breaks; breaks can be controlled externally via `manual_break_end()`
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class HumanParams:
    # Base interval in seconds (e.g., 3.0)
    base: float = 3.0

    # Human error-correction gain: 0.3–0.6 typical for competitive
    k_correction: float = 0.45

    # Motor noise (std dev, seconds)
    sigma_motor: float = 0.08

    # Maximum early lead from symmetric noise (seconds) — short left tail
    left_clip: float = 0.12

    # Right-tail slip probability per press
    p_tail: float = 0.015

    # Lognormal parameters for right-tail slip
    tail_mu_ln: float = -1.6
    tail_sigma_ln: float = 0.55

    # Optional micro-pause shaping (positive-only), feeds right tail
    p_micro: float = 0.10
    micro_min: float = 0.010
    micro_max: float = 0.035

    # Absolute minimal cooldown guard
    absolute_min_cooldown: float = 0.050

    # Warmup after manual break (press count)
    warmup_presses: int = 12
    warmup_sigma_boost: float = 0.60  # extra sigma factor at start, decays to 0
    warmup_bias: float = 0.0          # optional mean bias during warmup

    # Subtle fatigue shaping across long sessions
    fatigue_sigma_per_hour: float = 0.006           # +6 ms sigma per hour
    fatigue_gain_decay_per_hour: float = 0.02       # -2% gain per hour

    # RNG seed
    seed: Optional[int] = None


class HumanCooldown:
    """
    Single-instance, competitive human-like cooldown generator.
    Call `next()` to get the next interval (seconds).
    """
    __slots__ = (
        "_rng",
        "p",
        "_asynchrony",
        "_ideal_next_time",
        "_warmup_left",
        "_session_seconds",
        "_seconds_since_break",
        "_press_count_since_break",
    )

    def __init__(self, *, base: float = 3.0, seed: Optional[int] = None, **kwargs):
        # Allow overriding any HumanParams via kwargs
        params = HumanParams(base=base, seed=seed)
        for k, v in kwargs.items():
            if hasattr(params, k):
                setattr(params, k, v)
        self.p: HumanParams = params

        self._rng = random.Random(self.p.seed)
        self._asynchrony: float = 0.0          # actual - ideal
        self._ideal_next_time: float = 0.0     # abstract beat timeline
        self._warmup_left: int = 0
        self._session_seconds: float = 0.0
        self._seconds_since_break: float = 0.0
        self._press_count_since_break: int = 0

    # ----------------- Public API -----------------

    def start_session(self) -> None:
        """Initialize a new session (optional)."""
        self._reset_phase()
        self._warmup_left = 0
        self._session_seconds = 0.0
        self._seconds_since_break = 0.0
        self._press_count_since_break = 0

    def manual_break_end(self) -> None:
        """
        Call this after your own externally-controlled long break.
        Applies a short warmup and re-synchronizes phase.
        """
        self._reset_phase()
        self._warmup_left = max(self._warmup_left, self.p.warmup_presses)
        self._seconds_since_break = 0.0
        self._press_count_since_break = 0

    def next(self) -> float:
        """
        Return the next cooldown interval in seconds.
        Competitive human profile: metronome-locked with human jitter, no auto breaks.
        """
        p = self.p

        # -------- Warmup shaping (after manual break) --------
        sigma = p.sigma_motor
        warm_bias = 0.0
        if self._warmup_left > 0:
            i = self._warmup_left
            k = max(1, p.warmup_presses)
            fraction = max(0.0, min(1.0, i / k))
            decay = math.exp(-2.0 * (1.0 - fraction))  # 1 -> 0
            sigma *= (1.0 + p.warmup_sigma_boost * decay)
            warm_bias = p.warmup_bias * decay
            self._warmup_left -= 1

        # -------- Subtle fatigue shaping with session time --------
        hours = self._session_seconds / 3600.0 if self._session_seconds > 0.0 else 0.0
        sigma += p.fatigue_sigma_per_hour * hours
        k_corr = p.k_correction * (1.0 - p.fatigue_gain_decay_per_hour * hours)
        if k_corr < 0.2:
            k_corr = 0.2
        elif k_corr > 0.8:
            k_corr = 0.8

        # -------- Symmetric motor noise, left-truncated --------
        eps = self._rng.gauss(0.0, sigma)
        if eps < -p.left_clip:
            eps = -p.left_clip + abs(self._rng.gauss(0.0, sigma * 0.25))

        # -------- Rare right-tail delay (lognormal) --------
        delay_tail = 0.0
        if self._rng.random() < p.p_tail:
            delay_tail = self._rng.lognormvariate(p.tail_mu_ln, p.tail_sigma_ln)

        # -------- Optional tiny micro-pause (positive-only) --------
        micro = 0.0
        if self._rng.random() < p.p_micro:
            micro = self._rng.uniform(p.micro_min, p.micro_max)

        # -------- Human error-correction to the beat (negative lag-1) --------
        corr = -k_corr * self._asynchrony

        # -------- Routine interval --------
        routine = p.base + corr + warm_bias + eps

        # -------- Compose with positive-only delays (right-skew) --------
        total = routine + micro + delay_tail
        result = total if total > p.absolute_min_cooldown else p.absolute_min_cooldown

        # -------- Update phase and bookkeeping --------
        self._ideal_next_time += p.base
        self._asynchrony = self._asynchrony + result - p.base

        self._press_count_since_break += 1
        self._seconds_since_break += result
        self._session_seconds += result

        return result

    # ----------------- Internal helpers -----------------

    def _reset_phase(self) -> None:
        self._ideal_next_time = 0.0
        self._asynchrony = 0.0
        # warmup left remains as set by caller

