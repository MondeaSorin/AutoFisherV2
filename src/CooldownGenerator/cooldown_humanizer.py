# src/AutoFisher/CooldownGenerator/cooldown_humanizer.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from math import exp, log, log1p

from .seed_utils import derive_seed, bump_seed


@dataclass(frozen=True)
class BreakProfile:
    name: str
    mu_ln: float
    sigma_ln: float
    warmup_factor: float
    floor: float


@dataclass(frozen=True)
class BreakEvent:
    name: str
    duration: float
    warmup_clicks: int


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
        # Fast routine slips (rare sub-min samples)
        fast_slip_chance: float = 0.02,
        fast_slip_decay: float = 0.32,
        fast_slip_max_deviation: float = 1.1,
        # Real breaks (short & long)
        p_break: float = 0.004,
        short_break_s: tuple[float, float] = (7.0, 25.0),
        p_long_break: float = 0.10,
        long_break_s: tuple[float, float] = (60.0, 210.0),
        short_break_mu_ln: float | None = None,
        short_break_sigma_ln: float | None = None,
        long_break_mu_ln: float | None = None,
        long_break_sigma_ln: float | None = None,
        medium_break_s: tuple[float, float] = (120.0, 600.0),
        medium_break_mu_ln: float | None = None,
        medium_break_sigma_ln: float | None = None,
        overnight_break_s: tuple[float, float] = (2400.0, 14400.0),
        overnight_break_mu_ln: float | None = None,
        overnight_break_sigma_ln: float | None = None,
        short_break_warmup_factor: float = 1.0,
        long_break_warmup_factor: float = 1.6,
        medium_break_warmup_factor: float = 1.3,
        overnight_break_warmup_factor: float = 2.4,
        short_break_floor: float | None = None,
        long_break_floor: float | None = None,
        medium_break_floor: float | None = None,
        overnight_break_floor: float | None = None,
        circadian_break_windows: dict[str, dict[str, float | tuple[int, int]]] | None = None,
        break_rekey_threshold: float = 60.0,
        break_progression_ticks: float = 260.0,
        break_progression_strength: float = 2.4,
        break_progression_exponent: float = 1.12,
        break_progression_cap: float = 0.22,
        long_break_progression_cap: float = 0.55,
        break_warmup_duration_window: float = 15.0,
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
        fatigue_soft_seconds: float = 1800.0,
        fatigue_hard_seconds: float = 7200.0,
        fatigue_reset_seconds: float = 900.0,
        fatigue_hazard_strength: float = 1.6,
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
        self.fast_slip_chance = max(0.0, fast_slip_chance)
        self.fast_slip_decay = max(1e-3, fast_slip_decay)
        self.fast_slip_max_deviation = max(0.0, fast_slip_max_deviation)
        self.p_break = p_break
        self.short_break_s = self._sanitise_bounds(short_break_s)
        self.p_long_break = p_long_break
        self.long_break_s = self._sanitise_bounds(long_break_s)
        self.medium_break_s = self._sanitise_bounds(medium_break_s)
        self.overnight_break_s = self._sanitise_bounds(overnight_break_s)
        self.short_break_warmup_factor = max(0.0, short_break_warmup_factor)
        self.long_break_warmup_factor = max(0.0, long_break_warmup_factor)
        self.medium_break_warmup_factor = max(0.0, medium_break_warmup_factor)
        self.overnight_break_warmup_factor = max(0.0, overnight_break_warmup_factor)
        self.break_rekey_threshold = max(0.0, break_rekey_threshold)
        self.break_progression_ticks = max(1.0, break_progression_ticks)
        self.break_progression_strength = max(0.0, break_progression_strength)
        self.break_progression_exponent = max(0.1, break_progression_exponent)
        self.break_progression_cap = max(
            self.p_break, min(0.95, break_progression_cap)
        )
        self.long_break_progression_cap = max(
            self.break_progression_cap, min(0.95, long_break_progression_cap)
        )
        self.break_warmup_duration_window = max(1.0, break_warmup_duration_window)
        self.short_break_floor = (
            short_break_floor
            if short_break_floor is not None
            else max(0.5, self.short_break_s[0] * 0.5)
        )
        self.long_break_floor = (
            long_break_floor
            if long_break_floor is not None
            else max(3.0, self.long_break_s[0] * 0.5)
        )
        self.medium_break_floor = (
            medium_break_floor
            if medium_break_floor is not None
            else max(1.0, self.medium_break_s[0] * 0.8)
        )
        self.overnight_break_floor = (
            overnight_break_floor
            if overnight_break_floor is not None
            else max(5.0, self.overnight_break_s[0] * 0.6)
        )

        self._short_break_mu_user = short_break_mu_ln is not None
        self._short_break_sigma_user = short_break_sigma_ln is not None
        self._long_break_mu_user = long_break_mu_ln is not None
        self._long_break_sigma_user = long_break_sigma_ln is not None
        self._medium_break_mu_user = medium_break_mu_ln is not None
        self._medium_break_sigma_user = medium_break_sigma_ln is not None
        self._overnight_break_mu_user = overnight_break_mu_ln is not None
        self._overnight_break_sigma_user = overnight_break_sigma_ln is not None

        short_mu_guess, short_sigma_guess = self._lognormal_from_bounds(
            self.short_break_s
        )
        long_mu_guess, long_sigma_guess = self._lognormal_from_bounds(
            self.long_break_s
        )
        medium_mu_guess, medium_sigma_guess = self._lognormal_from_bounds(
            self.medium_break_s
        )
        overnight_mu_guess, overnight_sigma_guess = self._lognormal_from_bounds(
            self.overnight_break_s
        )

        self.short_break_mu_ln = (
            short_break_mu_ln if short_break_mu_ln is not None else short_mu_guess
        )
        self.short_break_sigma_ln = max(
            1e-6,
            short_break_sigma_ln
            if short_break_sigma_ln is not None
            else short_sigma_guess,
        )
        self.long_break_mu_ln = (
            long_break_mu_ln if long_break_mu_ln is not None else long_mu_guess
        )
        self.long_break_sigma_ln = max(
            1e-6,
            long_break_sigma_ln
            if long_break_sigma_ln is not None
            else long_sigma_guess,
        )
        self.medium_break_mu_ln = (
            medium_break_mu_ln if medium_break_mu_ln is not None else medium_mu_guess
        )
        self.medium_break_sigma_ln = max(
            1e-6,
            medium_break_sigma_ln
            if medium_break_sigma_ln is not None
            else medium_sigma_guess,
        )
        self.overnight_break_mu_ln = (
            overnight_break_mu_ln
            if overnight_break_mu_ln is not None
            else overnight_mu_guess
        )
        self.overnight_break_sigma_ln = max(
            1e-6,
            overnight_break_sigma_ln
            if overnight_break_sigma_ln is not None
            else overnight_sigma_guess,
        )
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
        self._ticks_since_break = 0
        self._seconds_since_break = 0.0

        # State
        self._ou = 0.0
        self._warmup_left = 0
        self._warmup_span = max(1, warmup_clicks)
        self.absolute_min_cooldown = max(0.1, min_clip * 0.25, 0.8)

        # Baseline ratios so we can retune the generator around a new base at runtime
        self._design_base = base if base > 0 else 1.0
        self._design_min_clip_factor = min_clip / self._design_base
        self._design_max_clip_factor = max_clip / self._design_base
        self._design_micro_min_factor = micro_min / self._design_base
        self._design_micro_max_factor = micro_max / self._design_base
        self._design_short_break_factor = tuple(
            v / self._design_base for v in self.short_break_s
        )
        self._design_long_break_factor = tuple(
            v / self._design_base for v in self.long_break_s
        )
        self._design_medium_break_factor = tuple(
            v / self._design_base for v in self.medium_break_s
        )
        self._design_overnight_break_factor = tuple(
            v / self._design_base for v in self.overnight_break_s
        )
        self._design_warmup_bias_factor = warmup_bias / self._design_base
        self._design_slip_mu_ln = slip_mu_ln
        self._design_fast_slip_decay_factor = fast_slip_decay / self._design_base
        self._design_fast_slip_span_factor = (
            fast_slip_max_deviation / self._design_base
            if self._design_base > 0
            else fast_slip_max_deviation
        )
        self._design_absolute_min_factor = (
            self.absolute_min_cooldown / self._design_base
            if self._design_base > 0
            else None
        )
        self._design_short_break_floor_factor = (
            self.short_break_floor / self._design_base
        )
        self._design_long_break_floor_factor = (
            self.long_break_floor / self._design_base
        )
        self._design_medium_break_floor_factor = (
            self.medium_break_floor / self._design_base
        )
        self._design_overnight_break_floor_factor = (
            self.overnight_break_floor / self._design_base
        )
        self._design_short_break_mu_ln = (
            self.short_break_mu_ln if self._short_break_mu_user else None
        )
        self._design_long_break_mu_ln = (
            self.long_break_mu_ln if self._long_break_mu_user else None
        )
        self._design_medium_break_mu_ln = (
            self.medium_break_mu_ln if self._medium_break_mu_user else None
        )
        self._design_overnight_break_mu_ln = (
            self.overnight_break_mu_ln if self._overnight_break_mu_user else None
        )
        self._design_short_break_sigma_ln = (
            self.short_break_sigma_ln if self._short_break_sigma_user else None
        )
        self._design_long_break_sigma_ln = (
            self.long_break_sigma_ln if self._long_break_sigma_user else None
        )
        self._design_medium_break_sigma_ln = (
            self.medium_break_sigma_ln if self._medium_break_sigma_user else None
        )
        self._design_overnight_break_sigma_ln = (
            self.overnight_break_sigma_ln if self._overnight_break_sigma_user else None
        )

        default_circadian = {
            "medium": {
                "window": (10, 15),
                "median": 420.0,
                "off_median": 270.0,
                "floor": 150.0,
                "off_floor": 90.0,
                "boost": 1.3,
                "off_boost": 0.9,
            },
            "overnight": {
                "window": (22, 6),
                "median": 7200.0,
                "off_median": 5400.0,
                "floor": 2400.0,
                "off_floor": 1500.0,
                "boost": 2.0,
                "off_boost": 0.7,
            },
        }
        user_circadian = circadian_break_windows or {}
        circadian_config: dict[str, dict[str, float | tuple[int, int]]] = {}
        for key in set(default_circadian) | set(user_circadian):
            merged: dict[str, float | tuple[int, int]] = dict(
                default_circadian.get(key, {})
            )
            merged.update(user_circadian.get(key, {}))
            window = merged.get("window")
            if isinstance(window, (list, tuple)) and len(window) == 2:
                start, end = int(window[0]), int(window[1])
                merged["window"] = (
                    max(0, min(23, start)),
                    max(0, min(23, end)),
                )
            circadian_config[key] = merged
        self.circadian_break_windows = circadian_config

        long_band = max(0.0, min(0.95, self.p_long_break))
        medium_share = long_band * 0.35
        overnight_share = long_band * 0.2
        long_share = max(0.0, long_band - medium_share - overnight_share)
        short_share = max(1e-6, 1.0 - long_band)
        self._profile_base_weights = {
            "short": short_share,
            "long": max(1e-6, long_share),
            "medium": max(1e-6, medium_share),
            "overnight": max(1e-6, overnight_share),
        }
        self._long_weight_total = (
            self._profile_base_weights["long"]
            + self._profile_base_weights["medium"]
            + self._profile_base_weights["overnight"]
        )

        self.fatigue_soft_seconds = max(0.0, fatigue_soft_seconds)
        self.fatigue_hard_seconds = max(
            self.fatigue_soft_seconds + 1.0, fatigue_hard_seconds
        )
        self.fatigue_reset_seconds = max(1.0, fatigue_reset_seconds)
        self.fatigue_hazard_strength = max(0.0, fatigue_hazard_strength)

        self._session_start_wall = time.time()
        self._session_seconds = 0.0
        self._last_tick_wall_time = self._session_start_wall
        self._profile_refresh_hour: int | None = None

        self._refresh_break_profiles()

    # --- internals ---

    @staticmethod
    def _sanitise_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
        lo, hi = bounds
        lo = max(1e-3, lo)
        hi = max(lo * 1.05, hi)
        return lo, hi

    @staticmethod
    def _lognormal_from_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
        lo, hi = HumanCooldown._sanitise_bounds(bounds)
        sigma = (log(hi) - log(lo)) / (2.0 * 1.645)
        sigma = max(1e-6, sigma)
        mu = (log(lo) + log(hi)) / 2.0
        return mu, sigma

    @staticmethod
    def _hour_in_window(hour: int, window: tuple[int, int]) -> bool:
        start, end = window
        start %= 24
        end %= 24
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    def _refresh_break_profiles(self, hour: int | None = None) -> None:
        if hour is None:
            hour = time.localtime().tm_hour
        self._profile_refresh_hour = hour

        def _band(name: str, floor_default: float, median_default: float) -> tuple[float, float]:
            config = self.circadian_break_windows.get(name, {})
            window = config.get("window")
            in_window = False
            if isinstance(window, tuple):
                in_window = self._hour_in_window(hour, window)
            floor_key = "floor" if in_window else "off_floor"
            median_key = "median" if in_window else "off_median"
            floor_val = config.get(floor_key, floor_default)
            median_val = config.get(median_key, median_default)
            try:
                floor = max(0.5, float(floor_val))
            except (TypeError, ValueError):
                floor = floor_default
            try:
                median = max(0.5, float(median_val))
            except (TypeError, ValueError):
                median = median_default
            return floor, median

        medium_floor, medium_median = _band(
            "medium", self.medium_break_floor, exp(self.medium_break_mu_ln)
        )
        overnight_floor, overnight_median = _band(
            "overnight", self.overnight_break_floor, exp(self.overnight_break_mu_ln)
        )

        self._break_profiles = {
            "short": BreakProfile(
                "short",
                self.short_break_mu_ln,
                self.short_break_sigma_ln,
                self.short_break_warmup_factor,
                self.short_break_floor,
            ),
            "long": BreakProfile(
                "long",
                self.long_break_mu_ln,
                self.long_break_sigma_ln,
                self.long_break_warmup_factor,
                self.long_break_floor,
            ),
            "medium": BreakProfile(
                "medium",
                log(medium_median),
                self.medium_break_sigma_ln,
                self.medium_break_warmup_factor,
                medium_floor,
            ),
            "overnight": BreakProfile(
                "overnight",
                log(overnight_median),
                self.overnight_break_sigma_ln,
                self.overnight_break_warmup_factor,
                overnight_floor,
            ),
        }

    def _circadian_multiplier(self, profile: str, hour: int) -> float:
        config = self.circadian_break_windows.get(profile)
        if not config:
            return 1.0
        window = config.get("window")
        in_window = False
        if isinstance(window, tuple):
            in_window = self._hour_in_window(hour, window)
        key = "boost" if in_window else "off_boost"
        value = config.get(key, 1.0)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 1.0

    def _fatigue_multiplier(self) -> float:
        if self._session_seconds <= self.fatigue_soft_seconds:
            return 1.0
        span = max(1.0, self.fatigue_hard_seconds - self.fatigue_soft_seconds)
        progress = min(1.0, (self._session_seconds - self.fatigue_soft_seconds) / span)
        return 1.0 + progress * self.fatigue_hazard_strength

    def _compute_break_hazard(self) -> dict[str, float]:
        current_hour = time.localtime().tm_hour
        if self._profile_refresh_hour != current_hour:
            self._refresh_break_profiles(current_hour)

        exposure_ticks = max(0.0, self._ticks_since_break)
        exposure_seconds = max(0.0, self._seconds_since_break)
        baseline_scale = max(1.0, self.break_progression_ticks)
        base_seconds = max(1.0, self.break_progression_ticks * max(self.base, 1e-3))
        scaled = exposure_ticks / baseline_scale + exposure_seconds / base_seconds
        ramp = log1p(scaled)
        boost = 1.0 + self.break_progression_strength * (ramp ** self.break_progression_exponent)

        base_hazard = self.p_break * boost
        long_portion = max(0.0, min(1.0, self.p_long_break))
        short_base = base_hazard * max(0.0, 1.0 - long_portion)
        short_hazard = min(self.break_progression_cap, short_base)

        longish_base = base_hazard * long_portion
        longish_base *= self._fatigue_multiplier()

        weights: dict[str, float] = {}
        for profile in ("long", "medium", "overnight"):
            base_weight = self._profile_base_weights.get(profile, 0.0)
            if base_weight <= 0.0:
                weights[profile] = 0.0
                continue
            circ = self._circadian_multiplier(profile, current_hour)
            weights[profile] = base_weight * circ

        weighted_total = sum(weights.values())
        base_total = self._long_weight_total if self._long_weight_total > 0 else 1.0
        if weighted_total > 0:
            circadian_boost = max(0.0, weighted_total / base_total)
            longish_base *= circadian_boost
        else:
            longish_base = 0.0

        longish_base = min(self.long_break_progression_cap, longish_base)

        hazards: dict[str, float] = {"short": short_hazard}

        if longish_base <= 0.0 or weighted_total <= 0.0:
            hazards.update({"long": 0.0, "medium": 0.0, "overnight": 0.0})
            return hazards

        for profile in ("long", "medium", "overnight"):
            weight = weights.get(profile, 0.0)
            if weight <= 0.0:
                hazards[profile] = 0.0
                continue
            share = weight / weighted_total
            hazards[profile] = min(self.long_break_progression_cap, longish_base * share)

        return hazards

    def _select_break_profile(
        self, hazards: dict[str, float], total: float
    ) -> BreakProfile:
        r = self._rng.random() * max(total, 1e-9)
        cumulative = 0.0
        for name in ("short", "medium", "long", "overnight"):
            hazard = max(0.0, hazards.get(name, 0.0))
            if hazard <= 0.0:
                continue
            cumulative += hazard
            if r < cumulative:
                return self._break_profiles[name]
        return self._break_profiles["short"]

    def _sample_break_duration(self, profile: BreakProfile) -> float:
        duration = self._rng.lognormvariate(profile.mu_ln, profile.sigma_ln)
        return max(profile.floor, duration)

    def _compute_warmup_clicks(self, duration: float, profile: BreakProfile) -> int:
        base_window = max(1.0, self.break_warmup_duration_window)
        baseline = max(1.0, self.base)
        duration_scale = 1.0 + min(1.0, duration / (baseline * base_window))
        warmup = self.warmup_clicks * profile.warmup_factor * duration_scale
        return max(0, int(round(warmup)))

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

    def _maybe_break(self) -> BreakEvent | None:
        hazards = self._compute_break_hazard()
        total_hazard = sum(max(0.0, h) for h in hazards.values())
        total_hazard = max(0.0, min(0.95, total_hazard))
        if total_hazard <= 0.0:
            return None
        if self._rng.random() >= total_hazard:
            return None

        profile = self._select_break_profile(hazards, total_hazard)
        duration = self._sample_break_duration(profile)
        warmup_clicks = self._compute_warmup_clicks(duration, profile)
        return BreakEvent(profile.name, duration, warmup_clicks)

    # --- public API ---

    def next(self) -> float:
        """Return the next cooldown (seconds)."""
        now = time.time()
        if self._last_tick_wall_time is not None:
            rest_wall = max(0.0, now - self._last_tick_wall_time)
            if rest_wall >= self.fatigue_reset_seconds:
                self._session_seconds = max(0.0, self._session_seconds - rest_wall)
        else:
            self._session_start_wall = now
        self._last_tick_wall_time = now

        self._ticks += 1
        if (self._ticks % self._rekey_every) == 0:
            n = self._ticks // self._rekey_every
            self._rekey(n)

        # 1) Real break?
        event = self._maybe_break()
        if event is not None:
            if event.duration >= self.break_rekey_threshold:
                n = (self._ticks // self._rekey_every) + 1
                self._rekey(n)
            self._warmup_left = event.warmup_clicks
            if event.warmup_clicks > 0:
                self._warmup_span = event.warmup_clicks
            else:
                self._warmup_span = max(1, self.warmup_clicks)
            self._ticks_since_break = 0
            self._seconds_since_break = 0.0
            return event.duration

        self._ticks_since_break += 1

        # 2) Slow drift
        drift = self._ou_step()

        # 3) Warm-up shaping (decaying bias + variance boost)
        sigma = self.sigma_short
        warm_bias = 0.0
        if self._warmup_left > 0:
            k = max(1, self._warmup_span)
            i = self._warmup_left
            fraction = max(0.0, min(1.0, i / k))
            decay = exp(-2.0 * (1.0 - fraction))  # ~0.135 .. 1 ramp
            warm_bias = self.warmup_bias * decay
            sigma *= (1.0 + self.warmup_sigma_boost * decay)
            self._warmup_left -= 1

        # 4) Short-term jitter (with **truncated sampling** for routine)
        #    Re-sample jitter until routine is within [min_clip, max_clip]
        #    to avoid boundary pile-ups in the histogram.
        attempts = 0
        lower_guard = max(
            self.absolute_min_cooldown, self.min_clip - self.fast_slip_max_deviation
        )
        while True:
            jitter = self._rng.gauss(0.0, sigma)
            routine = self.base + drift + jitter + warm_bias
            if self.min_clip <= routine <= self.max_clip:
                break
            deficit = self.min_clip - routine
            if deficit > 0:
                fast_prob = self.fast_slip_chance * exp(
                    -deficit / max(1e-6, self.fast_slip_decay)
                )
                fast_prob = max(0.0, min(0.45, fast_prob))
                if self._rng.random() < fast_prob:
                    routine = max(lower_guard, routine)
                    break
            attempts += 1
            if attempts >= self.max_truncation_resamples:
                # safety fallback: clamp if rejection failed too many times
                routine = max(lower_guard, min(self.max_clip, routine))
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

        # 8) Guard & bookkeeping
        result = max(self.absolute_min_cooldown, total)
        self._seconds_since_break += result
        self._session_seconds += result
        return result

    # --- configuration helpers ---

    def retune_base(self, base: float) -> None:
        """Re-centre the stochastic model around a new base interval."""

        if base <= 0:
            raise ValueError("HumanCooldown base must be positive.")

        scale = base / self._design_base if self._design_base > 0 else 1.0

        self.base = base
        self.min_clip = max(0.1, base * self._design_min_clip_factor)
        self.max_clip = max(self.min_clip + 0.1, base * self._design_max_clip_factor)
        self.micro_min = max(0.01, base * self._design_micro_min_factor)
        self.micro_max = max(self.micro_min + 0.01, base * self._design_micro_max_factor)
        self.fast_slip_decay = max(
            1e-3, base * self._design_fast_slip_decay_factor
        )
        self.fast_slip_max_deviation = max(
            0.0, base * self._design_fast_slip_span_factor
        )
        short_scaled = tuple(
            max(1.0, base * f) for f in self._design_short_break_factor
        )
        long_scaled = tuple(
            max(5.0, base * f) for f in self._design_long_break_factor
        )
        medium_scaled = tuple(
            max(5.0, base * f) for f in self._design_medium_break_factor
        )
        overnight_scaled = tuple(
            max(30.0, base * f) for f in self._design_overnight_break_factor
        )
        self.short_break_s = self._sanitise_bounds(short_scaled)
        self.long_break_s = self._sanitise_bounds(long_scaled)
        self.medium_break_s = self._sanitise_bounds(medium_scaled)
        self.overnight_break_s = self._sanitise_bounds(overnight_scaled)
        self.short_break_floor = max(
            0.1, base * self._design_short_break_floor_factor
        )
        self.long_break_floor = max(
            0.1, base * self._design_long_break_floor_factor
        )
        self.medium_break_floor = max(
            0.1, base * self._design_medium_break_floor_factor
        )
        self.overnight_break_floor = max(
            0.1, base * self._design_overnight_break_floor_factor
        )

        if self._design_short_break_mu_ln is not None and scale > 0:
            self.short_break_mu_ln = self._design_short_break_mu_ln + log(scale)
        else:
            short_mu, short_sigma = self._lognormal_from_bounds(self.short_break_s)
            self.short_break_mu_ln = short_mu
            if not self._short_break_sigma_user:
                self.short_break_sigma_ln = short_sigma

        if self._design_short_break_sigma_ln is not None:
            self.short_break_sigma_ln = max(1e-6, self._design_short_break_sigma_ln)

        if self._design_long_break_mu_ln is not None and scale > 0:
            self.long_break_mu_ln = self._design_long_break_mu_ln + log(scale)
        else:
            long_mu, long_sigma = self._lognormal_from_bounds(self.long_break_s)
            self.long_break_mu_ln = long_mu
            if not self._long_break_sigma_user:
                self.long_break_sigma_ln = long_sigma

        if self._design_long_break_sigma_ln is not None:
            self.long_break_sigma_ln = max(1e-6, self._design_long_break_sigma_ln)

        if self._design_medium_break_mu_ln is not None and scale > 0:
            self.medium_break_mu_ln = self._design_medium_break_mu_ln + log(scale)
        else:
            medium_mu, medium_sigma = self._lognormal_from_bounds(self.medium_break_s)
            self.medium_break_mu_ln = medium_mu
            if not self._medium_break_sigma_user:
                self.medium_break_sigma_ln = medium_sigma

        if self._design_medium_break_sigma_ln is not None:
            self.medium_break_sigma_ln = max(1e-6, self._design_medium_break_sigma_ln)

        if self._design_overnight_break_mu_ln is not None and scale > 0:
            self.overnight_break_mu_ln = self._design_overnight_break_mu_ln + log(scale)
        else:
            overnight_mu, overnight_sigma = self._lognormal_from_bounds(
                self.overnight_break_s
            )
            self.overnight_break_mu_ln = overnight_mu
            if not self._overnight_break_sigma_user:
                self.overnight_break_sigma_ln = overnight_sigma

        if self._design_overnight_break_sigma_ln is not None:
            self.overnight_break_sigma_ln = max(
                1e-6, self._design_overnight_break_sigma_ln
            )

        self.warmup_bias = self._design_warmup_bias_factor * base

        if scale > 0:
            self.slip_mu_ln = self._design_slip_mu_ln + log(scale)

        if self._design_absolute_min_factor is not None:
            self.absolute_min_cooldown = max(
                0.1, base * self._design_absolute_min_factor
            )
        else:
            self.absolute_min_cooldown = max(
                0.1, self.min_clip * 0.25, self.absolute_min_cooldown
            )

        self._refresh_break_profiles()
        self._warmup_span = max(1, self.warmup_clicks)


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
