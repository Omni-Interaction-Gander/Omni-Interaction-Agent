from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


DEFAULT_SHORT_GAP_BANDS = (
    (1, 3, 0.40),
    (4, 6, 0.35),
    (7, 10, 0.25),
)
DEFAULT_LONG_GAP_BANDS = (
    (5, 9, 0.40),
    (10, 14, 0.35),
    (15, 20, 0.25),
)
DEFAULT_HUMAN_TO_HUMAN_GAP_BANDS = (
    (1, 3, 0.35),
    (4, 6, 0.10),
)

# Agent profiles opt in to task-specific timeline augmentation.
DEFAULT_PENDING_TASK_GAP_BANDS: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class TimelinePolicy:
    enabled: bool = False
    insert_start: bool = True
    insert_between_turns: bool = True
    short_gap_bands: Sequence[Sequence[float]] = DEFAULT_SHORT_GAP_BANDS
    human_to_human_probability: float = 0.0
    human_to_human_gap_bands: Sequence[
        Sequence[float]
    ] = DEFAULT_HUMAN_TO_HUMAN_GAP_BANDS
    long_gap_probability: float = 0.0
    long_gap_bands: Sequence[Sequence[float]] = DEFAULT_LONG_GAP_BANDS
    long_gap_locations: Sequence[str] = ("start", "between", "tail")
    max_long_spans: int = 1
    # Event-free listen span after a task acknowledgement. Worker delivery moves
    # after the span; pending tasks receive an equivalent tail.
    pending_task_gap_bands: Sequence[
        Sequence[float]
    ] = DEFAULT_PENDING_TASK_GAP_BANDS
    # Optional unit cap for idle-gap augmentation.
    max_timeline_units: int | None = None


@dataclass(frozen=True)
class AugmentationProfile:
    name: str
    # Coherent acoustic scene applied independently of user speech activity.
    background_probability: float = 0.0
    category_weights: Mapping[str, float] = field(
        default_factory=lambda: {"ambient": 1.0}
    )
    snr_bands: Sequence[Sequence[float]] = ((15.0, 25.0, 1.0),)
    floor_probability: float = 0.0
    floor_category_weights: Mapping[str, float] = field(
        default_factory=lambda: {"ambient": 1.0}
    )
    floor_snr_bands: Sequence[Sequence[float]] = ((22.0, 30.0, 1.0),)
    transient_probability: float = 0.0
    device_probability: float = 0.0
    room_probability: float = 0.0
    room_preset_weights: Mapping[str, float] = field(default_factory=dict)
    speaker_spatial_probability: float = 0.0
    speaker_distance_weights: Mapping[str, float] = field(default_factory=dict)
    echo_probability: float = 0.0
    combined_probability: float = 0.0
    timeline: TimelinePolicy = field(default_factory=TimelinePolicy)

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "AugmentationProfile":
        timeline_value = value.get("timeline") or {}
        if not isinstance(timeline_value, Mapping):
            raise ValueError(f"Profile {name!r} timeline must be a mapping")
        timeline = TimelinePolicy(**dict(timeline_value))
        values = {key: item for key, item in value.items() if key != "timeline"}
        return cls(name=name, timeline=timeline, **values)


class ProfileResolver:
    """Resolve augmentation behavior from manifest metadata, without source-specific code."""

    def __init__(
        self,
        profiles: Mapping[str, Mapping[str, Any]],
        rules: Sequence[Mapping[str, Any]],
        *,
        default_profile: str,
    ) -> None:
        self.profiles = {
            str(name): AugmentationProfile.from_mapping(str(name), value)
            for name, value in profiles.items()
        }
        if default_profile not in self.profiles:
            raise ValueError(
                f"default_profile={default_profile!r} is absent from configured profiles"
            )
        self.default_profile = str(default_profile)
        self.rules: list[tuple[str, Mapping[str, Any]]] = []
        for index, rule in enumerate(rules):
            if not isinstance(rule, Mapping):
                raise ValueError(f"profile_rules[{index}] must be a mapping")
            profile = str(rule.get("profile") or "")
            match = rule.get("match") or {}
            if profile not in self.profiles:
                raise ValueError(
                    f"profile_rules[{index}] references unknown profile {profile!r}"
                )
            if not isinstance(match, Mapping) or not match:
                raise ValueError(f"profile_rules[{index}].match must be a non-empty mapping")
            self.rules.append((profile, match))

    def resolve(self, meta: Mapping[str, Any]) -> AugmentationProfile:
        explicit = meta.get("augmentation_profile")
        if explicit is not None:
            profile = str(explicit)
            if profile not in self.profiles:
                raise ValueError(
                    f"Sample requests unknown augmentation_profile={profile!r}"
                )
            return self.profiles[profile]
        for profile, match in self.rules:
            if _matches(meta, match):
                return self.profiles[profile]
        return self.profiles[self.default_profile]


def _matches(meta: Mapping[str, Any], match: Mapping[str, Any]) -> bool:
    for field_name, expected in match.items():
        actual = _get_dotted(meta, str(field_name))
        patterns = expected if isinstance(expected, (list, tuple)) else [expected]
        if not any(_matches_value(actual, pattern) for pattern in patterns):
            return False
    return True


def _get_dotted(meta: Mapping[str, Any], field_name: str) -> Any:
    value: Any = meta
    for component in field_name.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(component)
    return value


def _matches_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, (int, float)):
        return actual == expected
    if actual is None:
        return False
    return fnmatch.fnmatchcase(str(actual).lower(), str(expected).lower())
