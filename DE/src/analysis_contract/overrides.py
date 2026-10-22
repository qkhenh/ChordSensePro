"""Effective timeline = immutable AI baseline + the personal override layer."""

from collections.abc import Iterable

from .models import ChordEvent, EventOverride, OverrideOp, validate_timeline


class OverlapError(ValueError):
    """Applying the overrides produced overlapping or unsorted events."""


def apply_overrides(
    ai_events: Iterable[ChordEvent], overrides: Iterable[EventOverride]
) -> list[ChordEvent]:
    """Return the effective events; the baseline is never mutated.

    Events touched by ``set``/``insert`` are returned with ``is_user_edited=True``.
    """
    by_id = {e.event_id: e for e in ai_events}
    for o in overrides:
        if o.op == OverrideOp.DELETE:
            by_id.pop(o.event_id, None)
        else:
            if o.event is None:  # unreachable for a validated EventOverride
                raise ValueError(f"{o.op.value} override {o.event_id} has no event")
            by_id[o.event_id] = o.event.model_copy(update={"is_user_edited": True})
    effective = sorted(by_id.values(), key=lambda e: e.start_ms)
    try:
        validate_timeline(effective)
    except ValueError as exc:
        raise OverlapError(str(exc)) from exc
    return effective
