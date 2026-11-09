"""Hooktheory (Sheet Sage ``Hooktheory.json.gz``) → ChordSense annotation file.

Source dataset: https://github.com/chrisdonahue/sheetsage-data (CC BY-NC-SA 3.0, no audio).
Each entry carries a YouTube video, a beat→seconds alignment of a short (~20 s) clip and the
clip's harmony as ``root_pitch_class`` + ``root_position_intervals`` (+ ``inversion``).

What this module does
---------------------
* ``harmony_label``    stacked intervals → a Harte label the existing ``map_harte`` understands.
* ``beat_to_time``     piecewise-linear beat→second interpolation of the alignment.
* ``build_annotation`` one dataset entry → a small JSON (the *annotation file* of the crawl queue):

      {"format": "hooktheory", "id": ..., "youtube_id": ..., "duration": <video seconds>,
       "section": [start, end],           # absolute seconds of the video that gets downloaded
       "labelled_range": [start, end],    # clip time (0 = section start): where labels are valid
       "timeline": [[start, end, "C:(3,5,b7,9)"], ...]}   # clip time

Timeline times are **relative to the downloaded section**, so the existing midpoint lookup in
AnnotationHandler works unchanged. Only the annotated part of the song is labelled; the padding
around it (and anything outside ``labelled_range``) must not become 'N' — AnnotationHandler drops it.
"""
from __future__ import annotations
from bisect import bisect_right

_ROOTS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# semitones above the root → Harte degree (inside the first octave)
_DEG_LOW = {1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5", 8: "#5", 9: "6", 10: "b7", 11: "7"}
# second octave (compound intervals, 12..23) → Harte degree
_DEG_HIGH = {12: "1", 13: "b9", 14: "9", 15: "#9", 16: "3", 17: "11", 18: "#11", 19: "5",
             20: "b13", 21: "13", 22: "b7", 23: "7"}

DEFAULT_PAD_SEC = 3.0       # audio kept around the labelled region (context for Demucs / beat tracker)
MIN_LABELLED_SEC = 4.0      # shorter than two segments' worth of material → not worth a download


def _degree(semitones: int) -> str | None:
    if semitones in (0, 12, 24):
        return None                                   # root / octave duplicates carry no information
    if semitones < 12:
        return _DEG_LOW.get(semitones)
    return _DEG_HIGH.get(semitones)


def harmony_label(root_pitch_class: int, intervals: list[int]) -> str:
    """Hooktheory chord → Harte label, e.g. (7, [4,3,7]) → ``'G:(3,5,9)'`` (= Gadd9).

    ``intervals`` are successive steps above the root (``[4,3,4]`` = maj7). The result is an
    interval-list label (``Root:(deg,deg,...)``) except for added-6th chords, which use the
    ``maj6`` / ``min6`` shorthand because the interval-list parser has no 6th class.
    """
    root = _ROOTS[root_pitch_class % 12]
    semitones: list[int] = []
    total = 0
    for step in intervals:
        total += step
        semitones.append(total)

    degrees = [d for d in (_degree(s) for s in semitones) if d]
    if not degrees:
        return f"{root}:(1)"                          # lone root → map_harte: PARTIAL (root only)

    if "6" in degrees:
        has_b3, has_b5 = "b3" in degrees, "b5" in degrees
        if has_b3 and has_b5:                         # diminished seventh: stacked minor thirds
            degrees = ["bb7" if d == "6" else d for d in degrees]
        elif "b7" not in degrees and "7" not in degrees:
            return f"{root}:{'min6' if has_b3 else 'maj6'}"
        else:                                         # a 6th stacked on a seventh is the 13th
            degrees = ["13" if d == "6" else d for d in degrees]
    return f"{root}:({','.join(degrees)})"


def beat_to_time(beat: float, beats: list[float], times: list[float]) -> float:
    """Piecewise-linear beat → seconds; extrapolates with the first / last beat interval."""
    n = len(beats)
    if n == 1:
        return times[0]
    if beat <= beats[0]:
        i = 0
    elif beat >= beats[-1]:
        i = n - 2
    else:
        i = bisect_right(beats, beat) - 1
    b0, b1, t0, t1 = beats[i], beats[i + 1], times[i], times[i + 1]
    if b1 == b0:
        return t0
    return t0 + (beat - b0) * (t1 - t0) / (b1 - b0)


def usable_entry(entry: dict) -> bool:
    """Cheap pre-filter: has audio + harmony, no tempo changes, a YouTube id and its duration."""
    tags = set(entry.get("tags") or [])
    yt = entry.get("youtube") or {}
    return (
        {"HARMONY", "AUDIO_AVAILABLE"} <= tags
        and "TEMPO_CHANGES" not in tags
        and bool(yt.get("id"))
        and bool(yt.get("duration"))
        and bool((entry.get("annotations") or {}).get("harmony"))
    )


def build_annotation(entry: dict, pad: float = DEFAULT_PAD_SEC, refined_only: bool = True) -> dict | None:
    """One Hooktheory entry → annotation dict (see module doc), or None if not usable."""
    if not usable_entry(entry):
        return None
    alignment = entry.get("alignment") or {}
    align = alignment.get("refined") or (None if refined_only else alignment.get("user"))
    if not align or not align.get("beats") or not align.get("times"):
        return None
    beats = [float(b) for b in align["beats"]]
    times = [float(t) for t in align["times"]]
    if len(beats) != len(times) or len(beats) < 2 or any(b2 <= b1 for b1, b2 in zip(beats, beats[1:])):
        return None

    events: list[tuple[float, float, str]] = []
    for h in entry["annotations"]["harmony"]:
        try:
            onset, offset = float(h["onset"]), float(h["offset"])
            label = harmony_label(h["root_pitch_class"], list(h["root_position_intervals"]))
        except (KeyError, TypeError, ValueError):
            continue
        start = beat_to_time(onset, beats, times)
        end = beat_to_time(offset, beats, times)
        if end > start:
            events.append((start, end, label))
    if not events:
        return None
    events.sort(key=lambda e: e[0])

    labelled_start, labelled_end = events[0][0], max(e[1] for e in events)
    if labelled_end - labelled_start < MIN_LABELLED_SEC or labelled_start < 0:
        return None

    yt = entry["youtube"]
    video_duration = float(yt["duration"])
    section_start = max(0.0, labelled_start - pad)
    section_end = min(video_duration, labelled_end + pad)
    if section_end - section_start < MIN_LABELLED_SEC or labelled_start >= video_duration:
        return None

    meta = entry.get("hooktheory") or {}
    return {
        "format": "hooktheory",
        "id": meta.get("id", ""),
        "artist": meta.get("artist", ""),
        "song": meta.get("song", ""),
        "youtube_id": yt["id"],
        "duration": round(video_duration, 3),
        "section": [round(section_start, 3), round(section_end, 3)],
        "labelled_range": [round(labelled_start - section_start, 3),
                           round(min(labelled_end, section_end) - section_start, 3)],
        "timeline": [
            [round(s - section_start, 3), round(min(e, section_end) - section_start, 3), label]
            for s, e, label in events if s < section_end
        ],
    }
