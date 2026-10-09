from __future__ import annotations

import statistics
from fractions import Fraction

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol


def _is_barline(symbol: EncodedSymbol) -> bool:
    return "barline" in symbol.rhythm or "repeat" in symbol.rhythm


def _boundary_x(symbol: EncodedSymbol) -> float | None:
    if symbol.barline_geometry_page_x is not None:
        return symbol.barline_geometry_page_x
    if symbol.page_coordinates is not None:
        return symbol.page_coordinates[0]
    return None


def _best_spatial_rest(
    occupied_fraction: float, beats: int, beat_type: int
) -> tuple[str, float] | None:
    target = Fraction(beats, beat_type)
    candidates: list[tuple[float, str]] = []
    for rhythm in ("rest_16", "rest_8", "rest_4", "rest_2", "rest_1"):
        symbol = EncodedSymbol(rhythm, "_")
        fraction = float(symbol.get_duration().fraction / target)
        candidates.append((abs(fraction - occupied_fraction), rhythm))
    candidates.sort()
    best_error, best_rhythm = candidates[0]
    second_error = candidates[1][0]
    if best_error > 0.12 or second_error - best_error < 0.05:
        return None
    return best_rhythm, best_error


def recover_suppressed_barline_prefixes(
    staffs: list[list[EncodedSymbol]], beats: int, beat_type: int, profile_name: str
) -> int:
    """Collapse an invalid pre-boundary hallucination into one spatially-supported rest.

    The trigger is deliberately narrow but reusable: a system consensus must have removed a
    barline, and the symbols immediately before that boundary must already have been flagged
    by the independent structural-visual validator. The deleted boundary's x-position then
    tells us how much horizontal measure space the suspect prefix occupied. We only emit a
    standard rest when one duration is clearly the closest spatial interpretation.
    """
    recovered = 0
    for part_index, voice in enumerate(staffs, 1):
        measure_start = 0
        left_boundary_x: float | None = None
        index = 0
        while index < len(voice):
            symbol = voice[index]
            if symbol.rhythm == "newline":
                measure_start = index + 1
                left_boundary_x = None
                index += 1
                continue
            if not _is_barline(symbol):
                index += 1
                continue

            right_boundary_x = _boundary_x(symbol)
            if left_boundary_x is not None and right_boundary_x is not None:
                event_indices = [
                    i
                    for i in range(measure_start, index)
                    if voice[i].rhythm.startswith(("note", "rest"))
                ]
                prefix: list[int] = []
                suppressed_xs: list[float] = []
                for event_index in event_indices:
                    event = voice[event_index]
                    if not event.structural_visual_mismatch:
                        break
                    if event.suppressed_barline_after_page_x is None:
                        break
                    prefix.append(event_index)
                    suppressed_xs.append(event.suppressed_barline_after_page_x)

                if prefix and len(prefix) < len(event_indices):
                    suppressed_x = statistics.median(suppressed_xs)
                    width = right_boundary_x - left_boundary_x
                    if width > 0 and left_boundary_x < suppressed_x < right_boundary_x:
                        occupied_fraction = (suppressed_x - left_boundary_x) / width
                        rest = _best_spatial_rest(occupied_fraction, beats, beat_type)
                        if rest is not None:
                            rest_rhythm, error = rest
                            insert_at = prefix[0]
                            for remove_index in reversed(prefix):
                                voice.pop(remove_index)
                                index -= 1
                            replacement = EncodedSymbol(rest_rhythm, "_")
                            replacement.structural_recovery = True
                            replacement.page_coordinates = (
                                (left_boundary_x + suppressed_x) / 2,
                                0.0,
                            )
                            voice.insert(insert_at, replacement)
                            index += 1
                            recovered += 1
                            eprint(
                                f"PROFILE {profile_name} P{part_index} "
                                "STRUCTURAL_PREFIX_RECOVERY: "
                                f"symbols={len(prefix)} -> {rest_rhythm}, "
                                f"span={occupied_fraction:.3f}, error={error:.3f}"
                            )

            left_boundary_x = right_boundary_x
            measure_start = index + 1
            index += 1

    eprint(f"PROFILE {profile_name}: structural prefix recoveries={recovered}")
    return recovered

