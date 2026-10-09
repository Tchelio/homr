from __future__ import annotations

import re
from fractions import Fraction

from homr.profiles.measure_constraint_solver import (
    _events,
    _explicit_double_pairs,
    _measure_groups,
    _rhythm_candidate_probability,
    _shorter_rhythm_token,
)
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol


_STANDARD_RESTS = ("rest_16", "rest_8", "rest_4", "rest_2", "rest_1")
_STANDARD_NOTE = re.compile(r"^note_(1|2|4|8|16|32|64|128)(\.{0,2})$")


def _rest_for_exact_deficit(deficit: Fraction) -> str | None:
    matches = [
        rhythm
        for rhythm in _STANDARD_RESTS
        if EncodedSymbol(rhythm, "_").get_duration().fraction == deficit
    ]
    return matches[0] if len(matches) == 1 else None


def recover_missing_rests_between_tremolo_pairs(
    staffs: list[list[EncodedSymbol]],
    beats: int,
    beat_type: int,
    profile_name: str,
    *,
    minimum_visual_score: float = 0.45,
) -> int:
    """Recover one omitted rest only when strong independent constraints agree.

    The reusable failure class is a small symbol dropped by segmentation between two
    explicit double-tremolo pairs.  We preserve every decoded tremolo pair, compute the
    remaining meter deficit, require that deficit to equal exactly one standard rest, and
    then require a compact raw-page residual in an otherwise empty inter-pair gap.  Any
    ambiguity fails closed.
    """
    target = Fraction(beats, beat_type)
    recovered = 0

    for part_index, voice in enumerate(staffs, 1):
        for measure_index, measure in enumerate(_measure_groups(voice), 1):
            if any(symbol.rhythm == "chord" for symbol in measure):
                continue
            events = _events(measure)
            pairs = _explicit_double_pairs(events, allow_dotted_tremolo=True)
            if len(pairs) < 2:
                continue

            paired_ids = {id(symbol) for pair in pairs for symbol in pair}
            effective = sum(
                (
                    symbol.get_duration().fraction
                    * (Fraction(1, 2) if id(symbol) in paired_ids else Fraction(1))
                    for symbol in events
                ),
                Fraction(0),
            )
            deficit = target - effective
            if deficit <= 0:
                continue
            rest_rhythm = _rest_for_exact_deficit(deficit)
            if rest_rhythm is None:
                continue

            event_index = {id(symbol): index for index, symbol in enumerate(events)}
            ordered_pairs = sorted(pairs, key=lambda pair: event_index[id(pair[0])])
            candidates: list[tuple[float, float, EncodedSymbol, EncodedSymbol]] = []
            for left_pair, right_pair in zip(ordered_pairs, ordered_pairs[1:]):
                left_end = left_pair[1]
                right_start = right_pair[0]
                left_index = event_index[id(left_end)]
                right_index = event_index[id(right_start)]
                if right_index != left_index + 1:
                    continue
                if (
                    left_end.visual_gap_after_page_x is None
                    or left_end.visual_gap_after_score < minimum_visual_score
                ):
                    continue
                candidates.append(
                    (
                        left_end.visual_gap_after_score,
                        left_end.visual_gap_after_page_x,
                        left_end,
                        right_start,
                    )
                )

            if len(candidates) != 1:
                continue
            score, page_x, _left_end, right_start = candidates[0]

            insert_at = next(
                (index for index, symbol in enumerate(voice) if symbol is right_start),
                None,
            )
            if insert_at is None:
                continue
            replacement = EncodedSymbol(rest_rhythm, "_")
            replacement.page_coordinates = (page_x, 0.0)
            replacement.missing_rest_recovery = True
            voice.insert(insert_at, replacement)
            recovered += 1
            eprint(
                f"PROFILE {profile_name} P{part_index} m{measure_index} "
                "MISSING_REST_RECOVERY: "
                f"{rest_rhythm}, visual={score:.3f}, x={page_x:.1f}"
            )

    eprint(f"PROFILE {profile_name}: missing rest recoveries={recovered}")
    return recovered


def recover_underfull_unique_visual_gap_rest(
    staffs: list[list[EncodedSymbol]],
    beats: int,
    beat_type: int,
    profile_name: str,
    *,
    minimum_visual_score: float = 0.45,
) -> int:
    """Insert one standard rest into a unique visual rest->note gap of an underfull measure."""
    target = Fraction(beats, beat_type)
    recovered = 0
    for part_index, voice in enumerate(staffs, 1):
        for measure_index, measure in enumerate(_measure_groups(voice), 1):
            events = _events(measure)
            effective = sum(
                (symbol.get_effective_duration_fraction() for symbol in events),
                Fraction(0),
            )
            deficit = target - effective
            if deficit <= 0:
                continue
            rest_rhythm = _rest_for_exact_deficit(deficit)
            if rest_rhythm is None:
                continue

            candidates: list[tuple[EncodedSymbol, EncodedSymbol]] = []
            for left, right in zip(events, events[1:]):
                if not left.rhythm.startswith("rest") or not right.rhythm.startswith("note"):
                    continue
                if (
                    left.visual_gap_after_page_x is None
                    or left.visual_gap_after_score < minimum_visual_score
                ):
                    continue
                candidates.append((left, right))
            if len(candidates) != 1:
                continue

            left, right = candidates[0]
            insert_at = next(
                (index for index, symbol in enumerate(voice) if symbol is right),
                None,
            )
            if insert_at is None:
                continue
            replacement = EncodedSymbol(rest_rhythm, "_")
            replacement.page_coordinates = (left.visual_gap_after_page_x, 0.0)
            replacement.missing_rest_recovery = True
            voice.insert(insert_at, replacement)
            recovered += 1
            eprint(
                f"PROFILE {profile_name} P{part_index} m{measure_index} "
                "UNDERFULL_VISUAL_GAP_REST_RECOVERY: "
                f"{rest_rhythm}, visual={left.visual_gap_after_score:.3f}, "
                f"x={left.visual_gap_after_page_x:.1f}"
            )

    eprint(f"PROFILE {profile_name}: underfull visual-gap rest recoveries={recovered}")
    return recovered


def recover_exact_fit_masked_rests(
    staffs: list[list[EncodedSymbol]],
    beats: int,
    beat_type: int,
    profile_name: str,
    *,
    minimum_visual_ink_score: float = 0.45,
    minimum_shorter_probability: float = 0.15,
    minimum_relative_probability: float = 0.35,
    allow_after_note_gap: bool = False,
    allow_explicit_tremolo_overage: bool = False,
    allow_deeper_note_alternatives: bool = False,
    allow_visual_uncertain_eighth: bool = False,
) -> int:
    """Recover a missing rest masked by an overlong note-duration interpretation.

    This reuses the same raw-gap evidence as the tremolo-gap rest recovery.  Explicit
    double-tremolo pairs are provisionally evaluated with their established 2:1 timing, then
    a shorter decoder-backed note value may create room for exactly one standard rest.  The
    combined edit must land exactly on the expected meter and a measure must have exactly one
    surviving proposal.
    """
    target = Fraction(beats, beat_type)
    recovered = 0

    for part_index, voice in enumerate(staffs, 1):
        for measure_index, measure in enumerate(_measure_groups(voice), 1):
            if any(symbol.rhythm == "chord" for symbol in measure):
                continue
            events = _events(measure)
            if allow_explicit_tremolo_overage:
                explicit_pairs = _explicit_double_pairs(events, allow_dotted_tremolo=True)
                paired_ids = {id(symbol) for pair in explicit_pairs for symbol in pair}
                effective = sum(
                    (
                        symbol.get_duration().fraction
                        * (Fraction(1, 2) if id(symbol) in paired_ids else Fraction(1))
                        for symbol in events
                    ),
                    Fraction(0),
                )
                if effective < target:
                    continue
                overage = effective - target
            else:
                effective = sum(
                    (symbol.get_effective_duration_fraction() for symbol in events),
                    Fraction(0),
                )
                if effective != target:
                    continue
                overage = Fraction(0)

            candidates: list[
                tuple[EncodedSymbol, EncodedSymbol, str, str, float, float, float, int]
            ] = []
            for note_index, note in enumerate(events):
                if not note.rhythm.startswith("note") or not note.rhythm_candidates:
                    continue
                if note.rhythm_candidates[0][0] != note.rhythm:
                    continue
                current_duration = note.get_duration().fraction
                top_probability = note.rhythm_candidates[0][1]
                if allow_deeper_note_alternatives:
                    shorter_candidates = [
                        (rhythm, probability)
                        for rhythm, probability in note.rhythm_candidates[1:]
                        if _STANDARD_NOTE.match(rhythm)
                    ]
                else:
                    shorter = _shorter_rhythm_token(note.rhythm)
                    shorter_candidates = (
                        [(shorter, _rhythm_candidate_probability(note, shorter))]
                        if shorter is not None
                        else []
                    )
                for shorter, alternative_probability in shorter_candidates:
                    old_rhythm = note.rhythm
                    note.rhythm = shorter
                    note._duration = None
                    shorter_duration = note.get_duration().fraction
                    note.rhythm = old_rhythm
                    note._duration = None
                    reduction = current_duration - shorter_duration
                    if reduction <= 0:
                        continue

                    relative_probability = alternative_probability / max(top_probability, 1e-9)
                    strong_decoder_support = (
                        alternative_probability >= minimum_shorter_probability
                        and relative_probability >= minimum_relative_probability
                    )
                    # Reuse ver.12's visual-uncertainty gate for the shorthand-eighth failure
                    # class.  The image evidence does not say "eighth" by itself; it merely
                    # lets a weak note_8 decoder candidate survive when an independent raw-gap
                    # residual and exact-meter reconciliation agree.
                    visual_uncertain_eighth = (
                        allow_visual_uncertain_eighth
                        and
                        shorter == "note_8"
                        and alternative_probability > 0
                        and note.visual_notehead_solid_score <= 0.05
                    )
                    missing_duration = reduction - overage
                    if missing_duration <= 0:
                        continue
                    rest_rhythm = _rest_for_exact_deficit(missing_duration)
                    if rest_rhythm is None:
                        continue

                    gap_candidates: list[tuple[EncodedSymbol, int]] = []
                    has_strong_before_gap = False
                    if (
                        strong_decoder_support
                        and note_index > 0
                        and events[note_index - 1].rhythm.startswith("rest")
                    ):
                        previous = events[note_index - 1]
                        if (
                            previous.visual_gap_after_page_x is not None
                            and previous.visual_gap_after_score >= minimum_visual_ink_score
                        ):
                            gap_candidates.append((previous, 0))
                            has_strong_before_gap = True
                    if (
                        allow_after_note_gap
                        and visual_uncertain_eighth
                        and not has_strong_before_gap
                    ):
                        next_is_rest_or_boundary = (
                            note_index + 1 == len(events)
                            or events[note_index + 1].rhythm.startswith("rest")
                        )
                        if (
                            next_is_rest_or_boundary
                            and note.visual_gap_after_page_x is not None
                            and note.visual_gap_after_score >= minimum_visual_ink_score
                        ):
                            gap_candidates.append((note, 1))

                    for gap_owner, insert_after in gap_candidates:
                        candidates.append(
                            (
                                note,
                                gap_owner,
                                shorter,
                                rest_rhythm,
                                alternative_probability,
                                relative_probability,
                                gap_owner.visual_gap_after_score,
                                insert_after,
                            )
                        )

            if len(candidates) != 1:
                continue
            note, gap_owner, shorter, rest_rhythm, alt, ratio, visual, insert_after = candidates[0]
            page_x = gap_owner.visual_gap_after_page_x
            if page_x is None:
                continue
            note_at = next(
                (index for index, symbol in enumerate(voice) if symbol is note),
                None,
            )
            if note_at is None:
                continue
            insert_at = note_at + 1 if insert_after else note_at

            original_rhythm = note.rhythm
            note.rhythm = shorter
            note._duration = None
            replacement = EncodedSymbol(rest_rhythm, "_")
            replacement.page_coordinates = (page_x, 0.0)
            replacement.missing_rest_recovery = True
            voice.insert(insert_at, replacement)
            recovered += 1
            eprint(
                f"PROFILE {profile_name} P{part_index} m{measure_index} "
                "EXACT_FIT_MASKED_REST_RECOVERY: "
                f"{rest_rhythm} + {original_rhythm}->{shorter}, visual={visual:.3f}, "
                f"alt={alt:.3f}, ratio={ratio:.3f}"
            )

    eprint(f"PROFILE {profile_name}: exact-fit masked rest recoveries={recovered}")
    return recovered
