from __future__ import annotations

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol


def _has_tremolo(symbol: EncodedSymbol) -> bool:
    return "tremolo" in symbol.articulation.split("_")


def _pitch_rank(pitch: str) -> int | None:
    if len(pitch) < 2 or pitch[0] not in "CDEFGAB":
        return None
    try:
        octave = int(pitch[1:])
    except ValueError:
        return None
    return octave * 7 + "CDEFGAB".index(pitch[0])


def _vertical_direction(pair: tuple[EncodedSymbol, EncodedSymbol]) -> int:
    first, second = pair
    if first.image_coordinates is None or second.image_coordinates is None:
        return 0
    delta = first.image_coordinates[1] - second.image_coordinates[1]
    if abs(delta) < 3.0:
        return 0
    return 1 if delta > 0 else -1


def _pitch_direction(pitches: tuple[str, str]) -> int:
    first = _pitch_rank(pitches[0])
    second = _pitch_rank(pitches[1])
    if first is None or second is None or first == second:
        return 0
    # Image y grows downward, so a lower first pitch should have a larger y.
    return 1 if first < second else -1


def _measure_note_groups(voice: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
    measures: list[list[EncodedSymbol]] = []
    current: list[EncodedSymbol] = []
    for symbol in voice:
        if symbol.rhythm.startswith("note") or symbol.rhythm.startswith("rest"):
            current.append(symbol)
        if "barline" in symbol.rhythm or "repeat" in symbol.rhythm:
            measures.append(current)
            current = []
    if current:
        measures.append(current)
    return measures


def _tremolo_pairs(measure: list[EncodedSymbol]) -> list[tuple[EncodedSymbol, EncodedSymbol]]:
    tremolo_notes = [s for s in measure if s.rhythm.startswith("note") and _has_tremolo(s)]
    return [
        (tremolo_notes[index], tremolo_notes[index + 1])
        for index in range(0, len(tremolo_notes) - 1, 2)
    ]


def _top_probability(symbol: EncodedSymbol) -> float:
    if not symbol.pitch_candidates:
        return 0.0
    pitch, probability = symbol.pitch_candidates[0]
    return probability if pitch == symbol.pitch else 0.0


def _alternative_candidates(symbol: EncodedSymbol) -> list[tuple[str, float]]:
    return [
        (pitch, probability)
        for pitch, probability in symbol.pitch_candidates[1:]
        if pitch != symbol.pitch and pitch not in ("_", ".")
    ]


def reconcile_repeated_tremolo_pitches(
    staffs: list[list[EncodedSymbol]],
    profile_name: str,
    *,
    allow_structural_recovery: bool = False,
    allow_recovered_geometry_fallback: bool = False,
) -> int:
    """
    Reconsider a same-pitch tremolo pair only when independent evidence agrees.

    The correction is intentionally conservative. A model winner is changed only when:
    - the pair was predicted as the same pitch on both notes;
    - exactly one member is uncertain enough to have a substantial alternative pitch;
    - the other member is a high-confidence pitch anchor;
    - another tremolo pair in the same measure is confidently recognized with exactly the
      proposed ordered pitch pattern; and
    - both pairs' vertical attention ordering agrees with that pitch ordering.

    This lets a profile recover information discarded by greedy argmax without encoding a
    score-specific pitch such as "P4 m1 must be E-G".
    """
    corrected = 0

    for part_index, voice in enumerate(staffs, 1):
        for measure_index, measure in enumerate(_measure_note_groups(voice), 1):
            recovered_measure = allow_structural_recovery and any(
                symbol.structural_recovery or symbol.missing_rest_recovery
                for symbol in measure
            )
            rest_count = sum(1 for symbol in measure if symbol.rhythm.startswith("rest"))
            pitched_notes = [symbol for symbol in measure if symbol.rhythm.startswith("note")]
            if rest_count < 2 or not pitched_notes:
                continue
            # At symbol stage ties are still represented by the model's slur vocabulary.
            # Require the right-edge continuation anchor as an additional fail-closed signal;
            # convert_ties() will later decide whether it is truly a tie across the barline.
            if (
                not recovered_measure
                and "slurStart" not in pitched_notes[-1].slur.split("_")
            ):
                continue

            pairs = _tremolo_pairs(measure)
            if len(pairs) < 2 and not (
                allow_recovered_geometry_fallback
                and recovered_measure
                and len(pairs) == 1
            ):
                continue

            reference_patterns: set[tuple[str, str]] = set()
            for pair in pairs:
                pitches = (pair[0].pitch, pair[1].pitch)
                if pitches[0] == pitches[1]:
                    continue
                first_probability = _top_probability(pair[0])
                second_probability = _top_probability(pair[1])
                if recovered_measure:
                    if min(first_probability, second_probability) < 0.70:
                        continue
                    if max(first_probability, second_probability) < 0.95:
                        continue
                elif min(first_probability, second_probability) < 0.9:
                    continue
                visual_direction = _vertical_direction(pair)
                if visual_direction == 0 or visual_direction != _pitch_direction(pitches):
                    continue
                reference_patterns.add(pitches)

            proposals: list[tuple[EncodedSymbol, str, float, tuple[str, str]]] = []
            for pair in pairs:
                if pair[0].pitch != pair[1].pitch:
                    continue
                visual_direction = _vertical_direction(pair)
                if visual_direction == 0 and not recovered_measure:
                    continue

                for uncertain_index in (0, 1):
                    anchor_index = 1 - uncertain_index
                    uncertain = pair[uncertain_index]
                    anchor = pair[anchor_index]
                    if _top_probability(uncertain) > 0.85:
                        continue
                    if _top_probability(anchor) < 0.95:
                        continue

                    for alternative, probability in _alternative_candidates(uncertain):
                        if probability < 0.15:
                            continue
                        proposed = [pair[0].pitch, pair[1].pitch]
                        proposed[uncertain_index] = alternative
                        pattern = (proposed[0], proposed[1])
                        if reference_patterns:
                            if pattern not in reference_patterns:
                                continue
                            if (
                                not recovered_measure
                                and visual_direction != _pitch_direction(pattern)
                            ):
                                continue
                        else:
                            if not (
                                allow_recovered_geometry_fallback
                                and recovered_measure
                                and visual_direction != 0
                                and visual_direction == _pitch_direction(pattern)
                            ):
                                continue
                        proposals.append((uncertain, alternative, probability, pattern))

            # Fail closed if more than one interpretation survives the constraints.
            unique = {(id(symbol), pitch) for symbol, pitch, _, _ in proposals}
            if len(unique) != 1:
                continue

            symbol, alternative, probability, pattern = proposals[0]
            original = symbol.pitch
            symbol.pitch = alternative
            corrected += 1
            eprint(
                f"PROFILE {profile_name} P{part_index} m{measure_index} "
                "UNIQUE_TREMOLO_PITCH_RECONCILIATION: "
                f"{original}->{alternative} (alt={probability:.3f}, pattern={pattern[0]}-{pattern[1]})"
            )

    eprint(f"PROFILE {profile_name}: unique tremolo pitch corrections={corrected}")
    return corrected
