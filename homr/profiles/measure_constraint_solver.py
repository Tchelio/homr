from __future__ import annotations

import re
import math
from dataclasses import dataclass
from fractions import Fraction

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, empty


_SIMPLE_NOTE = re.compile(r"^note_(1|2|4|8|16|32|64|128)$")
_DOTTED_NOTE = re.compile(r"^note_(1|2|4|8|16|32|64|128)(\.{1,2})?$")
_SIMPLE_REST = re.compile(r"^rest_(1|2|4|8|16|32|64|128)(\.{1,2})?$")
_NEXT_SHORTER = {
    "note_1": "note_2",
    "note_2": "note_4",
    "note_4": "note_8",
    "note_8": "note_16",
    "note_16": "note_32",
    "note_32": "note_64",
    "note_64": "note_128",
}


def _shorter_rhythm_token(token: str) -> str | None:
    match = re.match(r"^(note|rest)_(1|2|4|8|16|32|64)(\.{0,2})$", token)
    if match is None:
        return None
    kind, denominator, dots = match.groups()
    return f"{kind}_{int(denominator) * 2}{dots}"


def _rhythm_candidate_probability(symbol: EncodedSymbol, target: str) -> float:
    for token, probability in symbol.rhythm_candidates:
        if token == target:
            return probability
    return 0.0


def _has_tremolo(symbol: EncodedSymbol) -> bool:
    return "tremolo" in symbol.articulation.split("_")


def _add_tremolo(symbol: EncodedSymbol) -> None:
    articulations = [
        item for item in symbol.articulation.split("_") if item not in ("", empty, ".")
    ]
    if "tremolo" not in articulations:
        articulations.append("tremolo")
    symbol.articulation = "_".join(sorted(articulations)) if articulations else empty


def _remove_tremolo(symbol: EncodedSymbol) -> None:
    articulations = [
        item
        for item in symbol.articulation.split("_")
        if item not in ("", empty, ".", "tremolo")
    ]
    symbol.articulation = "_".join(sorted(articulations)) if articulations else empty


def _remove_slur_token(symbol: EncodedSymbol, token: str) -> None:
    slurs = [
        item
        for item in symbol.slur.split("_")
        if item not in ("", empty, ".") and item != token
    ]
    symbol.slur = "_".join(sorted(slurs)) if slurs else empty


@dataclass
class _Action:
    name: str
    covered: frozenset[int]
    reduction: Fraction
    pair: tuple[EncodedSymbol, EncodedSymbol] | None = None
    shorter_pair: bool = False
    single: EncodedSymbol | None = None
    replacement_rhythm: str | None = None
    evidence_score: float = 0.0


def _measure_groups(voice: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
    measures: list[list[EncodedSymbol]] = []
    current: list[EncodedSymbol] = []
    for symbol in voice:
        current.append(symbol)
        if "barline" in symbol.rhythm or "repeat" in symbol.rhythm:
            measures.append(current)
            current = []
    if current:
        measures.append(current)
    return measures


def _events(measure: list[EncodedSymbol]) -> list[EncodedSymbol]:
    return [s for s in measure if s.rhythm.startswith(("note", "rest"))]


def _explicit_double_pairs(
    events: list[EncodedSymbol], allow_dotted_tremolo: bool
) -> list[tuple[EncodedSymbol, EncodedSymbol]]:
    tremolo_notes = [s for s in events if s.rhythm.startswith("note") and _has_tremolo(s)]
    pairs: list[tuple[EncodedSymbol, EncodedSymbol]] = []
    for index in range(0, len(tremolo_notes) - 1, 2):
        first = tremolo_notes[index]
        second = tremolo_notes[index + 1]
        try:
            first_event = events.index(first)
            second_event = events.index(second)
        except ValueError:
            continue
        if second_event != first_event + 1:
            continue
        if first.get_duration().fraction != second.get_duration().fraction:
            continue
        pattern = _DOTTED_NOTE if allow_dotted_tremolo else _SIMPLE_NOTE
        if not pattern.match(first.rhythm) or first.rhythm != second.rhythm:
            continue
        pairs.append((first, second))
    return pairs


def _visual_double_pairs(
    events: list[EncodedSymbol],
    explicit_pairs: list[tuple[EncodedSymbol, EncodedSymbol]],
    allow_repeated_pattern_medium_evidence: bool,
) -> list[tuple[EncodedSymbol, EncodedSymbol]]:
    result: list[tuple[EncodedSymbol, EncodedSymbol]] = []
    explicit_patterns = {(first.pitch, second.pitch) for first, second in explicit_pairs}
    for index in range(len(events) - 1):
        first = events[index]
        second = events[index + 1]
        if not first.rhythm.startswith("note") or not second.rhythm.startswith("note"):
            continue
        bars = first.visual_tremolo_with_next_bars
        strong_visual_evidence = bars >= 5
        repeated_pattern_medium_evidence = (
            allow_repeated_pattern_medium_evidence
            and bars >= 4
            and (first.pitch, second.pitch) in explicit_patterns
        )
        # Five repeated short bars remains the standalone conservative threshold. Four bars
        # may be admitted only as medium evidence when the same ordered pitch pair is already
        # present as an explicit tremolo in this measure. The meter solver still has to make
        # the combined interpretation uniquely actionable.
        if not strong_visual_evidence and not repeated_pattern_medium_evidence:
            continue
        if first.get_duration().fraction != second.get_duration().fraction:
            continue
        if not _SIMPLE_NOTE.match(first.rhythm) or first.rhythm != second.rhythm:
            continue
        result.append((first, second))
    return result


def _candidate_actions(
    measure: list[EncodedSymbol],
    allow_dotted_tremolo: bool,
    allow_rest_rhythm_alternatives: bool,
    allow_repeated_pattern_medium_tremolo: bool,
    allow_visual_uncertain_quarter_to_eighth: bool,
    allow_deeper_rest_rhythm_alternatives: bool,
) -> list[_Action]:
    events = _events(measure)
    actions: list[_Action] = []
    explicit = _explicit_double_pairs(events, allow_dotted_tremolo)
    explicit_ids = {frozenset((id(a), id(b))) for a, b in explicit}

    for first, second in explicit:
        duration = first.get_duration().fraction
        covered = frozenset((id(first), id(second)))
        actions.append(_Action("model-tremolo", covered, duration, (first, second)))
        shorter = _shorter_rhythm_token(first.rhythm)
        if not allow_dotted_tremolo:
            shorter = _NEXT_SHORTER.get(first.rhythm)
        if shorter is not None:
            first_probability = _rhythm_candidate_probability(first, shorter)
            second_probability = _rhythm_candidate_probability(second, shorter)
            # A shorter written value is an uncertainty-driven change rather than a
            # consequence of tremolo semantics itself. Keep the candidate even when the
            # decoder assigned it very little probability, because a unique exact-fit
            # solution may still justify it (as on the accepted m1). When alternatives
            # compete, these probabilities let the solver prefer the better-supported one.
            evidence_score = math.log(max(first_probability, 1e-9)) + math.log(
                max(second_probability, 1e-9)
            )
            actions.append(
                _Action(
                    "model-tremolo-shorter",
                    covered,
                    duration * Fraction(3, 2),
                    (first, second),
                    shorter_pair=True,
                    replacement_rhythm=shorter,
                    evidence_score=evidence_score,
                )
            )

    for first, second in _visual_double_pairs(
        events,
        explicit,
        allow_repeated_pattern_medium_evidence=allow_repeated_pattern_medium_tremolo,
    ):
        covered = frozenset((id(first), id(second)))
        if covered in explicit_ids:
            continue
        duration = first.get_duration().fraction
        # Visual evidence establishes presence of a pair, not a different written note type.
        actions.append(_Action("visual-tremolo", covered, duration, (first, second)))

    for symbol in events:
        if not re.match(r"^note_2(\.{0,2})$", symbol.rhythm):
            continue
        if symbol.visual_notehead_solid_score < 0.6:
            continue
        replacement_rhythm = _shorter_rhythm_token(symbol.rhythm)
        if replacement_rhythm is None:
            continue
        current_duration = symbol.get_duration().fraction
        old_rhythm = symbol.rhythm
        symbol.rhythm = replacement_rhythm
        symbol._duration = None
        replacement_duration = symbol.get_duration().fraction
        symbol.rhythm = old_rhythm
        symbol._duration = None
        actions.append(
            _Action(
                "solid-half-to-quarter",
                frozenset((id(symbol),)),
                current_duration - replacement_duration,
                single=symbol,
                replacement_rhythm=replacement_rhythm,
                evidence_score=math.log(max(symbol.visual_notehead_solid_score, 1e-9)),
            )
        )

    if allow_visual_uncertain_quarter_to_eighth:
        for symbol in events:
            if symbol.rhythm != "note_4" or _has_tremolo(symbol):
                continue
            # The independent notehead branch does not support the transformer's otherwise
            # confident quarter-note reading.  This is uncertainty evidence, not an eighth-
            # note classifier: exact meter must still need the change, and decoder top-k is
            # retained as the ranking evidence when multiple exact solutions compete.
            if symbol.visual_notehead_solid_score > 0.05:
                continue
            alternative_probability = _rhythm_candidate_probability(symbol, "note_8")
            if alternative_probability <= 0:
                continue
            actions.append(
                _Action(
                    "visual-uncertain-quarter-to-eighth",
                    frozenset((id(symbol),)),
                    Fraction(1, 8),
                    single=symbol,
                    replacement_rhythm="note_8",
                    evidence_score=math.log(max(alternative_probability, 1e-9)),
                )
            )

    if allow_rest_rhythm_alternatives:
        for symbol in events:
            if not _SIMPLE_REST.match(symbol.rhythm):
                continue
            shorter = _shorter_rhythm_token(symbol.rhythm)
            if shorter is None:
                continue
            current_duration = symbol.get_duration().fraction
            candidates = [(shorter, _rhythm_candidate_probability(symbol, shorter))]
            if allow_deeper_rest_rhythm_alternatives:
                candidates.extend(
                    (token, probability)
                    for token, probability in symbol.rhythm_candidates
                    if token != shorter and _SIMPLE_REST.match(token)
                )
            seen: set[str] = set()
            for candidate, alternative_probability in candidates:
                if candidate in seen or alternative_probability < 0.03:
                    continue
                seen.add(candidate)
                old_rhythm = symbol.rhythm
                symbol.rhythm = candidate
                symbol._duration = None
                candidate_duration = symbol.get_duration().fraction
                symbol.rhythm = old_rhythm
                symbol._duration = None
                reduction = current_duration - candidate_duration
                if reduction <= 0:
                    continue
                actions.append(
                    _Action(
                        "rest-rhythm-shorter",
                        frozenset((id(symbol),)),
                        reduction,
                        single=symbol,
                        replacement_rhythm=candidate,
                        evidence_score=math.log(alternative_probability),
                    )
                )
    return actions


def _find_solution(
    actions: list[_Action],
    target_reduction: Fraction,
    prefer_evidence_when_ambiguous: bool,
) -> list[_Action] | None:
    solutions: list[list[_Action]] = []

    def visit(index: int, used: frozenset[int], reduction: Fraction, chosen: list[_Action]) -> None:
        if len(solutions) >= 64 or reduction > target_reduction:
            return
        if reduction == target_reduction:
            solutions.append(chosen.copy())
            return
        if index >= len(actions):
            return
        for next_index in range(index, len(actions)):
            action = actions[next_index]
            if used.intersection(action.covered):
                continue
            visit(
                next_index + 1,
                used.union(action.covered),
                reduction + action.reduction,
                chosen + [action],
            )

    visit(0, frozenset(), Fraction(0), [])
    if len(solutions) == 1:
        return solutions[0]
    if len(solutions) == 0 or not prefer_evidence_when_ambiguous:
        return None

    ranked = sorted(
        ((sum(action.evidence_score for action in solution), solution) for solution in solutions),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best_solution = ranked[0]
    second_score = ranked[1][0]
    # Require at least a 4:1 likelihood margin before resolving an otherwise ambiguous
    # exact-meter solution. Otherwise preserve the ambiguity and make no edit.
    if best_score - second_score < math.log(4):
        return None
    return best_solution


def _apply_action(action: _Action, remove_visual_pair_slurs: bool) -> None:
    if action.single is not None:
        if action.replacement_rhythm is None:
            return
        action.single.rhythm = action.replacement_rhythm
        action.single._duration = None
        return
    if action.pair is None:
        return
    first, second = action.pair
    if action.shorter_pair:
        shorter = action.replacement_rhythm or _shorter_rhythm_token(first.rhythm)
        if shorter is None:
            return
        first.rhythm = shorter
        second.rhythm = shorter
        first._duration = None
        second._duration = None
    _add_tremolo(first)
    _add_tremolo(second)
    first.double_tremolo_role = "start"
    second.double_tremolo_role = "stop"
    source = "visual" if action.name == "visual-tremolo" else "model"
    first.double_tremolo_source = source
    second.double_tremolo_source = source
    first.duration_multiplier = Fraction(1, 2)
    second.duration_multiplier = Fraction(1, 2)

    # If TrOMR missed the tremolo and visual evidence recovered it, a pair-local slur
    # between two different pitches is a strong competing interpretation of the same
    # handwritten strokes. Remove only that local start/stop pair. A slurStop on the first
    # note (incoming same-pitch tie) and a slurStart continuing elsewhere are preserved.
    if (
        remove_visual_pair_slurs
        and source == "visual"
        and first.pitch != second.pitch
        and "slurStart" in first.slur.split("_")
        and "slurStop" in second.slur.split("_")
    ):
        _remove_slur_token(first, "slurStart")
        _remove_slur_token(second, "slurStop")


def reconcile_unique_measure_constraints(
    staffs: list[list[EncodedSymbol]],
    beats: int,
    beat_type: int,
    profile_name: str,
    *,
    allow_dotted_tremolo: bool = False,
    allow_rest_rhythm_alternatives: bool = False,
    prefer_evidence_when_ambiguous: bool = False,
    remove_visual_pair_slurs: bool = False,
    require_all_explicit_tremolo_pairs: bool = False,
    allow_repeated_pattern_medium_tremolo: bool = False,
    allow_visual_uncertain_quarter_to_eighth: bool = False,
    allow_deeper_rest_rhythm_alternatives: bool = False,
) -> int:
    """Resolve only uniquely-supported overfull measures from reusable evidence classes."""
    target = Fraction(beats, beat_type)
    corrected = 0
    for part_index, voice in enumerate(staffs, 1):
        for measure_index, measure in enumerate(_measure_groups(voice), 1):
            if any(symbol.rhythm == "chord" for symbol in measure):
                continue
            events = _events(measure)
            baseline = sum((s.get_effective_duration_fraction() for s in events), Fraction(0))
            if baseline <= target:
                continue
            actions = _candidate_actions(
                measure,
                allow_dotted_tremolo=allow_dotted_tremolo,
                allow_rest_rhythm_alternatives=allow_rest_rhythm_alternatives,
                allow_repeated_pattern_medium_tremolo=allow_repeated_pattern_medium_tremolo,
                allow_visual_uncertain_quarter_to_eighth=allow_visual_uncertain_quarter_to_eighth,
                allow_deeper_rest_rhythm_alternatives=allow_deeper_rest_rhythm_alternatives,
            )
            solution = _find_solution(
                actions,
                baseline - target,
                prefer_evidence_when_ambiguous=prefer_evidence_when_ambiguous,
            )
            if solution is None:
                continue

            if require_all_explicit_tremolo_pairs:
                explicit_pairs = _explicit_double_pairs(
                    events, allow_dotted_tremolo=allow_dotted_tremolo
                )
                selected_pair_ids = {
                    id(symbol)
                    for action in solution
                    if action.pair is not None
                    for symbol in action.pair
                }
                if any(
                    id(first) not in selected_pair_ids or id(second) not in selected_pair_ids
                    for first, second in explicit_pairs
                ):
                    # An exact meter fit is not enough evidence to erase a second explicit
                    # tremolo interpretation.  Leave the measure unresolved rather than
                    # silently stripping an unselected model pair after applying the chosen
                    # correction.  Older profiles keep their historical behavior unless
                    # they opt into this fail-closed guard.
                    eprint(
                        f"PROFILE {profile_name} P{part_index} m{measure_index} "
                        "EXPLICIT_TREMOLO_CONFLICT: exact-fit solution would discard "
                        "an unselected explicit tremolo pair; recognition left unchanged"
                    )
                    continue

            selected_pair_ids: set[int] = set()
            for action in solution:
                _apply_action(action, remove_visual_pair_slurs=remove_visual_pair_slurs)
                selected_pair_ids.update(action.covered if action.pair is not None else ())

            # Once an exact solution has established explicit double pairs, an unmatched
            # tremolo token in the same measure would only perturb start/stop state. Keep
            # unrelated articulations but fail closed on that unpaired tremolo label.
            if selected_pair_ids:
                for symbol in events:
                    if (
                        symbol.rhythm.startswith("note")
                        and id(symbol) not in selected_pair_ids
                        and _has_tremolo(symbol)
                    ):
                        _remove_tremolo(symbol)

            corrected += 1
            final_duration = sum(
                (s.get_effective_duration_fraction() for s in events), Fraction(0)
            )
            eprint(
                f"PROFILE {profile_name} P{part_index} m{measure_index} "
                "UNIQUE_MEASURE_RECONCILIATION: "
                f"{float(baseline * 4):g}->{float(final_duration * 4):g} quarter notes; "
                + ", ".join(action.name for action in solution)
            )
    eprint(f"PROFILE {profile_name}: unique measure corrections={corrected}")
    return corrected
