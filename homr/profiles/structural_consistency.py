from __future__ import annotations

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol


def _measure_groups(voice: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
    measures: list[list[EncodedSymbol]] = []
    current: list[EncodedSymbol] = []
    for symbol in voice:
        if symbol.rhythm == "newline":
            continue
        current.append(symbol)
        if "barline" in symbol.rhythm or "repeat" in symbol.rhythm:
            measures.append(current)
            current = []
    if current:
        measures.append(current)
    return measures


def validate_structural_visual_consistency(
    staffs: list[list[EncodedSymbol]], profile_name: str
) -> int:
    """Flag sequence-level recognitions unsupported by independent visual evidence.

    This validator never rewrites music. A measure is considered a high-confidence
    structural mismatch only when three independent warning signs coincide:

    * TrOMR emitted pitched note events.
    * The segmentation notehead mask contributes no component in the measure/staff zone.
    * Both rhythm and pitch branches are low-confidence at the measure level.

    The conjunction is intentionally conservative because hollow handwritten noteheads are
    sometimes missed by segmentation even in otherwise-correct measures.
    """
    mismatches = 0
    for part_index, voice in enumerate(staffs, 1):
        for measure_index, measure in enumerate(_measure_groups(voice), 1):
            notes = [symbol for symbol in measure if symbol.rhythm.startswith("note")]
            if not notes:
                continue
            evidence = next(
                (symbol for symbol in measure if symbol.visual_measure_notehead_count >= 0),
                None,
            )
            if evidence is None:
                continue
            notehead_count = evidence.visual_measure_notehead_count
            rhythm_median = evidence.visual_measure_rhythm_confidence_median
            pitch_median = evidence.visual_measure_pitch_confidence_median
            if not (
                notehead_count == 0
                and rhythm_median < 0.55
                and pitch_median < 0.60
            ):
                continue

            mismatches += 1
            for symbol in measure:
                symbol.structural_visual_mismatch = True
            eprint(
                f"PROFILE {profile_name} P{part_index} m{measure_index} "
                "STRUCTURAL_VISUAL_MISMATCH: "
                f"transformer_notes={len(notes)}, notehead_components={notehead_count}, "
                f"rhythm_median={rhythm_median:.3f}, pitch_median={pitch_median:.3f}; "
                "recognition left unchanged"
            )
    eprint(f"PROFILE {profile_name}: structural visual mismatches={mismatches}")
    return mismatches

