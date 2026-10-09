import xml.etree.ElementTree as ET

from homr.profiles.dorico_tremolo import normalize_dorico_reference_tremolos
from homr.profiles.expected_meter import validate_expected_meter
from homr.profiles.measure_constraint_solver import reconcile_unique_measure_constraints
from homr.profiles.pitch_reconciliation import reconcile_repeated_tremolo_pitches
from homr.transformer.vocabulary import EncodedSymbol


PROFILE_NAME = "2610-ver.5"


def apply_symbols(staffs: list[list[EncodedSymbol]]) -> None:
    # Same generic reconciliation architecture as v4. The solver now also removes a
    # pair-local slur only when a tremolo was recovered from independent visual evidence.
    reconcile_repeated_tremolo_pitches(staffs, profile_name=PROFILE_NAME)
    reconcile_unique_measure_constraints(
        staffs,
        beats=4,
        beat_type=4,
        profile_name=PROFILE_NAME,
        remove_visual_pair_slurs=True,
    )


def apply(root: ET.Element) -> None:
    validate_expected_meter(root, beats=4, beat_type=4, profile_name=PROFILE_NAME)
    normalize_dorico_reference_tremolos(root, profile_name=PROFILE_NAME)
