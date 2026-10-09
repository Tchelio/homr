import xml.etree.ElementTree as ET

from homr.profiles.dorico_tremolo import normalize_dorico_reference_tremolos
from homr.profiles.expected_meter import validate_expected_meter
from homr.profiles.measure_constraint_solver import reconcile_unique_measure_constraints
from homr.profiles.pitch_reconciliation import reconcile_repeated_tremolo_pitches
from homr.profiles.slur_suppression import (
    suppress_remaining_slurs,
    suppress_ties_into_double_tremolos,
)
from homr.transformer.vocabulary import EncodedSymbol


PROFILE_NAME = "2610-ver.6"


def apply_symbols(staffs: list[list[EncodedSymbol]]) -> None:
    # Deliberately return to v4's symbol policy. v6 tests only a Dorico-import compatibility
    # hypothesis: keep ties, but suppress graphical slurs that remain after tie conversion.
    reconcile_repeated_tremolo_pitches(staffs, profile_name=PROFILE_NAME)
    reconcile_unique_measure_constraints(staffs, beats=4, beat_type=4, profile_name=PROFILE_NAME)


def apply(root: ET.Element) -> None:
    validate_expected_meter(root, beats=4, beat_type=4, profile_name=PROFILE_NAME)
    normalize_dorico_reference_tremolos(root, profile_name=PROFILE_NAME)
    suppress_remaining_slurs(root, profile_name=PROFILE_NAME)
    suppress_ties_into_double_tremolos(root, profile_name=PROFILE_NAME)
