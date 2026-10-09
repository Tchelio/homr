import xml.etree.ElementTree as ET

from homr.profiles.barline_consensus import reconcile_system_barlines
from homr.profiles.dorico_tremolo import normalize_dorico_reference_tremolos
from homr.profiles.expected_meter import validate_expected_meter
from homr.profiles.measure_constraint_solver import reconcile_unique_measure_constraints
from homr.profiles.pitch_reconciliation import reconcile_repeated_tremolo_pitches
from homr.profiles.structural_consistency import validate_structural_visual_consistency
from homr.transformer.vocabulary import EncodedSymbol


PROFILE_NAME = "2610-ver.8"


def apply_symbols(staffs: list[list[EncodedSymbol]]) -> None:
    # Run the diagnostic on the transformer's original topology. Measure-level visual
    # metadata was collected before consensus, so validating it after removing/inserting
    # barlines would attach stale per-measure counts to the new boundaries.
    validate_structural_visual_consistency(staffs, profile_name=PROFILE_NAME)
    reconcile_system_barlines(staffs, profile_name=PROFILE_NAME)
    reconcile_repeated_tremolo_pitches(staffs, profile_name=PROFILE_NAME)
    reconcile_unique_measure_constraints(
        staffs,
        beats=4,
        beat_type=4,
        profile_name=PROFILE_NAME,
        allow_dotted_tremolo=True,
        allow_rest_rhythm_alternatives=True,
        prefer_evidence_when_ambiguous=True,
    )


def apply(root: ET.Element) -> None:
    validate_expected_meter(root, beats=4, beat_type=4, profile_name=PROFILE_NAME)
    normalize_dorico_reference_tremolos(root, profile_name=PROFILE_NAME)
