import xml.etree.ElementTree as ET

from homr.profiles.expected_meter import validate_expected_meter
from homr.profiles.tremolo_reconciliation import reconcile_anchored_double_tremolos


PROFILE_NAME = "2610-ver.1"


def apply(root: ET.Element, profile_name: str = PROFILE_NAME) -> None:
    # Score invariant supplied by the user: this work runs entirely in 4/4.
    # For now 4/4 is a constraint/validator only; we deliberately leave homr's emitted
    # time signature untouched until the recognized rhythm structure is internally consistent.
    reconcile_anchored_double_tremolos(
        root, beats=4, beat_type=4, profile_name=profile_name
    )
    validate_expected_meter(root, beats=4, beat_type=4, profile_name=profile_name)
