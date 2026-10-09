import xml.etree.ElementTree as ET

from homr.profiles.dorico_tremolo import normalize_dorico_reference_tremolos
from homr.profiles.pitch_reconciliation import reconcile_repeated_tremolo_pitches
from homr.profiles.profile_2610_v1 import apply as apply_v1
from homr.transformer.vocabulary import EncodedSymbol


PROFILE_NAME = "2610-ver.3"


def apply_symbols(staffs: list[list[EncodedSymbol]]) -> None:
    # Keep v2's conservative top-k pitch reconciliation.
    reconcile_repeated_tremolo_pitches(staffs, profile_name=PROFILE_NAME)


def apply(root: ET.Element) -> None:
    # Keep v1's conservative rhythm reconciliation and expected-meter diagnostics.
    apply_v1(root, profile_name=PROFILE_NAME)
    # Replace v2's broad beam hint with the structure observed in Dorico's own export.
    normalize_dorico_reference_tremolos(root, profile_name=PROFILE_NAME)
