import xml.etree.ElementTree as ET

from homr.profiles.dorico_tremolo import add_dorico_double_tremolo_beam_hints
from homr.profiles.pitch_reconciliation import reconcile_repeated_tremolo_pitches
from homr.profiles.profile_2610_v1 import apply as apply_v1
from homr.transformer.vocabulary import EncodedSymbol


PROFILE_NAME = "2610-ver.2"


def apply_symbols(staffs: list[list[EncodedSymbol]]) -> None:
    reconcile_repeated_tremolo_pitches(staffs, profile_name=PROFILE_NAME)


def apply(root: ET.Element) -> None:
    # Keep all conservative rhythm reconciliation / expected-meter validation from v1.
    apply_v1(root, profile_name=PROFILE_NAME)
    # Dorico compatibility is an output concern, so keep it isolated from recognition.
    add_dorico_double_tremolo_beam_hints(root, profile_name=PROFILE_NAME)
