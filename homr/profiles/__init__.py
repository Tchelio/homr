from collections.abc import Callable
import xml.etree.ElementTree as ET

from homr.profiles.profile_2610_v1 import apply as apply_2610_v1
from homr.profiles.profile_2610_v2 import apply as apply_2610_v2
from homr.profiles.profile_2610_v2 import apply_symbols as apply_symbols_2610_v2
from homr.profiles.profile_2610_v3 import apply as apply_2610_v3
from homr.profiles.profile_2610_v3 import apply_symbols as apply_symbols_2610_v3
from homr.profiles.profile_2610_v4 import apply as apply_2610_v4
from homr.profiles.profile_2610_v4 import apply_symbols as apply_symbols_2610_v4
from homr.profiles.profile_2610_v5 import apply as apply_2610_v5
from homr.profiles.profile_2610_v5 import apply_symbols as apply_symbols_2610_v5
from homr.profiles.profile_2610_v6 import apply as apply_2610_v6
from homr.profiles.profile_2610_v6 import apply_symbols as apply_symbols_2610_v6
from homr.profiles.profile_2610_v7 import apply as apply_2610_v7
from homr.profiles.profile_2610_v7 import apply_symbols as apply_symbols_2610_v7
from homr.profiles.profile_2610_v8 import apply as apply_2610_v8
from homr.profiles.profile_2610_v8 import apply_symbols as apply_symbols_2610_v8
from homr.profiles.profile_2610_v9 import apply as apply_2610_v9
from homr.profiles.profile_2610_v9 import apply_symbols as apply_symbols_2610_v9
from homr.profiles.profile_2610_v10 import apply as apply_2610_v10
from homr.profiles.profile_2610_v10 import apply_symbols as apply_symbols_2610_v10
from homr.profiles.profile_2610_v11 import apply as apply_2610_v11
from homr.profiles.profile_2610_v11 import apply_symbols as apply_symbols_2610_v11
from homr.profiles.profile_2610_v12 import apply as apply_2610_v12
from homr.profiles.profile_2610_v12 import apply_symbols as apply_symbols_2610_v12
from homr.profiles.profile_2610_v13 import apply as apply_2610_v13
from homr.profiles.profile_2610_v13 import apply_symbols as apply_symbols_2610_v13
from homr.profiles.profile_2610_v14 import apply as apply_2610_v14
from homr.profiles.profile_2610_v14 import apply_symbols as apply_symbols_2610_v14
from homr.transformer.vocabulary import EncodedSymbol


PROFILE_NAMES = (
    "2610-ver.1",
    "2610-ver.2",
    "2610-ver.3",
    "2610-ver.4",
    "2610-ver.5",
    "2610-ver.6",
    "2610-ver.7",
    "2610-ver.8",
    "2610-ver.9",
    "2610-ver.10",
    "2610-ver.11",
    "2610-ver.12",
    "2610-ver.13",
    "2610-ver.14",
)

_PROFILES: dict[str, Callable[[ET.Element], None]] = {
    "2610-ver.1": apply_2610_v1,
    "2610-ver.2": apply_2610_v2,
    "2610-ver.3": apply_2610_v3,
    "2610-ver.4": apply_2610_v4,
    "2610-ver.5": apply_2610_v5,
    "2610-ver.6": apply_2610_v6,
    "2610-ver.7": apply_2610_v7,
    "2610-ver.8": apply_2610_v8,
    "2610-ver.9": apply_2610_v9,
    "2610-ver.10": apply_2610_v10,
    "2610-ver.11": apply_2610_v11,
    "2610-ver.12": apply_2610_v12,
    "2610-ver.13": apply_2610_v13,
    "2610-ver.14": apply_2610_v14,
}

_SYMBOL_PROFILES: dict[str, Callable[[list[list[EncodedSymbol]]], None]] = {
    "2610-ver.2": apply_symbols_2610_v2,
    "2610-ver.3": apply_symbols_2610_v3,
    "2610-ver.4": apply_symbols_2610_v4,
    "2610-ver.5": apply_symbols_2610_v5,
    "2610-ver.6": apply_symbols_2610_v6,
    "2610-ver.7": apply_symbols_2610_v7,
    "2610-ver.8": apply_symbols_2610_v8,
    "2610-ver.9": apply_symbols_2610_v9,
    "2610-ver.10": apply_symbols_2610_v10,
    "2610-ver.11": apply_symbols_2610_v11,
    "2610-ver.12": apply_symbols_2610_v12,
    "2610-ver.13": apply_symbols_2610_v13,
    "2610-ver.14": apply_symbols_2610_v14,
}


def apply_profile(root: ET.Element, profile_name: str | None) -> None:
    if profile_name is None:
        return
    _PROFILES[profile_name](root)


def apply_symbol_profile(staffs: list[list[EncodedSymbol]], profile_name: str | None) -> None:
    if profile_name is None:
        return
    apply = _SYMBOL_PROFILES.get(profile_name)
    if apply is not None:
        apply(staffs)
