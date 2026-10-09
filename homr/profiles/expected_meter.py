import xml.etree.ElementTree as ET
from fractions import Fraction

from homr.simple_logging import eprint


def measure_extent(measure: ET.Element) -> int:
    """Return the furthest MusicXML timeline position reached inside one measure."""
    current = 0
    extent = 0
    last_note_start = 0

    for child in measure:
        if child.tag == "backup":
            current -= int(child.findtext("duration") or 0)
            continue
        if child.tag == "forward":
            current += int(child.findtext("duration") or 0)
            extent = max(extent, current)
            continue
        if child.tag != "note":
            continue

        duration = int(child.findtext("duration") or 0)
        if child.find("chord") is not None:
            extent = max(extent, last_note_start + duration)
            continue

        last_note_start = current
        current += duration
        extent = max(extent, current)

    return extent


def validate_expected_meter(root: ET.Element, beats: int, beat_type: int, profile_name: str) -> None:
    """Use a known score meter as a validator only; never rewrite the emitted meter."""
    expected_quarters = Fraction(beats * 4, beat_type)
    mismatch_count = 0

    for part_index, part in enumerate(root.findall("part"), 1):
        divisions: int | None = None
        declared_beats: str | None = None
        declared_beat_type: str | None = None

        for measure in part.findall("measure"):
            for attributes in measure.findall("attributes"):
                division_text = attributes.findtext("divisions")
                if division_text is not None:
                    divisions = int(division_text)
                time = attributes.find("time")
                if time is not None:
                    declared_beats = time.findtext("beats")
                    declared_beat_type = time.findtext("beat-type")

            if divisions is None or divisions <= 0:
                eprint(
                    f"PROFILE {profile_name} P{part_index} m{measure.get('number')} "
                    "UNKNOWN: no divisions value"
                )
                continue

            actual_quarters = Fraction(measure_extent(measure), divisions)
            delta = actual_quarters - expected_quarters
            if delta == 0:
                continue

            mismatch_count += 1
            status = "OVERFULL" if delta > 0 else "UNDERFULL"
            sign = "+" if delta > 0 else ""
            eprint(
                f"PROFILE {profile_name} P{part_index} m{measure.get('number')} {status}: "
                f"{float(actual_quarters):g}/{float(expected_quarters):g} quarter notes "
                f"({sign}{float(delta):g})"
            )

        if declared_beats is not None and declared_beat_type is not None:
            if (declared_beats, declared_beat_type) != (str(beats), str(beat_type)):
                eprint(
                    f"PROFILE {profile_name} P{part_index} DECLARED_METER_MISMATCH: "
                    f"emitted {declared_beats}/{declared_beat_type}, expected {beats}/{beat_type}; "
                    "left unchanged"
                )

    eprint(
        f"PROFILE {profile_name}: expected meter {beats}/{beat_type}; "
        f"duration mismatches={mismatch_count}; emitted meter left unchanged"
    )

