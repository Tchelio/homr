import xml.etree.ElementTree as ET

from homr.profiles.expected_meter import measure_extent
from homr.simple_logging import eprint


_SHORTER_TYPE = {
    "whole": "half",
    "half": "quarter",
    "quarter": "eighth",
    "eighth": "16th",
    "16th": "32nd",
    "32nd": "64th",
}


def _tremolo(note: ET.Element) -> ET.Element | None:
    return note.find("notations/ornaments/tremolo")


def _non_chord_notes(measure: ET.Element) -> list[ET.Element]:
    return [note for note in measure.findall("note") if note.find("chord") is None]


def _double_tremolo_pairs(notes: list[ET.Element]) -> list[tuple[ET.Element, ET.Element]]:
    pairs: list[tuple[ET.Element, ET.Element]] = []
    index = 0
    while index + 1 < len(notes):
        first = notes[index]
        second = notes[index + 1]
        first_tremolo = _tremolo(first)
        second_tremolo = _tremolo(second)
        first_duration = int(first.findtext("duration") or 0)
        second_duration = int(second.findtext("duration") or 0)

        is_pair = (
            first.find("rest") is None
            and second.find("rest") is None
            and first_tremolo is not None
            and second_tremolo is not None
            and first_tremolo.get("type") == "start"
            and second_tremolo.get("type") == "stop"
            and first_duration > 0
            and first_duration == second_duration
            and first.findtext("type") == second.findtext("type")
            and len(first.findall("dot")) == 0
            and len(second.findall("dot")) == 0
        )
        if is_pair:
            pairs.append((first, second))
            index += 2
        else:
            index += 1
    return pairs


def _has_right_tie_anchor(notes: list[ET.Element]) -> bool:
    pitched = [note for note in notes if note.find("rest") is None]
    if not pitched:
        return False
    return any(tie.get("type") == "start" for tie in pitched[-1].findall("tie"))


def _insert_time_modification(note: ET.Element) -> None:
    existing = note.find("time-modification")
    if existing is not None:
        note.remove(existing)

    time_mod = ET.Element("time-modification")
    ET.SubElement(time_mod, "actual-notes").text = "2"
    ET.SubElement(time_mod, "normal-notes").text = "1"

    children = list(note)
    insert_at = len(children)
    for index, child in enumerate(children):
        if child.tag in ("staff", "beam", "notations", "lyric"):
            insert_at = index
            break
    note.insert(insert_at, time_mod)


def _apply_pair_semantics(
    pair: tuple[ET.Element, ET.Element], shorter_type: str | None = None
) -> None:
    first, second = pair
    original_duration = int(first.findtext("duration") or 0)
    nominal_duration = original_duration if shorter_type is None else original_duration // 2
    actual_duration = nominal_duration // 2
    if actual_duration <= 0:
        raise ValueError("Double-note tremolo duration became non-positive")

    for note in (first, second):
        duration = note.find("duration")
        if duration is not None:
            duration.text = str(actual_duration)
        if shorter_type is not None:
            note_type = note.find("type")
            if note_type is not None:
                note_type.text = shorter_type
        _insert_time_modification(note)


def reconcile_anchored_double_tremolos(
    root: ET.Element, beats: int, beat_type: int, profile_name: str
) -> int:
    """
    Correct a double-note tremolo only when strong rhythmic anchors make the solution unique.

    This deliberately does not encode any handwritten glyph as "always an eighth note".
    A correction is allowed only when:
    - the measure has at least two recognized rests,
    - its last pitched note ties across the right barline,
    - it contains at least two explicit double-note tremolo pairs,
    - applying MusicXML 2:1 double-tremolo timing still leaves the known meter overfull, and
    - shortening exactly one tremolo pair by one standard duration step is the unique exact fit.
    """
    expected_quarters = beats * 4 / beat_type
    corrected = 0

    for part_index, part in enumerate(root.findall("part"), 1):
        divisions: int | None = None
        for measure in part.findall("measure"):
            for attributes in measure.findall("attributes"):
                division_text = attributes.findtext("divisions")
                if division_text is not None:
                    divisions = int(division_text)
            if divisions is None or divisions <= 0:
                continue
            if measure.find("backup") is not None or measure.find("forward") is not None:
                continue

            notes = _non_chord_notes(measure)
            rest_count = sum(1 for note in notes if note.find("rest") is not None)
            pairs = _double_tremolo_pairs(notes)
            if rest_count < 2 or len(pairs) < 2 or not _has_right_tie_anchor(notes):
                continue

            raw_extent = measure_extent(measure)
            pair_durations = [int(pair[0].findtext("duration") or 0) for pair in pairs]
            after_standard_semantics = raw_extent - sum(pair_durations)
            expected_units = int(expected_quarters * divisions)

            if after_standard_semantics <= expected_units:
                continue

            candidates: list[tuple[int, str]] = []
            for pair_index, pair in enumerate(pairs):
                duration = int(pair[0].findtext("duration") or 0)
                note_type = pair[0].findtext("type")
                shorter_type = _SHORTER_TYPE.get(note_type or "")
                if shorter_type is None or duration % 2 != 0:
                    continue
                candidate_extent = after_standard_semantics - duration // 2
                if candidate_extent == expected_units:
                    candidates.append((pair_index, shorter_type))

            if len(candidates) != 1:
                continue

            corrected_pair_index, shorter_type = candidates[0]
            original_type = pairs[corrected_pair_index][0].findtext("type") or "unknown"
            for pair_index, pair in enumerate(pairs):
                _apply_pair_semantics(
                    pair, shorter_type if pair_index == corrected_pair_index else None
                )

            corrected += 1
            eprint(
                f"PROFILE {profile_name} P{part_index} m{measure.get('number')} "
                "UNIQUE_TREMOLO_RECONCILIATION: "
                f"pair {corrected_pair_index + 1} {original_type}->{shorter_type}; "
                f"measure {raw_extent / divisions:g}->{measure_extent(measure) / divisions:g} "
                "quarter notes"
            )

    eprint(f"PROFILE {profile_name}: unique anchored tremolo corrections={corrected}")
    return corrected

