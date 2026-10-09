import xml.etree.ElementTree as ET

from homr.simple_logging import eprint


def _tremolo(note: ET.Element) -> ET.Element | None:
    return note.find("notations/ornaments/tremolo")


def _is_two_to_one(note: ET.Element) -> bool:
    return (
        note.findtext("time-modification/actual-notes") == "2"
        and note.findtext("time-modification/normal-notes") == "1"
    )


def _insert_beam(note: ET.Element, value: str) -> None:
    if note.find("beam") is not None:
        return
    beam = ET.Element("beam", number="1")
    beam.text = value
    children = list(note)
    insert_at = len(children)
    for index, child in enumerate(children):
        if child.tag in ("notations", "lyric", "play", "listen"):
            insert_at = index
            break
    note.insert(insert_at, beam)


def add_dorico_double_tremolo_beam_hints(root: ET.Element, profile_name: str) -> int:
    """
    Add an explicit beam group to already-confirmed 2:1 double-note tremolos.

    MusicXML represents attached beams separately from tremolo marks. Dorico's importer has
    historically tied multi-note tremolo import to beam grouping, so this profile-specific
    compatibility hint is deliberately limited to pairs that have already been reconciled as
    true 2:1 double-note tremolos. It does not create tremolos from unlabeled notes.
    """
    pairs_changed = 0
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            notes = [note for note in measure.findall("note") if note.find("chord") is None]
            for index in range(len(notes) - 1):
                first, second = notes[index], notes[index + 1]
                first_tremolo = _tremolo(first)
                second_tremolo = _tremolo(second)
                if first_tremolo is None or second_tremolo is None:
                    continue
                if first_tremolo.get("type") != "start" or second_tremolo.get("type") != "stop":
                    continue
                if not _is_two_to_one(first) or not _is_two_to_one(second):
                    continue
                _insert_beam(first, "begin")
                _insert_beam(second, "end")
                pairs_changed += 1

    eprint(f"PROFILE {profile_name}: Dorico double-tremolo beam hints={pairs_changed}")
    return pairs_changed


def _remove_beams(note: ET.Element) -> None:
    for beam in list(note.findall("beam")):
        note.remove(beam)


def _set_tremolo_marks(note: ET.Element, marks: int) -> None:
    tremolo = _tremolo(note)
    if tremolo is not None:
        tremolo.text = str(marks)


def _remove_tremolo(note: ET.Element) -> None:
    for notations in list(note.findall("notations")):
        ornaments = notations.find("ornaments")
        if ornaments is None:
            continue
        for tremolo in list(ornaments.findall("tremolo")):
            ornaments.remove(tremolo)
        if len(ornaments) == 0:
            notations.remove(ornaments)
        if len(notations) == 0:
            note.remove(notations)


def normalize_dorico_reference_tremolos(root: ET.Element, profile_name: str) -> int:
    """Normalize already-confirmed double tremolos to the Dorico round-trip reference.

    The current TrOMR vocabulary collapses every tremolo into one articulation token, so it
    cannot recover the number of tremolo marks from the model output.  For the 2610 profile we
    therefore use only note-type combinations that have been cross-validated against a Dorico
    edit/export of the same handwritten material:

    - half-note pair: four tremolo marks, no attached beam element;
    - eighth-note pair: three tremolo marks plus one begin/end beam.

    This function only touches pairs that already carry the MusicXML 2:1 double-tremolo time
    modification.  It does not create a tremolo from an arbitrary pair of notes.  Within a
    measure containing such confirmed pairs, dangling start/stop tremolo labels that are not
    part of a confirmed pair are removed instead of leaving an invalid half-pair for importers.
    """
    normalized_pairs = 0

    for part in root.findall("part"):
        for measure in part.findall("measure"):
            notes = [note for note in measure.findall("note") if note.find("chord") is None]
            confirmed_pair_note_ids: set[int] = set()
            has_normalized_pair = False

            for index in range(len(notes) - 1):
                first, second = notes[index], notes[index + 1]
                first_tremolo = _tremolo(first)
                second_tremolo = _tremolo(second)
                if first_tremolo is None or second_tremolo is None:
                    continue
                if first_tremolo.get("type") != "start" or second_tremolo.get("type") != "stop":
                    continue
                if not _is_two_to_one(first) or not _is_two_to_one(second):
                    continue

                first_type = first.findtext("type")
                second_type = second.findtext("type")
                if first_type != second_type:
                    continue

                if first_type == "half":
                    confirmed_pair_note_ids.add(id(first))
                    confirmed_pair_note_ids.add(id(second))
                    has_normalized_pair = True
                    _remove_beams(first)
                    _remove_beams(second)
                    _set_tremolo_marks(first, 4)
                    _set_tremolo_marks(second, 4)
                elif first_type == "eighth":
                    confirmed_pair_note_ids.add(id(first))
                    confirmed_pair_note_ids.add(id(second))
                    has_normalized_pair = True
                    _insert_beam(first, "begin")
                    _insert_beam(second, "end")
                    _set_tremolo_marks(first, 3)
                    _set_tremolo_marks(second, 3)
                elif first.find("dot") is not None and second.find("dot") is not None:
                    # Dotted written values are already semantically confirmed by the 2:1
                    # timing and start/stop pair, but we do not yet have a Dorico-specific
                    # normalization recipe for them. Preserve the writer's marks unchanged.
                    # This avoids deleting a valid dotted tremolo merely because another,
                    # supported pair appears in the same measure.
                    confirmed_pair_note_ids.add(id(first))
                    confirmed_pair_note_ids.add(id(second))
                    continue
                else:
                    # No Dorico round-trip evidence yet for this note type. Fail closed on
                    # normalization and retain the historical cleanup behavior.
                    continue
                normalized_pairs += 1

            # Preserve the historical behavior of measures containing only unsupported
            # written values: without a half/eighth pair there was never a cleanup pass.
            # A dotted pair only protects itself when cleanup is already required by a
            # supported pair in the same measure.
            if not has_normalized_pair:
                continue

            for note in notes:
                if id(note) in confirmed_pair_note_ids:
                    continue
                tremolo = _tremolo(note)
                if tremolo is not None and tremolo.get("type") in ("start", "stop"):
                    _remove_tremolo(note)

    eprint(f"PROFILE {profile_name}: Dorico reference tremolo pairs={normalized_pairs}")
    return normalized_pairs
