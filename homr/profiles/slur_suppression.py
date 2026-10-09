import xml.etree.ElementTree as ET

from homr.simple_logging import eprint


def suppress_remaining_slurs(root: ET.Element, profile_name: str) -> int:
    """Remove only MusicXML slur elements after tie conversion has already run.

    The generic writer converts same-pitch adjacent slurs into <tie>/<tied> before profile
    post-processing.  This compatibility pass therefore leaves sounding ties intact while
    discarding the remaining graphical <slur> elements for Dorico MusicXML import testing.
    """
    removed = 0
    for note in root.findall("./part/measure/note"):
        for notations in list(note.findall("notations")):
            for slur in list(notations.findall("slur")):
                notations.remove(slur)
                removed += 1
            if len(notations) == 0:
                note.remove(notations)
    eprint(f"PROFILE {profile_name}: suppressed remaining MusicXML slurs={removed}")
    return removed


def suppress_ties_into_double_tremolos(root: ET.Element, profile_name: str) -> int:
    """Remove the same-pitch connection immediately entering a double-note tremolo.

    `convert_ties()` runs before profile post-processing, so a handwritten slur between
    adjacent notes of the same pitch has already become a MusicXML tie by this point.  For
    the Dorico-import experiment we remove only the *local* tie from the preceding note into
    the first note of a confirmed double tremolo.  An earlier incoming tie into that
    preceding note (for example across a barline) is preserved.
    """
    removed_pairs = 0

    for measure in root.findall("./part/measure"):
        notes = [note for note in measure.findall("note") if note.find("chord") is None]
        for index in range(1, len(notes)):
            current = notes[index]
            tremolo = current.find("notations/ornaments/tremolo")
            if tremolo is None or tremolo.get("type") != "start":
                continue

            previous = notes[index - 1]
            previous_pitch = (
                previous.findtext("pitch/step"),
                previous.findtext("pitch/alter"),
                previous.findtext("pitch/octave"),
            )
            current_pitch = (
                current.findtext("pitch/step"),
                current.findtext("pitch/alter"),
                current.findtext("pitch/octave"),
            )
            if previous_pitch != current_pitch or previous_pitch[0] is None:
                continue

            previous_start = next(
                (tie for tie in previous.findall("tie") if tie.get("type") == "start"), None
            )
            current_stop = next(
                (tie for tie in current.findall("tie") if tie.get("type") == "stop"), None
            )
            previous_tied_start = next(
                (
                    tied
                    for tied in previous.findall("notations/tied")
                    if tied.get("type") == "start"
                ),
                None,
            )
            current_tied_stop = next(
                (
                    tied
                    for tied in current.findall("notations/tied")
                    if tied.get("type") == "stop"
                ),
                None,
            )
            if previous_start is None or current_stop is None:
                continue

            previous.remove(previous_start)
            current.remove(current_stop)

            if previous_tied_start is not None:
                notations = previous.find("notations")
                if notations is not None:
                    notations.remove(previous_tied_start)
                    if len(notations) == 0:
                        previous.remove(notations)
            if current_tied_stop is not None:
                notations = current.find("notations")
                if notations is not None:
                    notations.remove(current_tied_stop)
                    if len(notations) == 0:
                        current.remove(notations)

            removed_pairs += 1

    eprint(
        f"PROFILE {profile_name}: suppressed ties entering double tremolos={removed_pairs}"
    )
    return removed_pairs
