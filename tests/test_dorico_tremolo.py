import unittest
import xml.etree.ElementTree as ET

from homr.profiles.dorico_tremolo import normalize_dorico_reference_tremolos


def _confirmed_note(note_type: str, tremolo_type: str, marks: str = "3") -> ET.Element:
    note = ET.Element("note")
    ET.SubElement(note, "duration").text = "1"
    ET.SubElement(note, "type").text = note_type
    time_mod = ET.SubElement(note, "time-modification")
    ET.SubElement(time_mod, "actual-notes").text = "2"
    ET.SubElement(time_mod, "normal-notes").text = "1"
    notations = ET.SubElement(note, "notations")
    ornaments = ET.SubElement(notations, "ornaments")
    tremolo = ET.SubElement(ornaments, "tremolo", type=tremolo_type)
    tremolo.text = marks
    return note


class TestDoricoTremolo(unittest.TestCase):
    def test_preserves_confirmed_unsupported_note_type(self) -> None:
        root = ET.Element("score-partwise")
        part = ET.SubElement(root, "part")
        measure = ET.SubElement(part, "measure")
        first = _confirmed_note("quarter", "start")
        ET.SubElement(first, "dot")
        second = _confirmed_note("quarter", "stop")
        ET.SubElement(second, "dot")
        measure.extend((first, second))

        normalized = normalize_dorico_reference_tremolos(root, "test")

        self.assertEqual(normalized, 0)
        self.assertEqual(first.findtext("notations/ornaments/tremolo"), "3")
        self.assertEqual(second.findtext("notations/ornaments/tremolo"), "3")


if __name__ == "__main__":
    unittest.main()
