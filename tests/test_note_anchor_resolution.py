import unittest

import numpy as np

from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_evidence import VisualEvidencePage


class _Point:
    y = [40.0, 50.0, 60.0, 70.0, 80.0]
    average_unit_size = 10.0


class _Note:
    def __init__(self, x: float, y: float = 60.0) -> None:
        self.center = (x, y)


class _Staff:
    average_unit_size = 10.0

    def __init__(self, notes: list[_Note]) -> None:
        self._notes = notes

    def get_at(self, _x: float) -> _Point:
        return _Point()

    def get_notes(self) -> list[_Note]:
        return self._notes


def _event(x: float) -> EncodedSymbol:
    symbol = EncodedSymbol("note_8", "B4")
    symbol.page_coordinates = (x, 60.0)
    return symbol


class TestNoteAnchorResolution(unittest.TestCase):
    def test_order_alignment_exposes_gap_backed_missing_note(self) -> None:
        events = [_event(10), _event(20), _event(60), _event(120)]
        events[2].visual_gap_after_score = 1.0
        events[2].visual_gap_after_page_x = 100.0
        staff = _Staff([_Note(10), _Note(20), _Note(33), _Note(100), _Note(120)])
        evidence = VisualEvidencePage(np.full((140, 160), 255, np.uint8), None)

        anchors, missing = evidence.resolve_note_anchors(events, staff)

        self.assertEqual(anchors[id(events[2])].source, "stage1")
        self.assertEqual(anchors[id(events[2])].page_x, 33.0)
        self.assertEqual([(item.page_x, item.score) for item in missing], [(100.0, 1.0)])

    def test_ledger_support_reads_real_horizontal_ink(self) -> None:
        image = np.full((120, 120), 255, np.uint8)
        image[29:32, 38:63] = 0
        evidence = VisualEvidencePage(image, None)
        staff = _Staff([])

        count, score = evidence._ledger_line_support(staff, 50.0, "A5", "clef_G2")

        self.assertEqual(count, 1)
        self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()
