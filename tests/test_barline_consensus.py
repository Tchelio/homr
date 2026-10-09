import unittest

from homr.profiles.barline_consensus import reconcile_system_barlines
from homr.profiles.structural_recovery import recover_suppressed_barline_prefixes
from homr.transformer.vocabulary import EncodedSymbol


def _symbol(rhythm: str, x: float) -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm)
    symbol.page_coordinates = (x, 0.0)
    if "barline" in rhythm or "repeat" in rhythm:
        symbol.barline_geometry_page_x = x
        symbol.barline_geometry_unit_size = 10.0
    return symbol


class TestBarlineConsensus(unittest.TestCase):
    def test_removes_isolated_and_fills_majority_barline(self) -> None:
        voices = []
        for voice in range(4):
            first_system = [
                _symbol("note_4", 50),
                _symbol("barline", 100),
                _symbol("note_4", 140),
            ]
            if voice == 3:
                first_system.insert(2, _symbol("barline", 160))

            second_system = [_symbol("note_4", 250)]
            if voice != 2:
                second_system.append(_symbol("barline", 300))
            second_system.append(_symbol("note_4", 340))

            voices.append(first_system + [EncodedSymbol("newline")] + second_system)

        removed, inserted = reconcile_system_barlines(voices, "test")

        self.assertEqual(removed, 1)
        self.assertEqual(inserted, 1)
        for voice in voices:
            systems = [[]]
            for symbol in voice:
                if symbol.rhythm == "newline":
                    systems.append([])
                else:
                    systems[-1].append(symbol)
            if systems and not systems[-1]:
                systems.pop()
            self.assertEqual(
                [[round(symbol.barline_geometry_page_x or 0) for symbol in system if symbol.rhythm == "barline"] for system in systems],
                [[100], [300]],
            )

    def test_structural_prefix_uses_suppressed_boundary_span(self) -> None:
        left = _symbol("barline", 100)
        bad_a = _symbol("note_8G", 112)
        bad_b = _symbol("note_1", 120)
        for bad in (bad_a, bad_b):
            bad.structural_visual_mismatch = True
            bad.suppressed_barline_after_page_x = 132
        good = _symbol("note_2", 150)
        right = _symbol("barline", 200)
        voice = [left, bad_a, bad_b, good, right]

        recovered = recover_suppressed_barline_prefixes([voice], 4, 4, "test")

        self.assertEqual(recovered, 1)
        self.assertEqual([symbol.rhythm for symbol in voice], ["barline", "rest_4", "note_2", "barline"])


if __name__ == "__main__":
    unittest.main()
