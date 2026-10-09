import unittest

from homr.profiles.pitch_reconciliation import reconcile_repeated_tremolo_pitches
from homr.transformer.vocabulary import EncodedSymbol


class TestPitchReconciliation(unittest.TestCase):
    def test_recovered_measure_can_use_unique_geometry_only_tremolo_pitch(self) -> None:
        first = EncodedSymbol("note_2", "G3", articulation="tremolo")
        first.pitch_candidates = [("G3", 0.576), ("E3", 0.388)]
        first.image_coordinates = (100.0, 110.0)
        second = EncodedSymbol("note_2", "G3", articulation="tremolo")
        second.pitch_candidates = [("G3", 0.999)]
        second.image_coordinates = (140.0, 100.0)
        recovered_rest = EncodedSymbol("rest_8", "_")
        recovered_rest.missing_rest_recovery = True
        voice = [first, second, recovered_rest, EncodedSymbol("rest_4"), EncodedSymbol("barline")]

        corrected = reconcile_repeated_tremolo_pitches(
            [voice],
            profile_name="test",
            allow_structural_recovery=True,
            allow_recovered_geometry_fallback=True,
        )

        self.assertEqual(corrected, 1)
        self.assertEqual(first.pitch, "E3")

    def test_geometry_only_fallback_requires_recovered_measure(self) -> None:
        first = EncodedSymbol("note_2", "G3", articulation="tremolo")
        first.pitch_candidates = [("G3", 0.576), ("E3", 0.388)]
        first.image_coordinates = (100.0, 110.0)
        second = EncodedSymbol("note_2", "G3", articulation="tremolo")
        second.pitch_candidates = [("G3", 0.999)]
        second.image_coordinates = (140.0, 100.0)
        voice = [first, second, EncodedSymbol("rest_8"), EncodedSymbol("rest_4"), EncodedSymbol("barline")]

        corrected = reconcile_repeated_tremolo_pitches(
            [voice],
            profile_name="test",
            allow_structural_recovery=True,
            allow_recovered_geometry_fallback=True,
        )

        self.assertEqual(corrected, 0)
        self.assertEqual(first.pitch, "G3")


if __name__ == "__main__":
    unittest.main()
