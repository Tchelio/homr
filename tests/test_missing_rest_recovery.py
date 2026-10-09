import unittest

from homr.profiles.missing_rest_recovery import (
    recover_exact_fit_masked_rests,
    recover_missing_rests_between_tremolo_pairs,
    recover_underfull_unique_visual_gap_rest,
)
from homr.transformer.vocabulary import EncodedSymbol


def _tremolo(rhythm: str, pitch: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm, pitch=pitch, articulation="tremolo")


class TestMissingRestRecovery(unittest.TestCase):
    def test_recovers_exact_standard_rest_from_unique_visual_gap(self) -> None:
        first = _tremolo("note_2", "E3")
        second = _tremolo("note_2", "G3")
        third = _tremolo("note_4.", "E3")
        fourth = _tremolo("note_4.", "G3")
        second.visual_gap_after_score = 0.8
        second.visual_gap_after_page_x = 150.0
        voice = [first, second, third, fourth, EncodedSymbol("barline")]

        recovered = recover_missing_rests_between_tremolo_pairs([voice], 4, 4, "test")

        self.assertEqual(recovered, 1)
        self.assertEqual(
            [symbol.rhythm for symbol in voice],
            ["note_2", "note_2", "rest_8", "note_4.", "note_4.", "barline"],
        )
        self.assertTrue(voice[2].missing_rest_recovery)

    def test_fails_closed_without_visual_support(self) -> None:
        first = _tremolo("note_2", "E3")
        second = _tremolo("note_2", "G3")
        third = _tremolo("note_4.", "E3")
        fourth = _tremolo("note_4.", "G3")
        voice = [first, second, third, fourth, EncodedSymbol("barline")]

        recovered = recover_missing_rests_between_tremolo_pairs([voice], 4, 4, "test")

        self.assertEqual(recovered, 0)
        self.assertEqual(len(voice), 5)

    def test_exact_fit_masking_recovers_rest_and_shorter_note_together(self) -> None:
        half_rest = EncodedSymbol("rest_2", "_")
        quarter_rest = EncodedSymbol("rest_4", "_")
        quarter_rest.visual_gap_after_score = 0.527
        quarter_rest.visual_gap_after_page_x = 150.0
        note = EncodedSymbol("note_4", "G3")
        note.rhythm_candidates = [("note_4", 0.561), ("note_8", 0.267)]
        voice = [half_rest, quarter_rest, note, EncodedSymbol("barline")]

        recovered = recover_exact_fit_masked_rests(
            [voice], beats=4, beat_type=4, profile_name="test"
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(
            [symbol.rhythm for symbol in voice],
            ["rest_2", "rest_4", "rest_8", "note_8", "barline"],
        )
        self.assertTrue(voice[2].missing_rest_recovery)

    def test_exact_fit_masking_fails_closed_on_weak_rhythm_alternative(self) -> None:
        half_rest = EncodedSymbol("rest_2", "_")
        quarter_rest = EncodedSymbol("rest_4", "_")
        quarter_rest.visual_gap_after_score = 0.527
        quarter_rest.visual_gap_after_page_x = 150.0
        note = EncodedSymbol("note_4", "G3")
        note.rhythm_candidates = [("note_4", 0.8), ("note_8", 0.12)]
        note.visual_notehead_solid_score = 0.7
        voice = [half_rest, quarter_rest, note, EncodedSymbol("barline")]

        recovered = recover_exact_fit_masked_rests(
            [voice], beats=4, beat_type=4, profile_name="test"
        )

        self.assertEqual(recovered, 0)
        self.assertEqual(note.rhythm, "note_4")

    def test_masked_rest_can_follow_visual_uncertain_eighth_at_measure_end(self) -> None:
        dotted_half = EncodedSymbol("note_2.", "G3")
        dotted_half.rhythm_candidates = [("note_2.", 0.86)]
        note = EncodedSymbol("note_4", "G3")
        note.rhythm_candidates = [("note_4", 0.892), ("note_8", 0.006)]
        note.visual_notehead_solid_score = 0.0
        note.visual_gap_after_score = 1.0
        note.visual_gap_after_page_x = 200.0
        voice = [dotted_half, note, EncodedSymbol("barline")]

        recovered = recover_exact_fit_masked_rests(
            [voice],
            beats=4,
            beat_type=4,
            profile_name="test",
            allow_after_note_gap=True,
            allow_deeper_note_alternatives=True,
            allow_visual_uncertain_eighth=True,
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(
            [symbol.rhythm for symbol in voice],
            ["note_2.", "note_8", "rest_8", "barline"],
        )

    def test_masked_rest_preserves_explicit_tremolo_timing_when_overfull(self) -> None:
        first = _tremolo("note_2", "G3")
        first.rhythm_candidates = [("note_2", 0.938)]
        first.slur = "slurStart"
        second = _tremolo("note_2", "G3")
        second.rhythm_candidates = [("note_2", 0.939)]
        note = EncodedSymbol("note_2", "G3")
        note.rhythm_candidates = [("note_2", 0.788), ("note_4", 0.117), ("note_8", 0.003)]
        note.visual_notehead_solid_score = 0.0
        note.visual_gap_after_score = 1.0
        note.visual_gap_after_page_x = 250.0
        rest = EncodedSymbol("rest_4", "_")
        rest.rhythm_candidates = [("rest_4", 0.83)]
        voice = [first, second, note, rest, EncodedSymbol("barline")]

        recovered = recover_exact_fit_masked_rests(
            [voice],
            beats=4,
            beat_type=4,
            profile_name="test",
            allow_after_note_gap=True,
            allow_explicit_tremolo_overage=True,
            allow_deeper_note_alternatives=True,
            allow_visual_uncertain_eighth=True,
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(
            [symbol.rhythm for symbol in voice],
            ["note_2", "note_2", "note_8", "rest_8", "rest_4", "barline"],
        )

    def test_underfull_unique_rest_to_note_gap_inserts_standard_rest(self) -> None:
        first = EncodedSymbol("rest_4", "_")
        second = EncodedSymbol("rest_4", "_")
        second.visual_gap_after_score = 1.0
        second.visual_gap_after_page_x = 150.0
        note = EncodedSymbol("note_8", "G3")
        voice = [EncodedSymbol("note_8", "E3"), first, second, note, EncodedSymbol("barline")]

        recovered = recover_underfull_unique_visual_gap_rest(
            [voice], beats=4, beat_type=4, profile_name="test"
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(
            [symbol.rhythm for symbol in voice],
            ["note_8", "rest_4", "rest_4", "rest_4", "note_8", "barline"],
        )


if __name__ == "__main__":
    unittest.main()
