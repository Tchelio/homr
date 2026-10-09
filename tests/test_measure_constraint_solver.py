import unittest

from homr.profiles.measure_constraint_solver import reconcile_unique_measure_constraints
from homr.transformer.vocabulary import EncodedSymbol


def _note(rhythm: str, pitch: str) -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm, pitch=pitch, articulation="tremolo")
    symbol.rhythm_candidates = [(rhythm, 0.9)]
    return symbol


def _plain_note(rhythm: str, pitch: str, visual_bars: int = 0) -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm, pitch=pitch)
    symbol.rhythm_candidates = [(rhythm, 0.9)]
    symbol.visual_tremolo_with_next_bars = visual_bars
    return symbol


class TestMeasureConstraintSolver(unittest.TestCase):
    def test_fail_closed_when_exact_fit_discards_explicit_tremolo_pair(self) -> None:
        first = _note("note_2", "E3")
        second = _note("note_2", "G3")
        third = _note("note_4.", "E3")
        fourth = _note("note_4.", "G3")
        voice = [first, second, third, fourth, EncodedSymbol("barline")]

        corrected = reconcile_unique_measure_constraints(
            [voice],
            beats=4,
            beat_type=4,
            profile_name="test",
            allow_dotted_tremolo=True,
            prefer_evidence_when_ambiguous=True,
            require_all_explicit_tremolo_pairs=True,
        )

        self.assertEqual(corrected, 0)
        self.assertEqual(
            [first.rhythm, second.rhythm, third.rhythm, fourth.rhythm],
            ["note_2", "note_2", "note_4.", "note_4."],
        )
        self.assertEqual(
            [first.articulation, second.articulation, third.articulation, fourth.articulation],
            ["tremolo", "tremolo", "tremolo", "tremolo"],
        )

    def test_medium_visual_tremolo_reuses_explicit_pitch_pattern(self) -> None:
        first = EncodedSymbol("rest_8")
        second = _note("note_4.", "E3")
        third = _note("note_4.", "G3")
        fourth = EncodedSymbol("rest_4")
        fifth = _plain_note("note_4", "E3", visual_bars=4)
        sixth = _plain_note("note_4", "G3")
        voice = [first, second, third, fourth, fifth, sixth, EncodedSymbol("barline")]

        corrected = reconcile_unique_measure_constraints(
            [voice],
            beats=4,
            beat_type=4,
            profile_name="test",
            allow_dotted_tremolo=True,
            prefer_evidence_when_ambiguous=True,
            require_all_explicit_tremolo_pairs=True,
            allow_repeated_pattern_medium_tremolo=True,
        )

        self.assertEqual(corrected, 1)
        self.assertEqual(fifth.double_tremolo_role, "start")
        self.assertEqual(sixth.double_tremolo_role, "stop")
        self.assertEqual(fifth.double_tremolo_source, "visual")
        self.assertEqual(sixth.double_tremolo_source, "visual")

    def test_medium_visual_tremolo_rejects_different_pitch_pattern(self) -> None:
        first = EncodedSymbol("rest_8")
        second = _note("note_4.", "E3")
        third = _note("note_4.", "G3")
        fourth = EncodedSymbol("rest_4")
        fifth = _plain_note("note_4", "F3", visual_bars=4)
        sixth = _plain_note("note_4", "A3")
        voice = [first, second, third, fourth, fifth, sixth, EncodedSymbol("barline")]

        corrected = reconcile_unique_measure_constraints(
            [voice],
            beats=4,
            beat_type=4,
            profile_name="test",
            allow_dotted_tremolo=True,
            prefer_evidence_when_ambiguous=True,
            require_all_explicit_tremolo_pairs=True,
            allow_repeated_pattern_medium_tremolo=True,
        )

        self.assertEqual(corrected, 0)
        self.assertIsNone(fifth.double_tremolo_role)
        self.assertIsNone(sixth.double_tremolo_role)

    def test_solid_dotted_half_preserves_dot_when_shortened(self) -> None:
        first = _plain_note("note_2", "A3")
        second = _plain_note("note_8", "A3")
        third = _plain_note("note_2.", "A3")
        third.visual_notehead_solid_score = 0.637
        voice = [first, second, third, EncodedSymbol("barline")]

        corrected = reconcile_unique_measure_constraints(
            [voice],
            beats=4,
            beat_type=4,
            profile_name="test",
            prefer_evidence_when_ambiguous=True,
        )

        self.assertEqual(corrected, 1)
        self.assertEqual(third.rhythm, "note_4.")

    def test_dotted_half_below_solid_threshold_is_unchanged(self) -> None:
        first = _plain_note("note_2", "A3")
        second = _plain_note("note_8", "A3")
        third = _plain_note("note_2.", "A3")
        third.visual_notehead_solid_score = 0.59
        voice = [first, second, third, EncodedSymbol("barline")]

        corrected = reconcile_unique_measure_constraints(
            [voice],
            beats=4,
            beat_type=4,
            profile_name="test",
            prefer_evidence_when_ambiguous=True,
        )

        self.assertEqual(corrected, 0)
        self.assertEqual(third.rhythm, "note_2.")

    def test_visual_uncertain_quarter_can_become_eighth_for_unique_meter_fit(self) -> None:
        first = _plain_note("note_2", "A3")
        second = _plain_note("note_4", "A3")
        second.rhythm_candidates = [("note_4", 0.874), ("note_8", 0.003)]
        second.visual_notehead_solid_score = 0.0
        third = _plain_note("note_4.", "A3")
        voice = [first, second, third, EncodedSymbol("barline")]

        corrected = reconcile_unique_measure_constraints(
            [voice],
            beats=4,
            beat_type=4,
            profile_name="test",
            prefer_evidence_when_ambiguous=True,
            allow_visual_uncertain_quarter_to_eighth=True,
        )

        self.assertEqual(corrected, 1)
        self.assertEqual(second.rhythm, "note_8")

    def test_deeper_rest_alternative_can_preserve_stronger_tremolo_reading(self) -> None:
        first = _plain_note("note_4", "A3")
        first.rhythm_candidates = [("note_4", 0.841), ("note_8", 0.004)]
        first.visual_notehead_solid_score = 0.0
        second = EncodedSymbol("rest_2")
        second.rhythm_candidates = [("rest_2", 0.431), ("rest_8", 0.216), ("rest_4", 0.060)]
        third = EncodedSymbol("rest_4")
        third.rhythm_candidates = [("rest_4", 0.892)]
        fourth = _note("note_2", "E3")
        fourth.rhythm_candidates = [("note_2", 0.889), ("note_4", 0.032)]
        fifth = _note("note_2", "G3")
        fifth.rhythm_candidates = [("note_2", 0.922)]
        voice = [first, second, third, fourth, fifth, EncodedSymbol("barline")]

        corrected = reconcile_unique_measure_constraints(
            [voice],
            beats=4,
            beat_type=4,
            profile_name="test",
            allow_dotted_tremolo=True,
            allow_rest_rhythm_alternatives=True,
            prefer_evidence_when_ambiguous=True,
            require_all_explicit_tremolo_pairs=True,
            allow_visual_uncertain_quarter_to_eighth=True,
            allow_deeper_rest_rhythm_alternatives=True,
        )

        self.assertEqual(corrected, 1)
        self.assertEqual(first.rhythm, "note_8")
        self.assertEqual(second.rhythm, "rest_8")
        self.assertEqual(fourth.rhythm, "note_2")
        self.assertEqual(fifth.rhythm, "note_2")
        self.assertEqual(fourth.double_tremolo_role, "start")
        self.assertEqual(fifth.double_tremolo_role, "stop")


if __name__ == "__main__":
    unittest.main()
