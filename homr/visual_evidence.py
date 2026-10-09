from __future__ import annotations

import math
import statistics

import cv2
import numpy as np

from homr.model import Staff
from homr.point_mapping import PointMapping
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray


_NOTE_TO_DIATONIC = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
_CLEF_ANCHORS = {
    "clef_G1": "D5",
    "clef_G2": "B4",
    "clef_F3": "F3",
    "clef_F4": "D3",
    "clef_F5": "B2",
    "clef_C1": "G4",
    "clef_C2": "E4",
    "clef_C3": "C4",
    "clef_C4": "A3",
    "clef_C5": "F3",
}


def _pitch_rank(pitch: str) -> int | None:
    if len(pitch) < 2 or pitch[0] not in _NOTE_TO_DIATONIC:
        return None
    try:
        octave = int(pitch[1:])
    except ValueError:
        return None
    return octave * 7 + _NOTE_TO_DIATONIC[pitch[0]]


class VisualEvidencePage:
    """Independent image evidence used to challenge greedy transformer decisions.

    The transformer remains the primary recognizer.  This helper deliberately exposes only
    conservative evidence that can be measured directly from the page image / segmentation
    masks and therefore provides an independent vote for a later constraint solver.
    """

    def __init__(self, page_image: NDArray, notehead_mask: NDArray | None) -> None:
        self.gray = (
            page_image
            if page_image.ndim == 2
            else cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        )
        self.notehead_components: list[tuple[float, float, int, int, int, float]] = []
        if notehead_mask is not None:
            mask = (notehead_mask > 0).astype(np.uint8)
            count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            for index in range(1, count):
                x, y, width, height, area = [int(value) for value in stats[index]]
                if area < 3 or width <= 0 or height <= 0:
                    continue
                density = area / float(width * height)
                self.notehead_components.append(
                    (
                        float(centroids[index][0]),
                        float(centroids[index][1]),
                        width,
                        height,
                        area,
                        density,
                    )
                )

    @staticmethod
    def _expected_note_y(staff: Staff, x: float, pitch: str, clef: str) -> tuple[float, float] | None:
        if clef not in _CLEF_ANCHORS:
            return None
        pitch_rank = _pitch_rank(pitch)
        anchor_rank = _pitch_rank(_CLEF_ANCHORS[clef])
        point = staff.get_at(x)
        if pitch_rank is None or anchor_rank is None or point is None or len(point.y) != 5:
            return None
        # CLEF_ANCHORS encode the pitch of the middle staff line. One diatonic step is
        # half a staff-space in the image coordinate system.
        y = point.y[2] - (pitch_rank - anchor_rank) * point.average_unit_size / 2
        return float(y), float(point.average_unit_size)

    def _solid_notehead_score(self, x: float, y: float, unit: float) -> float:
        """Return positive evidence for a compact, filled notehead near the expected pitch."""
        best = 0.0
        for cx, cy, width, height, area, density in self.notehead_components:
            dx = abs(cx - x)
            dy = abs(cy - y)
            if dx > unit or dy > 0.9 * unit:
                continue
            if not (0.3 * unit <= width <= 1.4 * unit):
                continue
            if not (0.3 * unit <= height <= 1.4 * unit):
                continue
            if density < 0.65 or area < 0.2 * unit * unit:
                continue
            distance = math.sqrt((dx / unit) ** 2 + (dy / unit) ** 2)
            closeness = max(0.0, 1.0 - 0.3 * distance)
            best = max(best, density * closeness)
        return best

    def _short_bar_count(
        self,
        first: tuple[float, float, float],
        second: tuple[float, float, float],
    ) -> int:
        """Count repeated short, nearly-parallel bars between two note positions.

        Long horizontal staff / ledger lines and vertical stems are removed morphologically.
        A high count is only visual *evidence* for a tremolo candidate; the score profile's
        meter solver decides whether the candidate may actually alter recognition.
        """
        x1, y1, unit1 = first
        x2, y2, unit2 = second
        if x2 <= x1:
            return 0
        unit = (unit1 + unit2) / 2
        gap = x2 - x1
        if gap < 1.5 * unit or gap > 18 * unit:
            return 0

        best_count = 0
        for side in ("up", "down"):
            x_start = int(max(0, x1 + 0.35 * unit))
            x_stop = int(min(self.gray.shape[1], x2 - 0.35 * unit))
            if x_stop <= x_start:
                continue
            if side == "up":
                y_start = int(max(0, min(y1, y2) - 5.5 * unit))
                y_stop = int(max(y_start + 1, min(y1, y2) - 0.8 * unit))
            else:
                y_start = int(min(self.gray.shape[0] - 1, max(y1, y2) + 0.8 * unit))
                y_stop = int(min(self.gray.shape[0], max(y1, y2) + 5.5 * unit))
            crop = self.gray[y_start:y_stop, x_start:x_stop]
            if crop.size == 0:
                continue

            dark = (crop < 175).astype(np.uint8) * 255
            horizontal = cv2.morphologyEx(
                dark,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(2.5 * unit)), 1)),
            )
            vertical = cv2.morphologyEx(
                dark,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(2.5 * unit)))),
            )
            residual = cv2.subtract(cv2.subtract(dark, horizontal), vertical)
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                (residual > 0).astype(np.uint8), 8
            )
            short_bars = 0
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                width = int(stats[component, cv2.CC_STAT_WIDTH])
                height = int(stats[component, cv2.CC_STAT_HEIGHT])
                if area < max(4, int(0.08 * unit * unit)):
                    continue
                points = np.column_stack(np.where(labels == component))[:, ::-1].astype(np.float32)
                if len(points) < 2:
                    continue
                vx, vy, _x0, _y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
                angle = abs(math.degrees(math.atan2(float(vy), float(vx))))
                if angle > 90:
                    angle = 180 - angle
                if (
                    angle <= 15
                    and 0.65 * unit <= width <= 3.5 * unit
                    and height <= 0.65 * unit
                ):
                    short_bars += 1
            best_count = max(best_count, short_bars)
        return best_count

    def _measure_notehead_count(self, staff: Staff, left: float, right: float) -> int:
        """Count raw notehead-mask components in one staff/measure x-range.

        This intentionally uses the segmentation mask rather than Stage-1 Note objects so
        weak components that failed later stem/note construction still count as visual
        support. It is a diagnostic count, not a note-count ground truth.
        """
        if right < left:
            left, right = right, left
        count = 0
        for cx, cy, _width, _height, _area, _density in self.notehead_components:
            if not (left < cx < right):
                continue
            point = staff.get_at(cx)
            if point is None:
                continue
            tolerance = 5 * point.average_unit_size
            if point.y[0] - tolerance <= cy <= point.y[-1] + tolerance:
                count += 1
        return count

    def _annotate_measure_structure(
        self,
        measure: list[EncodedSymbol],
        staff: Staff,
        left: float,
        right: float,
    ) -> None:
        notes = [symbol for symbol in measure if symbol.rhythm.startswith("note")]
        notehead_count = self._measure_notehead_count(staff, left, right)
        rhythm_confidences = [
            symbol.rhythm_candidates[0][1]
            for symbol in notes
            if symbol.rhythm_candidates
        ]
        pitch_confidences = [
            symbol.pitch_candidates[0][1]
            for symbol in notes
            if symbol.pitch_candidates
        ]
        rhythm_median = statistics.median(rhythm_confidences) if rhythm_confidences else 1.0
        pitch_median = statistics.median(pitch_confidences) if pitch_confidences else 1.0
        for symbol in measure:
            symbol.visual_measure_notehead_count = notehead_count
            symbol.visual_measure_note_count = len(notes)
            symbol.visual_measure_rhythm_confidence_median = float(rhythm_median)
            symbol.visual_measure_pitch_confidence_median = float(pitch_median)

    def annotate_symbols(
        self,
        symbols: list[EncodedSymbol],
        staff: Staff,
        staff_to_page: PointMapping,
    ) -> None:
        clef = "clef_G2"
        geometry: dict[int, tuple[float, float, float]] = {}

        for symbol in symbols:
            if symbol.rhythm.startswith("clef"):
                clef = symbol.rhythm
                continue
            if not symbol.rhythm.startswith("note") or symbol.coordinates is None:
                continue
            center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
            if len(center) < 2:
                continue
            page_x, _page_y = staff_to_page((float(center[0]), float(center[1])))
            expected = self._expected_note_y(staff, page_x, symbol.pitch, clef)
            if expected is None:
                continue
            expected_y, unit = expected
            geometry[id(symbol)] = (float(page_x), expected_y, unit)
            symbol.visual_notehead_solid_score = self._solid_notehead_score(
                float(page_x), expected_y, unit
            )

        measure_notes: list[EncodedSymbol] = []
        for symbol in symbols:
            if symbol.rhythm.startswith("note"):
                measure_notes.append(symbol)
            if "barline" in symbol.rhythm or "repeat" in symbol.rhythm:
                for index in range(len(measure_notes) - 1):
                    first = measure_notes[index]
                    second = measure_notes[index + 1]
                    first_geometry = geometry.get(id(first))
                    second_geometry = geometry.get(id(second))
                    if first_geometry is None or second_geometry is None:
                        continue
                    first.visual_tremolo_with_next_bars = self._short_bar_count(
                        first_geometry, second_geometry
                    )
                measure_notes = []

        measure: list[EncodedSymbol] = []
        left = float(staff.min_x)
        for symbol in symbols:
            measure.append(symbol)
            if "barline" not in symbol.rhythm and "repeat" not in symbol.rhythm:
                continue
            right = float(staff.max_x)
            if symbol.coordinates is not None:
                center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
                if len(center) >= 2:
                    right, _ = staff_to_page((float(center[0]), float(center[1])))
            self._annotate_measure_structure(measure, staff, left, right)
            left = right
            measure = []
        if measure:
            self._annotate_measure_structure(measure, staff, left, float(staff.max_x))

