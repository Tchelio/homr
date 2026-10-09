from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ResolvedNoteAnchor:
    page_x: float
    page_y: float
    source: str
    ledger_support: float = 1.0


@dataclass(frozen=True)
class MissingNoteCandidate:
    page_x: float
    page_y: float
    score: float


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

    def _ledger_line_support(
        self, staff: Staff, x: float, pitch: str, clef: str
    ) -> tuple[int, float]:
        """Measure short horizontal support for ledger lines implied by a pitch.

        This is raw-image evidence only. It does not create or change a pitch; it lets
        diagnostics distinguish an off-staff pitch projection that is visually supported
        by the expected local horizontal strokes from one that is not.
        """
        expected = self._expected_note_y(staff, x, pitch, clef)
        point = staff.get_at(x)
        if expected is None or point is None:
            return 0, 0.0
        y, unit = expected
        top = float(point.y[0])
        bottom = float(point.y[-1])
        if y < top - 0.55 * unit:
            direction = -1
            distance = top - y
            outer = top
        elif y > bottom + 0.55 * unit:
            direction = 1
            distance = y - bottom
            outer = bottom
        else:
            return 0, 1.0

        expected_count = max(0, int(math.floor(distance / unit + 0.25)))
        if expected_count == 0:
            return 0, 1.0

        scores: list[float] = []
        half_width = max(2, int(round(1.45 * unit)))
        center_gap = max(1, int(round(0.38 * unit)))
        band = max(1, int(round(0.20 * unit)))
        cx = int(round(x))
        for index in range(1, expected_count + 1):
            ledger_y = outer + direction * index * unit
            cy = int(round(ledger_y))
            x0 = max(0, cx - half_width)
            x1 = min(self.gray.shape[1], cx + half_width + 1)
            y0 = max(0, cy - band)
            y1 = min(self.gray.shape[0], cy + band + 1)
            if x1 <= x0 or y1 <= y0:
                scores.append(0.0)
                continue
            crop = self.gray[y0:y1, x0:x1] < 175
            local_center = cx - x0
            left_stop = max(0, local_center - center_gap)
            right_start = min(crop.shape[1], local_center + center_gap + 1)
            wings = []
            for wing in (crop[:, :left_stop], crop[:, right_start:]):
                if wing.size == 0:
                    wings.append(0.0)
                else:
                    wings.append(float(np.mean(np.any(wing, axis=0))))
            scores.append(min(1.0, (wings[0] + wings[1]) / 0.7))
        return expected_count, min(scores) if scores else 0.0

    def resolve_note_anchors(
        self,
        measure: list[EncodedSymbol],
        staff: Staff,
        clef: str = "clef_G2",
    ) -> tuple[dict[int, ResolvedNoteAnchor], list[MissingNoteCandidate]]:
        """Resolve final pitched events onto Stage-1 noteheads without hiding missing notes.

        Strong local matches are taken first. If one event remains between already-resolved
        neighbors, a wider order-preserving match may use the clearly closer unused notehead.
        Any still-unused Stage-1 notehead is reported as a missing-note candidate only when
        an existing raw-gap signal independently points to the same location.
        """
        notes = [
            symbol
            for symbol in measure
            if symbol.rhythm.startswith("note")
            and symbol.pitch not in ("_", ".")
            and symbol.page_coordinates is not None
        ]
        if not notes:
            return {}, []

        unit = max(float(staff.average_unit_size), 1e-6)
        stage1 = sorted(staff.get_notes(), key=lambda note: float(note.center[0]))
        left = min(float(symbol.page_coordinates[0]) for symbol in notes) - 4 * unit
        right = max(float(symbol.page_coordinates[0]) for symbol in notes) + 4 * unit
        stage1 = [note for note in stage1 if left <= float(note.center[0]) <= right]

        expected: dict[int, tuple[float, float]] = {}
        for symbol in notes:
            x = float(symbol.page_coordinates[0])
            result = self._expected_note_y(staff, x, symbol.pitch, clef)
            if result is not None:
                expected[id(symbol)] = result

        assigned: dict[int, int] = {}
        used: set[int] = set()
        for event_index, symbol in enumerate(notes):
            if id(symbol) not in expected:
                continue
            x = float(symbol.page_coordinates[0])
            y, local_unit = expected[id(symbol)]
            candidates: list[tuple[float, int]] = []
            for note_index, note in enumerate(stage1):
                if note_index in used:
                    continue
                dx = (float(note.center[0]) - x) / local_unit
                dy = (float(note.center[1]) - y) / local_unit
                if abs(dx) <= 2.2 and abs(dy) <= 1.4:
                    candidates.append((dx * dx + dy * dy, note_index))
            candidates.sort()
            if candidates and (
                len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= 0.30
            ):
                assigned[event_index] = candidates[0][1]
                used.add(candidates[0][1])

        # Recover only isolated unresolved events whose surrounding order is already known.
        for event_index, symbol in enumerate(notes):
            if event_index in assigned or id(symbol) not in expected:
                continue
            previous = next(
                (assigned[index] for index in range(event_index - 1, -1, -1) if index in assigned),
                None,
            )
            following = next(
                (
                    assigned[index]
                    for index in range(event_index + 1, len(notes))
                    if index in assigned
                ),
                None,
            )
            if previous is None or following is None:
                continue
            low = float(stage1[previous].center[0])
            high = float(stage1[following].center[0])
            if high <= low:
                continue
            x = float(symbol.page_coordinates[0])
            y, local_unit = expected[id(symbol)]
            candidates = []
            for note_index, note in enumerate(stage1):
                if note_index in used:
                    continue
                nx = float(note.center[0])
                if not (low < nx < high):
                    continue
                dx = abs(nx - x) / local_unit
                dy = abs(float(note.center[1]) - y) / local_unit
                if dx <= 6.0 and dy <= 1.4:
                    candidates.append((dx, dy, note_index))
            candidates.sort(key=lambda item: (item[0], item[1]))
            if not candidates:
                continue
            if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 1.0:
                continue
            assigned[event_index] = candidates[0][2]
            used.add(candidates[0][2])

        anchors: dict[int, ResolvedNoteAnchor] = {}
        for event_index, symbol in enumerate(notes):
            if id(symbol) not in expected:
                continue
            semantic_x = float(symbol.page_coordinates[0])
            expected_y, _local_unit = expected[id(symbol)]
            if event_index in assigned:
                note = stage1[assigned[event_index]]
                x = float(note.center[0])
                y = float(note.center[1])
                source = "stage1"
            else:
                x = semantic_x
                y = expected_y
                source = "hybrid"
            _ledger_count, ledger_support = self._ledger_line_support(
                staff, x, symbol.pitch, clef
            )
            anchors[id(symbol)] = ResolvedNoteAnchor(x, y, source, ledger_support)

        missing: list[MissingNoteCandidate] = []
        unused = [index for index in range(len(stage1)) if index not in used]
        for event_index, symbol in enumerate(notes[:-1]):
            if symbol.visual_gap_after_score < 0.45 or symbol.visual_gap_after_page_x is None:
                continue
            first = anchors.get(id(symbol))
            second = anchors.get(id(notes[event_index + 1]))
            if first is None or second is None or second.page_x <= first.page_x:
                continue
            gap_x = float(symbol.visual_gap_after_page_x)
            candidates = [
                index
                for index in unused
                if first.page_x < float(stage1[index].center[0]) < second.page_x
                and abs(float(stage1[index].center[0]) - gap_x) <= 1.1 * unit
            ]
            if len(candidates) != 1:
                continue
            note_index = candidates[0]
            note = stage1[note_index]
            missing.append(
                MissingNoteCandidate(
                    float(note.center[0]),
                    float(note.center[1]),
                    float(symbol.visual_gap_after_score),
                )
            )
            unused.remove(note_index)
        return anchors, missing

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

    def _staff_gap_residual(
        self,
        staff: Staff,
        first: tuple[float, float, float],
        second: tuple[float, float, float],
        *,
        edge_exclusion_units: float = 1.5,
    ) -> tuple[float, float | None]:
        """Find a compact raw-page glyph between two decoded notes.

        The purpose is not to classify the glyph.  Staff lines are removed from the raw
        grayscale image, then only compact residual components well inside the note-to-note
        gap are considered.  This gives later profile logic an independent indication that
        Stage-1/TrOMR may have skipped a small symbol.  Meter and musical semantics must still
        determine whether any recovery is justified.
        """
        x1, _y1, unit1 = first
        x2, _y2, unit2 = second
        if x2 <= x1:
            return 0.0, None
        unit = (unit1 + unit2) / 2
        gap = x2 - x1
        if gap < 4 * unit or gap > 18 * unit:
            return 0.0, None

        x_start = int(max(0, x1 + edge_exclusion_units * unit))
        x_stop = int(min(self.gray.shape[1], x2 - edge_exclusion_units * unit))
        if x_stop - x_start < max(2, int(0.75 * unit)):
            return 0.0, None

        midpoint = (x1 + x2) / 2
        point = staff.get_at(midpoint)
        if point is None or len(point.y) != 5:
            return 0.0, None
        y_start = int(max(0, min(point.y) - 0.75 * unit))
        y_stop = int(min(self.gray.shape[0], max(point.y) + 0.75 * unit))
        if y_stop <= y_start:
            return 0.0, None

        crop = self.gray[y_start:y_stop, x_start:x_stop]
        residual = (crop < 175).astype(np.uint8)

        # Remove the five modeled staff lines column-by-column rather than using a broad
        # horizontal opening.  This preserves the non-horizontal fragments of a tiny glyph
        # even when it crosses a staff line.
        for page_x in range(x_start, x_stop):
            staff_point = staff.get_at(float(page_x))
            if staff_point is None:
                continue
            band = max(1, int(round(0.18 * staff_point.average_unit_size)))
            local_x = page_x - x_start
            for line_y in staff_point.y:
                center_y = int(round(line_y)) - y_start
                top = max(0, center_y - band)
                bottom = min(residual.shape[0], center_y + band + 1)
                residual[top:bottom, local_x] = 0

        # Long horizontal leftovers are almost always staff/ledger residue.  A tiny close
        # reconnects glyph fragments cut by the staff-line removal without inventing a large
        # shape across the gap.
        horizontal = cv2.morphologyEx(
            residual,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(2.5 * unit)), 1)),
        )
        residual = cv2.subtract(residual, horizontal)
        residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(residual, 8)
        best_score = 0.0
        best_x: float | None = None
        for index in range(1, count):
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < max(2, int(0.025 * unit * unit)):
                continue
            if not (0.12 * unit <= width <= 1.6 * unit):
                continue
            if not (0.12 * unit <= height <= 2.2 * unit):
                continue
            density = area / float(width * height)
            if density < 0.35:
                continue
            normalized_area = area / float(unit * unit)
            if normalized_area > 1.5:
                continue
            score = min(1.0, normalized_area / 0.08) * min(1.0, density / 0.65)
            if score > best_score:
                best_score = score
                best_x = float(x_start + centroids[index][0])
        return best_score, best_x

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
        event_geometry: dict[int, tuple[float, float, float]] = {}

        for symbol in symbols:
            if symbol.rhythm.startswith("clef"):
                clef = symbol.rhythm
                continue
            if symbol.rhythm.startswith(("note", "rest")) and symbol.coordinates is not None:
                center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
                if len(center) >= 2:
                    page_x, _page_y = staff_to_page((float(center[0]), float(center[1])))
                    staff_point = staff.get_at(float(page_x))
                    if staff_point is not None:
                        event_geometry[id(symbol)] = (
                            float(page_x),
                            0.0,
                            float(staff_point.average_unit_size),
                        )
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
        measure_events: list[EncodedSymbol] = []
        for symbol in symbols:
            if symbol.rhythm.startswith("note"):
                measure_notes.append(symbol)
            if symbol.rhythm.startswith(("note", "rest")):
                measure_events.append(symbol)
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
                    if (
                        "tremolo" in first.articulation.split("_")
                        and "tremolo" in second.articulation.split("_")
                    ):
                        gap_score, gap_x = self._staff_gap_residual(
                            staff, first_geometry, second_geometry
                        )
                        first.visual_gap_after_score = gap_score
                        first.visual_gap_after_page_x = gap_x
                for index in range(len(measure_events) - 1):
                    first = measure_events[index]
                    second = measure_events[index + 1]
                    first_geometry = event_geometry.get(id(first))
                    second_geometry = event_geometry.get(id(second))
                    if first_geometry is None or second_geometry is None:
                        continue
                    if first.visual_gap_after_score > 0:
                        continue
                    gap_score, gap_x = self._staff_gap_residual(
                        staff,
                        first_geometry,
                        second_geometry,
                        edge_exclusion_units=1.0,
                    )
                    first.visual_gap_after_score = gap_score
                    first.visual_gap_after_page_x = gap_x
                if measure_events:
                    last = measure_events[-1]
                    last_geometry = event_geometry.get(id(last))
                    if last_geometry is not None and last.visual_gap_after_score <= 0:
                        boundary_x: float | None = None
                        if symbol.coordinates is not None:
                            center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
                            if len(center) >= 2:
                                boundary_x, _ = staff_to_page(
                                    (float(center[0]), float(center[1]))
                                )
                        if boundary_x is not None:
                            boundary_point = staff.get_at(float(boundary_x))
                            if boundary_point is not None:
                                gap_score, gap_x = self._staff_gap_residual(
                                    staff,
                                    last_geometry,
                                    (
                                        float(boundary_x),
                                        0.0,
                                        float(boundary_point.average_unit_size),
                                    ),
                                    edge_exclusion_units=1.0,
                                )
                                last.visual_gap_after_score = gap_score
                                last.visual_gap_after_page_x = gap_x
                measure_notes = []
                measure_events = []

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
