import math

import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import RotatedBoundingBox
from homr.debug import Debug
from homr.image_utils import crop_image_and_return_new_top
from homr.model import MultiStaff, Staff
from homr.point_mapping import PointMapping, chain, identity, undo_crop, undo_resize
from homr.simple_logging import eprint
from homr.staff_dewarping import StaffDewarping, dewarp_staff_image
from homr.staff_parsing_tromr import parse_staff_tromr
from homr.staff_regions import StaffRegions
from homr.transformer.configs import Config, default_config
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols
from homr.type_definitions import NDArray
from homr.visual_evidence import VisualEvidencePage


def _flatten_staffs(staffs: list[MultiStaff]) -> list[Staff]:
    return [s for multi_staff in staffs for s in multi_staff.staffs]


def _regroup_by_period(
    flat_staffs: list[Staff], period: int, front_trim: int, back_trim: int
) -> list[MultiStaff]:
    core = flat_staffs[front_trim : len(flat_staffs) - back_trim]
    return [MultiStaff(core[i : i + period], []) for i in range(0, len(core), period)]


def _find_periodic_core(flat_staffs: list[Staff]) -> tuple[int, int, int] | None:
    """
    Find a repeating sequence of staff layouts among individual staffs, e.g.
    a solo staff followed by a piano grand staff (2 staffs), repeated for
    every system in a vocal score with piano accompaniment.

    We work on the flattened sequence of raw staffs rather than on the
    MultiStaff rows produced upstream, because that upstream grouping is
    itself only a heuristic (staffs sharing a bar line or clef get merged
    into one row) and can be inconsistent across a page: the same kind of
    solo-staff-plus-grand-staff pair might end up pre-merged into one row for
    one system and left as two separate rows for another, purely because of
    how cleanly a bar line lined up. Searching row-by-row would then see two
    different "shapes" for what is structurally the same repeating pattern.
    Working on individual staffs sidesteps that inconsistency entirely.

    A system right at the start or end of the page can break the pattern on
    its own without invalidating it: an introduction or coda system with a
    genuinely different layout, or simply the most poorly detected staff on
    the page. We therefore allow trimming up to one period's worth of staffs
    from either edge before requiring the remainder to tile exactly. We
    never trim from the middle of the page: a mismatch there is a detection
    problem to fix upstream, not something to paper over here.

    Returns (period, front_trim, back_trim) for the smallest total trim and,
    among ties, the smallest period -- so an already-uniform page (period 1,
    no trim) is always preferred when it fits, and we never discard more of
    the page than necessary. Returns None if no repeating core of at least
    two full cycles can be found.
    """
    layout = [s.is_grandstaff for s in flat_staffs]
    n = len(layout)
    best: tuple[int, int, int, int] | None = None
    for period in range(1, n // 2 + 1):
        for front_trim in range(period + 1):
            for back_trim in range(period + 1):
                core = layout[front_trim : n - back_trim]
                if len(core) < 2 * period or len(core) % period != 0:
                    continue
                rows = [tuple(core[i : i + period]) for i in range(0, len(core), period)]
                if not all(row == rows[0] for row in rows):
                    continue
                candidate = (front_trim + back_trim, period, front_trim, back_trim)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
    if best is None:
        return None
    _, period, front_trim, back_trim = best
    return period, front_trim, back_trim


def _ensure_same_number_of_staffs(staffs: list[MultiStaff]) -> list[MultiStaff]:
    """
    If every system already has the same number of *more than one* staff, trust that
    directly rather than re-deriving it via _find_periodic_core. That function's signature
    is each flat staff's is_grandstaff flag, which is a fine way to tell "solo staff" from
    "piano grand staff" apart when the two are pre-merged inconsistently across the page
    (see its own docstring) - but it carries zero information when a page has N genuinely
    independent, same-type staffs per system and none of them are a grand staff (e.g. a
    string quartet): the flattened signature is then a constant sequence, which trivially -
    and wrongly - satisfies period=1, collapsing all N voices into one. Checking uniformity
    upfront on the untouched, already-correct per-system grouping sidesteps that degenerate
    case entirely.

    Restricted to row length > 1: a page where every row is already a single raw staff
    (nothing grouped yet, e.g. a solo-plus-piano page where no bar line happened to
    pre-merge any pair) is *also* uniform by this same measure, but there _find_periodic_
    core is exactly what's needed to discover the real, larger repeating pattern from
    scratch - that's the case this function was originally written for, and it is never
    already uniform at a row length above 1.
    """
    row_lengths = {len(multi_staff.staffs) for multi_staff in staffs}
    if len(row_lengths) == 1 and next(iter(row_lengths)) > 1:
        return staffs
    flat_staffs = _flatten_staffs(staffs)
    core = _find_periodic_core(flat_staffs)
    if core is not None:
        period, front_trim, back_trim = core
        if front_trim > 0:
            eprint(
                f"Removing the first {front_trim} staff(s), as they don't fit "
                "the staff layout the rest of the page repeats"
            )
        if back_trim > 0:
            eprint(
                f"Removing the last {back_trim} staff(s), as they don't fit "
                "the staff layout the rest of the page repeats"
            )
        if period > 1:
            eprint(
                "Systems repeat every",
                period,
                "staffs with a different layout each time, combining them into one row",
            )
        return _regroup_by_period(flat_staffs, period, front_trim, back_trim)
    result: list[MultiStaff] = []
    for staff in staffs:
        result.extend(staff.break_apart())
    return sorted(result, key=lambda s: s.staffs[0].min_y)


def _get_number_of_voices(staffs: list[MultiStaff]) -> int:
    return len(staffs[0].staffs)


tr_omr_max_height = default_config.max_height
tr_omr_max_width = default_config.max_width


def get_tr_omr_canvas_size(
    image_shape: tuple[int, ...], margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    tr_omr_ratio = float(tr_omr_max_height_with_margin) / tr_omr_max_width
    height, width = image_shape[:2]

    # Calculate the new size such that it fits exactly into the
    # tr_omr_max_height and tr_omr_max_width
    # while maintaining the aspect ratio of height and width.

    if height / width > tr_omr_ratio:
        # The height is the limiting factor.
        new_shape = [
            int(width / height * tr_omr_max_height_with_margin),
            tr_omr_max_height_with_margin,
        ]
    else:
        # The width is the limiting factor.
        new_shape = [tr_omr_max_width, int(height / width * tr_omr_max_width)]
    return np.array(new_shape)


def _canvas_y_offset(resized_height: int, margin_top: int = 0, margin_bottom: int = 0) -> int:
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    return (tr_omr_max_height_with_margin - resized_height) // 2 + margin_top


def center_image_on_canvas(
    image: NDArray, canvas_size: NDArray, margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    is_grayscale = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)

    resized = cv2.resize(image, canvas_size)  # type: ignore

    if is_grayscale:
        new_image = np.full(
            (tr_omr_max_height, tr_omr_max_width),
            255,
            dtype=np.uint8,
        )
    else:
        new_image = np.full(
            (tr_omr_max_height, tr_omr_max_width, 3),
            255,
            dtype=np.uint8,
        )

    x_offset = 0
    y_offset = _canvas_y_offset(resized.shape[0], margin_top, margin_bottom)

    new_image[
        y_offset : y_offset + resized.shape[0],
        x_offset : x_offset + resized.shape[1],
    ] = resized

    return new_image


def add_image_into_tr_omr_canvas(image: NDArray) -> NDArray:
    new_shape = get_tr_omr_canvas_size(image.shape)
    new_image = center_image_on_canvas(image, new_shape)
    return new_image


def remove_black_contours_at_edges_of_image(gray: NDArray, unit_size: float) -> NDArray:
    _, thresh = cv2.threshold(gray, 97, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh
    contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    threshold = constants.black_spot_removal_threshold(unit_size)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < threshold or h < threshold:
            continue
        is_at_edge_of_image = x == 0 or y == 0 or x + w == gray.shape[1] or y + h == gray.shape[0]
        if not is_at_edge_of_image:
            continue
        average_gray_intensity = 127
        is_mostly_dark = np.mean(thresh[y : y + h, x : x + w]) < average_gray_intensity
        if is_mostly_dark:
            continue
        gray[y : y + h, x : x + w] = 255
    return gray


def _calculate_region(staff: Staff, regions: StaffRegions) -> NDArray:
    x_min = staff.min_x - 2 * staff.average_unit_size
    x_max = staff.max_x + 2 * staff.average_unit_size
    y_min = max(
        staff.min_y - 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_above(staff.min_y),
    )
    y_max = min(
        staff.max_y + 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_below(staff.max_y),
    )
    return np.array([int(x_min), int(y_min), int(x_max), int(y_max)])


def prepare_staff_image(
    debug: Debug, index: int, staff: Staff, staff_image: NDArray, regions: StaffRegions
) -> tuple[NDArray, Staff, PointMapping]:
    """
    Returns the staff image for the transformer, the staff in the coordinates of that image
    and a mapping from the coordinates of that image back to the coordinates of the page.
    """
    region = _calculate_region(staff, regions)
    image_dimensions = get_tr_omr_canvas_size(
        (int(region[3] - region[1]), int(region[2] - region[0]))
    )
    scaling_factor = image_dimensions[1] / (region[3] - region[1])
    page_shape = staff_image.shape
    staff_image = cv2.resize(
        staff_image,
        (int(staff_image.shape[1] * scaling_factor), int(staff_image.shape[0] * scaling_factor)),
    )
    undo_scaling = undo_resize(page_shape, staff_image.shape)
    region = np.round(region * scaling_factor)
    eprint("Dewarping staff", index)
    region_step1 = np.array(region) + np.array([-10, -50, 10, 50])
    staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step1)
    undo_crop1 = undo_crop(*top_left)
    region_step2 = np.array(region) - np.array([*top_left, *top_left])
    top_left = top_left / scaling_factor
    staff = _dewarp_staff(staff, None, top_left, scaling_factor)
    dewarp = dewarp_staff_image(staff_image, staff, index, debug)
    staff_image = dewarp.dewarp(staff_image)
    staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step2)
    undo_crop2 = undo_crop(*top_left)
    scaling_factor = 1

    eprint("Dewarping staff", index, "done")

    staff_image = remove_black_contours_at_edges_of_image(staff_image, staff.average_unit_size)
    before_canvas_shape = staff_image.shape
    staff_image = center_image_on_canvas(staff_image, image_dimensions)
    canvas_content_shape = (int(image_dimensions[1]), int(image_dimensions[0]))
    to_page = chain(
        undo_crop(0, -_canvas_y_offset(canvas_content_shape[0])),
        undo_resize(before_canvas_shape, canvas_content_shape),
        undo_crop2,
        dewarp.undewarp_point,
        undo_crop1,
        undo_scaling,
    )
    debug.write_image_with_fixed_suffix(f"_staff-{index}_input.jpg", staff_image)
    if debug.debug:
        transformed_staff = _dewarp_staff(staff, dewarp, top_left, scaling_factor)
        transformed_staff_image = staff_image.copy()
        for symbol in transformed_staff.symbols:
            center = symbol.center
            cv2.circle(transformed_staff_image, (int(center[0]), int(center[1])), 5, (0, 0, 255))
            cv2.putText(
                transformed_staff_image,
                type(symbol).__name__,
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )
        debug.write_image_with_fixed_suffix(
            f"_staff-{index}_debug_annotated.jpg", transformed_staff_image
        )
    return staff_image, staff, to_page


def _dewarp_staff(
    staff: Staff, dewarp: StaffDewarping | None, region: NDArray, scaling: float
) -> Staff:
    """
    Applies the same transformation on the staff coordinates as we did on the image.
    """

    def transform_coordinates(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        x -= region[0]
        y -= region[1]
        if dewarp is not None:
            x, y = dewarp.dewarp_point((x, y))
        x = x * scaling
        y = y * scaling
        return x, y

    return staff.transform_coordinates(transform_coordinates)


def _get_symbol_center(symbol: EncodedSymbol) -> tuple[float, float] | None:
    if symbol.coordinates is None or symbol.rhythm.startswith("chord"):
        return None
    center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
    if len(center) < 2 or math.isnan(center[0]) or math.isnan(center[1]):  # noqa: PLR2004
        return None
    return float(center[0]), float(center[1])


def _barline_geometry_fitness(staff: Staff, box: RotatedBoundingBox) -> float:
    """Prefer vertical boxes that actually span and center on the staff.

    TrOMR's barline token localizes only approximately. Stage-1 can therefore expose
    several nearby vertical boxes, including note stems. A real barline normally covers
    almost the whole five-line staff and is vertically centered on it; stems commonly do
    neither. Keep this as an image-geometry score only -- musical/system topology remains
    the responsibility of the later profile reconciliation.
    """
    point = staff.get_at(float(box.center[0]))
    if point is None:
        return float("-inf")
    staff_top = float(point.y[0])
    staff_bottom = float(point.y[-1])
    staff_span = max(staff_bottom - staff_top, 1e-6)
    staff_mid = (staff_top + staff_bottom) / 2
    box_top = float(box.center[1] - box.size[1] / 2)
    box_bottom = float(box.center[1] + box.size[1] / 2)
    overlap = max(0.0, min(box_bottom, staff_bottom) - max(box_top, staff_top))
    coverage = overlap / staff_span
    center_offset = abs(float(box.center[1]) - staff_mid) / (staff_span / 2)
    protrusion = (
        max(0.0, staff_top - box_top) + max(0.0, box_bottom - staff_bottom)
    ) / staff_span
    return coverage - 0.30 * center_offset - 0.20 * protrusion


def _select_stage1_barline_geometry(
    staff: Staff,
    semantic_x: float,
    candidates: list[RotatedBoundingBox],
) -> RotatedBoundingBox | None:
    """Choose a clearly more barline-like Stage-1 box without making long jumps.

    Preserve the old nearest-x behavior unless another nearby candidate has materially
    stronger staff-spanning geometry. The distance guard is important: an isolated false
    TrOMR barline must not be snapped onto a real boundary several symbols away.
    """
    if not candidates:
        return None
    unit = max(float(staff.average_unit_size), 1e-6)
    nearby = [box for box in candidates if abs(float(box.center[0]) - semantic_x) <= 12 * unit]
    if not nearby:
        return None
    nearest = min(nearby, key=lambda box: abs(float(box.center[0]) - semantic_x))
    best = max(
        nearby,
        key=lambda box: (
            _barline_geometry_fitness(staff, box),
            -abs(float(box.center[0]) - semantic_x),
        ),
    )
    nearest_fitness = _barline_geometry_fitness(staff, nearest)
    best_fitness = _barline_geometry_fitness(staff, best)
    nearest_distance = abs(float(nearest.center[0]) - semantic_x) / unit
    best_distance = abs(float(best.center[0]) - semantic_x) / unit
    if (
        best is not nearest
        and best_fitness >= 0.72
        and best_fitness >= nearest_fitness + 0.15
        and best_distance <= nearest_distance + 2.0
    ):
        return best
    return nearest


def parse_staff_image(
    debug: Debug,
    index: int,
    staff: Staff,
    image: NDArray,
    regions: StaffRegions,
    config: Config,
    page_to_input_image: PointMapping = identity,
    visual_evidence: VisualEvidencePage | None = None,
    bar_line_boxes: list[RotatedBoundingBox] | None = None,
) -> list[EncodedSymbol]:
    page_staff = staff
    staff_image, transformed_staff, staff_to_page = prepare_staff_image(
        debug, index, staff, image, regions=regions
    )
    eprint("Running TrOmr inference on staff image", index)
    result = parse_staff_tromr(staff_image=staff_image, staff=transformed_staff, config=config)

    stage1_barlines: list[RotatedBoundingBox] = []
    if bar_line_boxes is not None:
        for box in bar_line_boxes:
            point = page_staff.get_at(box.center[0])
            if point is None:
                continue
            tolerance = 1.6 * point.average_unit_size
            if point.y[0] - tolerance <= box.center[1] <= point.y[-1] + tolerance:
                stage1_barlines.append(box)

    if visual_evidence is not None:
        visual_evidence.annotate_symbols(result, page_staff, staff_to_page)
    for symbol in result:
        center = _get_symbol_center(symbol)
        if center is not None:
            page_center = staff_to_page(center)
            symbol.page_coordinates = page_center
            symbol.image_coordinates = page_to_input_image(page_center)
            if "barline" in symbol.rhythm or "repeat" in symbol.rhythm:
                geometry = _select_stage1_barline_geometry(
                    page_staff, float(page_center[0]), stage1_barlines
                )
                if geometry is not None:
                    symbol.barline_geometry_page_x = float(geometry.center[0])
                    symbol.barline_geometry_unit_size = float(page_staff.average_unit_size)
    if debug.debug:
        result_image = staff_image.copy()
        for i, symbol in enumerate(result):
            center = _get_symbol_center(symbol)
            if center is None:
                continue
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(result_image, center_int, 5, color=(0, 0, 255), thickness=2)
            cv2.putText(
                result_image,
                str(i) + ": " + symbol.rhythm,
                (center_int[0], center_int[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )

        debug.write_image_with_fixed_suffix(f"_staff-{index}_output.jpg", result_image)
    return result


def parse_staffs(
    debug: Debug,
    staffs: list[MultiStaff],
    image: NDArray,
    config: Config,
    selected_staff: int = -1,
    page_to_input_image: PointMapping = identity,
    notehead_mask: NDArray | None = None,
    bar_line_boxes: list[RotatedBoundingBox] | None = None,
) -> list[list[EncodedSymbol]]:
    """
    Dewarps each staff and then runs it through an algorithm which extracts
    the rhythm and pitch information.

    page_to_input_image maps the coordinates of image back to the image the user provided,
    it's used to fill EncodedSymbol.image_coordinates.
    """
    staffs = _ensure_same_number_of_staffs(staffs)
    # For simplicity we call every staff in a multi staff a voice,
    # even if it's part of a grand staff.
    number_of_voices = _get_number_of_voices(staffs)
    i = 0
    voices = []
    regions = StaffRegions(staffs)
    visual_evidence = VisualEvidencePage(image, notehead_mask) if notehead_mask is not None else None
    for voice in range(number_of_voices):
        staffs_for_voice = [staff.staffs[voice] for staff in staffs]
        result_for_voice = []
        for staff_index, staff in enumerate(staffs_for_voice):
            if selected_staff >= 0 and staff_index != selected_staff:
                eprint("Ignoring staff due to selected_staff argument", i)
                i += 1
                continue
            result_staff = parse_staff_image(
                debug,
                i,
                staff,
                image,
                regions,
                config,
                page_to_input_image,
                visual_evidence,
                bar_line_boxes,
            )
            if len(result_staff) == 0:
                eprint("Skipping empty staff", i)
                i += 1
                continue
            result_staff.append(EncodedSymbol("newline"))
            result_for_voice.extend(result_staff)
            i += 1

        voices.append(remove_duplicated_symbols(result_for_voice))
    return voices
