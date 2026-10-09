import unittest

from homr.staff_parsing import _select_stage1_barline_geometry


class _Point:
    y = [100.0, 108.0, 116.0, 124.0, 132.0]
    average_unit_size = 8.0


class _Staff:
    average_unit_size = 8.0

    def get_at(self, _x: float) -> _Point:
        return _Point()


class _Box:
    def __init__(self, x: float, y: float, height: float) -> None:
        self.center = (x, y)
        self.size = (3.0, height)


class TestBarlineGeometry(unittest.TestCase):
    def test_prefers_staff_spanning_box_when_only_slightly_farther(self) -> None:
        stem_like = _Box(119.0, 127.0, 24.0)
        barline = _Box(126.0, 116.0, 32.0)

        selected = _select_stage1_barline_geometry(_Staff(), 121.0, [stem_like, barline])

        self.assertIs(selected, barline)

    def test_does_not_snap_false_token_to_distant_real_barline(self) -> None:
        local_false_mark = _Box(100.0, 127.0, 24.0)
        distant_barline = _Box(175.0, 116.0, 32.0)

        selected = _select_stage1_barline_geometry(
            _Staff(), 100.0, [local_false_mark, distant_barline]
        )

        self.assertIs(selected, local_false_mark)


if __name__ == "__main__":
    unittest.main()
