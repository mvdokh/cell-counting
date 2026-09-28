from pathlib import Path

import numpy as np
import pytest

from slicereg.detect import detect_sections
from slicereg.io import make_thumbnail, open_slide

EXAMPLE = Path(r"C:\Users\marti\Desktop\4x_all_1.tif")


def test_synthetic_grid_with_debris():
    img = np.full((400, 900, 3), 10, np.uint8)
    for r in range(2):
        for c in range(3):
            y, x = 40 + r * 180, 40 + c * 280
            img[y:y + 120, x:x + 200] = 150
    img[10:14, 880:884] = 200
    boxes, _, dark = detect_sections(img)
    assert dark and len(boxes) == 6
    assert [b[0] < boxes[i + 1][0] for i, b in enumerate(boxes[:2])] == [True, True]
    assert boxes[3][1] > boxes[0][3] - 10


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example slide not available")
def test_example_slide_has_24_sections():
    boxes, _, _ = detect_sections(make_thumbnail(open_slide(EXAMPLE)))
    assert len(boxes) == 24
