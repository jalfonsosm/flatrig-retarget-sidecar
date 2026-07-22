import bpy  # noqa: F401 -- importing bpy registers Blender's bmesh module
import numpy as np
import pytest
from PIL import Image

from flatrig.texture import _build_soft_ring_alpha


def _source_over_alpha(top, bottom):
    return top + bottom * (1.0 - top)


def _two_part_matte(width, *, left, seam_coverage=1.0):
    height = max(16, width // 2)
    seam = width // 2
    outer_min = width // 8
    outer_max = width - outer_min
    ring_width = max(6, width // 8)

    core = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros_like(core)
    if left:
        core[2:-2, outer_min:seam] = 1.0
        core[2:-2, seam] = 0.5
        coverage[2:-2, outer_min : seam + ring_width] = 1.0
    else:
        core[2:-2, seam] = 0.5
        core[2:-2, seam + 1 : outer_max] = 1.0
        coverage[2:-2, seam - ring_width : outer_max] = 1.0
    coverage[2:-2, seam] = seam_coverage
    return core, coverage


def _resize_alpha(alpha, size):
    pixels = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    resized = Image.fromarray(pixels, mode="L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def test_two_soft_ring_mattes_do_not_double_fade_at_shared_seam():
    left_core, left_coverage = _two_part_matte(64, left=True)
    right_core, right_coverage = _two_part_matte(64, left=False)

    naive = _source_over_alpha(right_core, left_core)
    assert naive[10, 32] == pytest.approx(0.75)

    left = _build_soft_ring_alpha(left_core, left_coverage)
    right = _build_soft_ring_alpha(right_core, right_coverage)
    composite = _source_over_alpha(right, left)

    # The canonical core on either side is the opaque underlay.  The farther
    # borrowed ring is not hardened, but source-over never exposes background.
    assert np.min(composite[10, 8:56]) == pytest.approx(1.0)
    assert left[10, 35] < 1.0
    assert right[10, 29] < 1.0
    assert np.all(left <= left_coverage)
    assert np.all(right <= right_coverage)


def test_antialiased_borrowed_coverage_becomes_an_opaque_seam_underlay():
    left_core, left_coverage = _two_part_matte(64, left=True, seam_coverage=0.5)
    right_core, right_coverage = _two_part_matte(64, left=False, seam_coverage=0.5)

    left = _build_soft_ring_alpha(left_core, left_coverage)
    right = _build_soft_ring_alpha(right_core, right_coverage)
    composite = _source_over_alpha(right, left)
    seam = left.shape[1] // 2

    # Clipping an independently antialiased sample to source coverage leaves
    # two 0.5 layers at only 0.75.  Borrowed geometry is instead an opaque
    # underlay inside the cut band, even when that local coverage sample is 0.5.
    assert left[10, seam] >= 0.99
    assert right[10, seam] >= 0.99
    assert composite[10, seam] >= 0.9999
    assert left[10, seam] > left_coverage[10, seam]
    assert right[10, seam] > right_coverage[10, seam]

    # Hardening alpha never invents support where the source render had none.
    assert np.count_nonzero(left[left_coverage == 0.0]) == 0
    assert np.count_nonzero(right[right_coverage == 0.0]) == 0


def test_soft_ring_preserves_outer_silhouette_and_true_holes():
    core = np.zeros((28, 40), dtype=np.float32)
    coverage = np.zeros_like(core)
    core[5:23, 5:20] = 1.0
    coverage[5:23, 5:31] = 1.0

    # An antialiased outer silhouette owned by the core must not be hardened.
    core[4, 5:20] = 0.35
    coverage[4, 5:20] = 0.35
    core[23, 5:20] = 0.65
    coverage[23, 5:20] = 0.65

    # A genuine coverage hole remains a hole even though it lies within the
    # dilation distance of opaque pixels.
    core[12:15, 16:19] = 0.0
    coverage[12:15, 16:19] = 0.0

    result = _build_soft_ring_alpha(core, coverage)

    # Both antialiased exterior rows are adjacent to the borrowed ring in image
    # space, so this proves the boundary classification -- rather than mere
    # distance from the ring -- preserves them exactly.
    assert np.array_equal(result[4, 5:20], coverage[4, 5:20])
    assert np.array_equal(result[23, 5:20], coverage[23, 5:20])
    assert np.count_nonzero(result[12:15, 16:19]) == 0
    assert np.array_equal(result[coverage == 0.0], coverage[coverage == 0.0])


@pytest.mark.parametrize(
    ("left_size", "right_size"),
    [(64, 128), (128, 64), (64, 256), (256, 64)],
)
def test_opaque_underlay_survives_different_page_scales(left_size, right_size):
    left_core, left_coverage = _two_part_matte(left_size, left=True)
    right_core, right_coverage = _two_part_matte(right_size, left=False)
    left = _build_soft_ring_alpha(left_core, left_coverage)
    right = _build_soft_ring_alpha(right_core, right_coverage)

    common_size = (512, 128)
    left = _resize_alpha(left, common_size)
    right = _resize_alpha(right, common_size)
    composite = _source_over_alpha(right, left)

    # Compare in a narrow world-space corridor around the common cut.  The
    # sprites deliberately use up to a 4x difference in pixel density.
    seam = common_size[0] // 2
    assert np.min(composite[common_size[1] // 2, seam - 12 : seam + 12]) == pytest.approx(1.0)
