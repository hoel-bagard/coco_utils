"""Tests for the segmentation conversion functions."""
# import pytest
import numpy as np

from src.utils.segmentation_conversions import mask_to_rle


# @pytest.mark.parametrize("width, height, etc...",
#                          [(),
#                           (),
#                           ])
def test_mask_to_rle():
    square_img = np.zeros((40, 40), dtype=np.uint8)
    square_img[5:10, 6:11] = 255
    assert mask_to_rle(square_img) == [245, 5, 35, 5, 35, 5, 35, 5, 35, 5, 1190]
