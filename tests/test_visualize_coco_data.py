import pytest
from pycocotools import mask as mask_utils



@pytest.fixture
def tuple_sample() -> tuple[float, float, float]:
    return 1, 2, 3


@pytest.fixture
def vector_sample(tuple_sample) -> Vector:
    return Vector(*tuple_sample)

encoded_rle = "iX13a>i0^O:F9H7I8J4K6J6K4L4L4M3L3N3L3N3L3N3M2M4M2N3M2N2N2N2N2N2N2N2O1N2N2N1O2N2O1N2N101N2N101N2N2O0O2O0O2O1N101N101N101O0O2O001N101N10001N10001N10001O0O10001O0O1000001O000000001O0000000000000O1000000000000000000001O00000000000000000O2O00000O2O00000O2O00001N10001N101O0O101N101N101O0O2O0O1O2O1N101N101N2N101N2N101N2N2N1O2N2N2O1N2N2N2N2M3N2N2N2N3M2N2M3N3M2M4M2M4L3N3L4L4L4L5K4K6J6J7H8H9E?^OcVa6"
height, width = 480, 640

segmentation = {"size": [height, width], "counts": rle}
mask = mask_utils.decode([segmentation])
mask = np.squeeze(mask)
mask_coco = 255 * mask.astype(np.uint8)

# rle_coco = mask_to_rle(mask)
# print(f"{np.asarray(rle_coco)=}")


mask = decode_rle(rle, height, width)
mask = 255 * mask.astype(np.uint8)

equal = np.all(mask_coco == mask)
print(equal)

@pytest.mark.pycocotools
class TestDuration:
    @pytest.mark.parametrize("duration_str, duration_value",
                             [("10s", 10),
                              ("60s", 60)])
    def test_seconds(self, duration_str: str, duration_value: int):
        assert countdown.duration(duration_str) == duration_value
