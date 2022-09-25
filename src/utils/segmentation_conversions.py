import numpy as np
import numpy.typing as npt

from src.types.img_types import Tmask


def encoded_rle_to_rle(encoded_count_rle: str) -> npt.NDArray[np.uint32]:
    """Decode encoded rle segmentation information into a rle.

    See the (hard to read) implementation:
        https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c#L218
        https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/_mask.pyx#L145

    LEB128 wikipedia article: https://en.wikipedia.org/wiki/LEB128#Decode_signed_integer
    It is similar to LEB128, but here shift is incremented by 5 instead of 7 because the implementation uses
    6 bits per byte instead of 8. (no idea why, I guess it's more efficient for the COCO dataset?)

    I'm reimplementing it in python here half by curiosity and half for debugging purposes.
    Performance is obviously worse.

    Args:
        encoded_count_rle: The encoded string from the annotations file.

    Returns:
        The decoded RLE list.
    """
    bytes_rle = str.encode(encoded_count_rle, encoding="ascii")

    # Get the RLE from the encoded RLE.
    current_count_idx, current_byte_idx, counts = 0, 0, np.zeros(len(encoded_count_rle), dtype=np.uint32)
    while current_byte_idx < len(bytes_rle):
        continuous_pixels, shift, high_order_bit = 0, 0, 1

        # When the high order bit of a byte becomes 0, we have decoded the integer and can move on to the next one.
        while high_order_bit:
            byte = bytes_rle[current_byte_idx] - 48  # The encoding uses the ascii chars 48-111.
            # 0x1f is 31, i.e. 001111 --> Here we select the first four bits of the byte.
            continuous_pixels |= (byte & 0x1f) << shift
            # 0x20 is 32 as int, i.e. 2**5, i.e 010000 --> Here we select the fifth bit of the byte.
            high_order_bit = byte & 0x20
            current_byte_idx += 1
            shift += 5
            if (not high_order_bit) and (byte & 0x10):  # 0x10 is 16 as int, i.e. 1000
                continuous_pixels |= (~0 << shift)

        if current_count_idx > 2:
            # My hypothesis as to what is happening here, is that most objects are going to be somewhat
            # 'vertically convex' (i.e. have only one continuous run per line).
            # In which case, the next 'row' of black/white pixels is going to be similar to the one preceding it.
            # Therefore, by have the continuous count of pixels be an offset of the one preceding it, we can have it be
            # a smaller int and therefore use less bits to encode it.
            continuous_pixels += counts[current_count_idx-2]
        counts[current_count_idx] = continuous_pixels
        current_count_idx += 1
    return counts


def rle_to_mask(rle: list[int] | npt.NDArray[np.uint32], height: int, width: int) -> npt.NDArray[np.bool_]:
    """Converts a RLE to its uncompressed mask.

    Args:
        rle: The RLE list corresponding to the mask.
        height: The height of the image.
        width: The with of the image.

    Returns:
        A boolean mask indicating for each pixel whether it belongs to the object or not.
    """
    # Construct the mask from the RLE.
    # (Could do it fully in numpy by using np.repeat and starting from an array of alternating 0s and 1)
    mask = np.zeros(height * width, dtype=np.bool_)
    current_value, current_position = 0, 0
    for nb_pixels in rle:
        mask[current_position:current_position+nb_pixels] = current_value
        current_value = 0 if current_value else 1
        current_position += nb_pixels

    return mask.reshape((height, width), order="F")


def mask_to_rle(mask: npt.NDArray[Tmask]) -> list[int]:
    """Convert a mask into its RLE form.

    Args:
        mask: A mask indicating for each pixel whether it belongs to the object or not.

    Returns:
        The RLE list corresponding to the mask.
    """
    if mask.shape[-1] == 1:
        mask = np.squeeze(mask)
    assert mask.ndim == 2, "Mask must not be RGB."
    previous_value, count = 0, 0
    rle: list[int] = []
    for pixel in mask.ravel(order="F"):
        if pixel != previous_value:
            rle.append(count)
            previous_value, count = pixel, 0
        count += 1
    rle.append(count)
    return rle
