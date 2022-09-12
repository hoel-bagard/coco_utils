import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from src.types.coco_types import Annotation, Category, Image
from src.utils.imgs_misc import show_img


def encoded_rle_to_mask(encoded_count_rle: str, height: int, width: int) -> npt.NDArray[np.bool_]:
    """Decode encoded rle segmentation information into a mask.

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
        height: The height of the image.
        width: The with of the image.

    Returns:
        A boolean mask indicating for each pixel whether it belongs to the object or not.
    """
    bytes_rle = str.encode(encoded_count_rle, encoding="ascii")

    m, current_byte_idx, counts = 0, 0, np.zeros(len(encoded_count_rle), dtype=np.uint32)
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
        if m > 2:
            continuous_pixels += counts[m-2]
        counts[m] = continuous_pixels
        m += 1

    mask = np.zeros(height * width, dtype=np.bool_)
    current_value, current_position = 0, 0
    for nb_pixels in counts:
        mask[current_position:current_position+nb_pixels] = current_value
        current_value = 0 if current_value else 1
        current_position += nb_pixels

    # I'm not sure why the transpose, investigate that later. (there's probably something wrong above)
    return mask.reshape(width, height).T


def mask_to_rle(mask: npt.NDArray[np.uint8] | npt.NDArray[np.bool_]) -> list[int]:
    """Convert a mask into its RLE form.

    Args:
        mask: A mask indicating for each pixel whether it belongs to the object or not.

    Returns:
        The RLE list corresponding to the mask.
    """
    previous_value, count, rle = 0, 0, []
    for pixel in mask.flatten():
        if pixel != previous_value:
            rle.append(count)
            count = 0
        else:
            count += 1
        previous_value = pixel
    return rle


def main():
    parser = argparse.ArgumentParser(description=("Tool to visualize coco labels. "
                                                  "Use with 'python -m src.visualize_coco_data <path to image folder> "
                                                  "<path to json annotation file>'"),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images.")
    parser.add_argument("json_path", type=Path, help="Path to the json file with the coco annotations.")
    parser.add_argument("--show_bbox", "-sb", action="store_true", help="Show the bounding boxes.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    json_path: Path = args.json_path
    show_bbox: bool = args.show_bbox

    with open(json_path, 'r', encoding="utf-8") as annotations_file:
        coco_dataset = json.load(annotations_file)

    img_entries: list[Image] = coco_dataset["images"]
    annotations: list[Annotation] = coco_dataset["annotations"]
    categories: list[Category] = coco_dataset["categories"]

    nb_imgs = len(img_entries)
    for i, img_entry in enumerate(img_entries, start=1):
        msg = f"Showing image: {img_entry['file_name']} ({i}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)),
              end='\r' if i != nb_imgs else '\n', flush=True)
        print()
        print()

        img = cv2.imread(str(data_path / img_entry["file_name"]))

        img_annotations = [annotation for annotation in annotations
                           if annotation["image_id"] == img_entry["id"]]
        for annotation in img_annotations:
            # Add the bounding boxes to the image
            if show_bbox:
                top_x, top_y, width, height = annotation["bbox"]
                img = cv2.rectangle(img,
                                    (int(top_x), int(top_y)),
                                    (int(top_x+width), int(top_y+height)),
                                    (255, 0, 0), 5)

            # Add the segmentation masks
            if "segmentation" in annotation:
                segmentation = annotation["segmentation"]
                if isinstance(segmentation, list):
                    # Polygon
                    # Old code snipet.
                    # points: list[dict[str, float]] = region["points"]
                    # pts = np.asarray([[point["x"], point["y"]] for point in points], dtype=np.int32)
                    # pts = pts.reshape((-1, 1, 2))
                    # img = cv2.fillPoly(img, [pts], color)
                    raise NotImplementedError("Polygon segmentation is not implemented yet.")
                else:
                    # Use match/case here ?  (once linters/type checkers support it.)
                    if isinstance(segmentation["counts"], list):
                        raise NotImplementedError("Mask segmentation is not implemented yet.")
                    # Encoded RLE
                    elif isinstance(segmentation["counts"], str):
                        mask = encoded_rle_to_mask(segmentation["counts"], *segmentation["size"])
                    else:
                        raise ValueError(f"Unsupported type for count: {type(segmentation['counts'])}")
                color = np.random.randint(0, high=255, size=3, dtype=np.uint8)
                mask = color * np.expand_dims(mask, -1)
                img = cv2.addWeighted(img, 0.7, mask, 0.3, 0.0)
        show_img(img, img_entry["file_name"])


if __name__ == "__main__":
    main()