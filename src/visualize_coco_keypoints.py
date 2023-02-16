import argparse
from pathlib import Path

import cv2
import numpy as np

from src.types.coco_keypoints import CategoryKP, DatasetKP
from src.types.coco_object_detection import RLE
from src.utils.imgs_misc import show_img
from src.utils.misc import clean_print
from src.utils.segmentation_conversions import encoded_rle_to_rle, rle_to_mask


def get_class_from_id(cls_id: int, categories: list[CategoryKP]) -> tuple[str, list[str]]:
    """Returns the string representation of the given class id.

    Args:
        cls_id: The id whose class name should be returned.
        categories: The list of categories from the coco dataset.

    Returns:
        The string representation for the given class id, and a list with the name for each keypoint.

    Raises:
        ValueError: If `cls_id` is not in the categories.
    """
    for cat in categories:
        if cat.id == cls_id:
            return cat.name, cat.keypoints
    raise ValueError(f"There is no class with the id {cls_id}")


def main():
    parser = argparse.ArgumentParser(description=("Tool to visualize coco labels. "
                                                  "Use with 'python -m src.visualize_coco_data <path to image folder> "
                                                  "<path to json annotation file>'"),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images.")
    parser.add_argument("json_path", type=Path, help="Path to the json file with the coco annotations.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    json_path: Path = args.json_path

    rng = np.random.default_rng(42)

    with open(json_path, encoding="utf-8") as data_file:
        dataset = DatasetKP.parse_raw(data_file.read())

    nb_imgs = len(dataset.images)
    for i, img_entry in enumerate(dataset.images, start=1):
        clean_print(f"Showing image: {img_entry.file_name} ({i}/{nb_imgs})", end="\r" if i != nb_imgs else "\n")

        img = cv2.imread(str(data_path / img_entry.file_name))

        img_annotations = [annotation for annotation in dataset.annotations
                           if annotation.image_id == img_entry.id]
        for annotation in img_annotations:
            # Add the segmentation masks
            color = rng.integers(low=0, high=255, size=3, dtype=np.uint8)
            if isinstance(annotation.segmentation, list):
                raise NotImplementedError("Polygon segmentation is not implemented yet.")
            elif isinstance(annotation.segmentation, RLE):
                mask = rle_to_mask(annotation.segmentation.counts, *annotation.segmentation.size)
            else:
                rle = encoded_rle_to_rle(annotation.segmentation.counts)
                mask = rle_to_mask(rle, *annotation.segmentation.size)
            mask = color * np.expand_dims(mask, -1)
            img = np.where(mask, 0.3*mask + 0.7*img, img).astype(np.uint8)

            # Add the keypoints
            color_tuple = tuple(int(c) for c in color)
            _class_name, keypoints_names = get_class_from_id(annotation.category_id, dataset.categories)
            for i in range(0, annotation.num_keypoints):
                x, y, visibility_flag = annotation.keypoints[3*i:3*(i+1)]
                if visibility_flag == 0:
                    continue
                elif visibility_flag == 1:
                    # Should a labeled but non-visible point be displayed ?
                    pass
                keypoints_name = keypoints_names[i]
                cv2.circle(img, (x, y), 5, color_tuple, thickness=cv2.FILLED, lineType=cv2.FILLED)

                _text_width, text_height = cv2.getTextSize(keypoints_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.putText(img, keypoints_name, (x, y + text_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                if i != 0:
                    prev_x, prev_y, prev_visibility_flag = annotation.keypoints[3*(i-1):3*i]
                    if prev_visibility_flag == 0:
                        continue
                    cv2.line(img, (prev_x, prev_y), (x, y), color_tuple, thickness=1)

        show_img(img, img_entry.file_name)


if __name__ == "__main__":
    main()
