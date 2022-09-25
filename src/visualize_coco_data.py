import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.types.coco_types import Annotation, Category, Image
from src.utils.imgs_misc import show_img
from src.utils.misc import clean_print
from src.utils.segmentation_conversions import encoded_rle_to_rle, rle_to_mask


def get_class_from_id(cls_id: int, categories: list[Category]) -> str:
    """Returns the string representation of the given class id.

    Args:
        cls_id: The id whose class name should be returned.
        categories: The list of categories from the coco dataset.

    Returns:
        The string representation for the given class id.

    Raises:
        ValueError: If `cls_id` is not in the categories.
    """
    for cat in categories:
        if cat["id"] == cls_id:
            return cat["name"]
    raise ValueError(f"There is no class with the id {cls_id}")


def main():
    parser = argparse.ArgumentParser(description=("Tool to visualize coco labels. "
                                                  "Use with 'python -m src.visualize_coco_data <path to image folder> "
                                                  "<path to json annotation file>'"),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images.")
    parser.add_argument("json_path", type=Path, help="Path to the json file with the coco annotations.")
    parser.add_argument("--show_bbox", "-sb", action="store_true", help="Show the bounding boxes.")
    parser.add_argument("--show_individual_masks", "-sm", action="store_true", help="Show the masks one by one.")
    parser.add_argument("--image_name", "-i", type=str, default=None,
                        help="If given, only that image will be displayed.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    json_path: Path = args.json_path
    show_bbox: bool = args.show_bbox
    show_individual_masks: bool = args.show_individual_masks
    img_name: Optional[str] = args.image_name

    with open(json_path, "r", encoding="utf-8") as annotations_file:
        coco_dataset = json.load(annotations_file)

    img_entries: list[Image] = coco_dataset["images"]
    annotations: list[Annotation] = coco_dataset["annotations"]
    categories: list[Category] = coco_dataset["categories"]

    bbox_thickness = 2
    nb_imgs = len(img_entries)
    for i, img_entry in enumerate(img_entries, start=1):
        clean_print(f"Showing image: {img_entry['file_name']} ({i}/{nb_imgs})", end="\r" if i != nb_imgs else "\n")

        if img_name is not None and img_name != img_entry["file_name"]:
            continue

        img = cv2.imread(str(data_path / img_entry["file_name"]))

        img_annotations = [annotation for annotation in annotations
                           if annotation["image_id"] == img_entry["id"]]
        for annotation in img_annotations:
            color = np.random.randint(0, high=255, size=3, dtype=np.uint8)
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
                    if isinstance(segmentation["counts"], list):
                        mask = rle_to_mask(segmentation["counts"], *segmentation["size"])
                    else:
                        rle = encoded_rle_to_rle(segmentation["counts"])
                        mask = rle_to_mask(rle, *segmentation["size"])
                mask = color * np.expand_dims(mask, -1)
                if show_individual_masks:
                    show_img(mask, get_class_from_id(annotation["category_id"], categories))
                img = cv2.addWeighted(img, 0.7, mask, 0.3, 0.0)

            # Add the bounding boxes to the image
            if show_bbox:
                top_x, top_y, width, height = annotation["bbox"]
                top_x, top_y, width, height = int(top_x), int(top_y), int(width), int(height)
                cv2.rectangle(img,
                              (top_x, top_y),
                              (top_x+width, top_y+height),
                              tuple(int(c) for c in color), bbox_thickness)

                # Add class
                class_name = get_class_from_id(annotation["category_id"], categories)
                text_width, text_height = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(img,
                              (top_x+bbox_thickness, top_y+bbox_thickness),
                              (top_x+4*bbox_thickness+text_width, top_y+4*bbox_thickness+text_height),
                              (0, 0, 0), -1)
                cv2.putText(img, class_name, (top_x+2*bbox_thickness, top_y+2*bbox_thickness+text_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        show_img(img, img_entry["file_name"])


if __name__ == "__main__":
    main()
