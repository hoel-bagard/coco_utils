import argparse
from pathlib import Path
from typing import Optional

import pycocotools.mask
import pycocotools
import cv2
import numpy as np
from coco_types import DatasetKP, COCO_RLE

from src.utils.imgs_misc import show_img
from src.utils.misc import clean_print


def main():
    parser = argparse.ArgumentParser(description=("Tool to visualize coco labels. "
                                                  "Use with 'python -m src.visualize_coco_data <path to image folder> "
                                                  "<path to json annotation file>'"),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--annotations_path", "-a", required=True, type=Path,
                        help="Path to the json file with the coco annotations.")
    parser.add_argument("--imgs_folder_path", "-i", required=True, type=Path, help="Path to the directory with the images.")
    parser.add_argument("--show_bbox", "-sb", action="store_true", help="Show the bounding boxes.")
    parser.add_argument("--image_name", "-n", type=str, default=None,
                        help="If given, only that image will be displayed.")
    args = parser.parse_args()

    imgs_folder_path: Path = args.imgs_folder_path
    annotations_path: Path = args.annotations_path
    show_bbox: bool = args.show_bbox
    img_name: Optional[str] = args.image_name

    with annotations_path.open("r", encoding="utf-8") as annotations_file:
        dataset = DatasetKP.parse_raw(annotations_file.read())

    categories = {cat.id: cat.name for cat in dataset.categories}

    bbox_thickness = 2
    nb_imgs = len(dataset.images)
    for i, img_entry in enumerate(dataset.images, start=1):
        clean_print(f"Showing image: {img_entry.file_name} ({i}/{nb_imgs})", end="\r" if i != nb_imgs else "\n")

        if img_name is not None and img_name != img_entry.file_name:
            continue

        img = cv2.imread(str(imgs_folder_path / img_entry.file_name))

        img_annotations = [ann for ann in dataset.annotations if ann.image_id == img_entry.id]
        for ann in img_annotations:
            color = np.random.randint(0, high=255, size=3, dtype=np.uint8)
            # Add the segmentation masks
            assert isinstance(ann.segmentation, COCO_RLE)
            mask = pycocotools.mask.decode({"size": ann.segmentation.size, "counts": ann.segmentation.counts})
            mask = color * np.expand_dims(mask, -1)
            img = np.where(mask, 0.3*mask + 0.7*img, img).astype(np.uint8)

            # Add the bounding boxes to the image
            if show_bbox:
                left, top, width, height = ann.bbox
                top_x, top_y, width, height = int(left), int(top), int(width), int(height)
                cv2.rectangle(img,
                              (top_x, top_y),
                              (top_x+width, top_y+height),
                              tuple(int(c) for c in color), bbox_thickness)

                # Add class
                class_name = categories[ann.category_id]
                text_width, text_height = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(img,
                              (top_x+bbox_thickness, top_y+bbox_thickness),
                              (top_x+4*bbox_thickness+text_width, top_y+4*bbox_thickness+text_height),
                              (0, 0, 0), -1)
                cv2.putText(img, class_name, (top_x+2*bbox_thickness, top_y+2*bbox_thickness+text_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            color = tuple(int(component) for component in color)
            for j in range(3, ann.num_keypoints, 3):
                point1 = (ann.keypoints[j-3], ann.keypoints[j+1-3])
                point2 = (ann.keypoints[j], ann.keypoints[j+1])
                img = cv2.circle(img, point1, radius=3, color=color, thickness=-1)
                img = cv2.circle(img, point2, radius=5, color=color, thickness=-1)
                img = cv2.line(img, point1, point2, color=color, thickness=3)
        show_img(img, img_entry.file_name)


if __name__ == "__main__":
    main()
