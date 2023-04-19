"""Script to visualize the labels of a coco-like dataset using the official coco api."""
import argparse
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rpycocotools
from rpycocotools import COCO

from src.utils.imgs_misc import show_img


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool to visualize coco labels.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images.")
    parser.add_argument("json_path", type=Path, help="Path to the json file with the coco annotations.")
    parser.add_argument("--show_bbox", "-sb", action="store_true", help="Show the bounding boxes.")
    parser.add_argument("--image_name", "-i", type=str, default=None,
                        help="If given, only that image will be displayed.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    json_path: Path = args.json_path
    show_bbox: bool = args.show_bbox
    img_name: Optional[str] = args.image_name

    rng = np.random.default_rng()
    coco = COCO(str(json_path), str(data_path))
    img_entries = coco.get_imgs()

    nb_samples= len(img_entries)
    for i, img_entry in enumerate(img_entries):
        if img_name is not None and img_name != img_entry.file_name:
            continue

        msg = f"Showing image: {img_entry.file_name} ({i+1}/{nb_samples})"
        print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r", flush=True)


        img = cv2.imread(str(data_path / img_entry.file_name))
        anns = coco.get_img_anns(img_entry.id)
        color = rng.integers(0, high=255, size=3, dtype=np.uint8)
        for annotation in anns:
            if show_bbox:
                top_x, top_y, width, height = annotation.bbox
                img = cv2.rectangle(img, (int(top_x), int(top_y)), (int(top_x+width), int(top_y+height)),
                                    (255, 0, 0), 5)

            mask = rpycocotools.mask.decode(annotation.segmentation)
            mask = color * np.expand_dims(mask, -1)
            img = np.where(mask, 0.3*mask + 0.7*img, img).astype(np.uint8)

        show_img(img, img_entry.file_name)


if __name__ == "__main__":
    main()
