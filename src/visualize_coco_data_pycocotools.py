"""Script to visualize the labels of a coco-like dataset using the official coco api."""
import argparse
import shutil
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def main():
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

    coco = COCO(json_path)

    # Get all the image ids.
    img_ids = coco.getImgIds()

    for i, img_id in enumerate(img_ids):
        img_data = coco.loadImgs([img_id])[0]
        if img_name is not None and img_name != img_data["file_name"]:
            continue
        msg = f"Showing image: {img_data['file_name']} ({i+1}/{len(img_ids)})"
        print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r", flush=True)

        ann_ids = coco.getAnnIds(imgIds=[img_data["id"]])
        anns = coco.loadAnns(ann_ids)

        # Load an image and its corresponding instance annotations then display it
        img = cv2.imread(str(data_path / img_data["file_name"]))
        if show_bbox:
            # Add the bounding box to the image
            for annotation in anns:
                top_x, top_y, width, height = annotation["bbox"]
                img = cv2.rectangle(img, (int(top_x), int(top_y)), (int(top_x+width), int(top_y+height)),
                                    (255, 0, 0), 5)

        plt.imshow(img)
        plt.axis("off")
        coco.showAnns(anns)
        plt.show()


if __name__ == "__main__":
    main()
