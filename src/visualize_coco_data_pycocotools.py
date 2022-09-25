import argparse
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from src.types.coco_types import Annotation, Image


def main():
    parser = argparse.ArgumentParser(description="Tool to visualize coco labels.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images.")
    parser.add_argument("json_path", type=Path, help="Path to the json file with the coco annotations.")
    parser.add_argument("--show_bbox", "-sb", action="store_true", help="Show the bounding boxes.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    json_path: Path = args.json_path
    show_bbox: bool = args.show_bbox

    coco = COCO(json_path)

    # Get all images containing given categories.
    img_ids: list[int] = coco.getImgIds()

    for i in range(len(img_ids)):
        img_data: Image = coco.loadImgs([img_ids[i]])[0]  # type: ignore
        msg = f"Showing image: {img_data['file_name']} ({i+1}/{len(img_ids)})"
        print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r", flush=True)

        ann_ids: list[int] = coco.getAnnIds(imgIds=[img_data["id"]])
        anns: list[Annotation] = coco.loadAnns(ann_ids)  # type: ignore

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
