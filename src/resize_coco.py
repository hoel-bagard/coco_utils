import argparse
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from coco_types import Annotation, Image


def worker(args: tuple[Image, Path, Path, int, int]):
    """Worker in charge of resizing an image.

    Args:
        args: Tuple containing the following:
              - image: the image to process
              - data_path: path to the image directory
              - output_path: path to the output directory
              - new_width: width to which resize the image
              - new_height: height to which resize the image
    """
    image, data_path, output_path, new_width, new_height = args

    img = cv2.imread(str(data_path / image["file_name"]))
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    out_img_path: Path = output_path / "images" / image["file_name"]
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), resized_img)


def polygon_area(x: list[float], y: list[float]) -> float:
    """Returns the area of a polygon.

    Taken from: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    Args:
        x (list): List of with x coordinates of the vertices
        y (list): List of with y coordinates of the vertices

    Exemple:
        If the polygon is defined by a list of [x1, y1, ...., x_n, y_n] coordinates, then split it like this:
        x = [239.97, 222.04, 199.84, 213.5, 259.62, 274.13, 277.55, 249.37, 237.41, 242.54, 228.87]
        y = [260.24, 270.49, 253.41, 227.79, 200.46, 202.17, 210.71, 253.41, 264.51, 261.95, 271.34]

        then:
            >>> PolyArea(x, y)
            2765.1486500000005

    Returns:
        (float): The area of the polygon
    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def get_img_from_id(images: list[Image], img_id: str) -> Image | None:
    """Returns the image with the given id, or None is no such image exists."""
    for image in images:
        if image["id"] == img_id:
            return image
    return None


def main():
    parser = argparse.ArgumentParser(description="Resizes the images labels of a COCO dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images.")
    parser.add_argument("annotations", type=Path, help="Path to COCO annotations file.")
    parser.add_argument("output_path", type=Path, help="Where to store the resized dataset.")
    parser.add_argument("size", nargs=2, type=int, help="Size to which the images will be resized.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path
    size: tuple[int, int] = args.size

    new_width, new_height = size

    # Load the dataset
    with open(args.annotations, 'r', encoding="utf-8") as annotations:
        coco_dataset = json.load(annotations)
    images = coco_dataset["images"]
    annotations = coco_dataset["annotations"]
    categories = coco_dataset["categories"]

    resized_images: list[Image] = []
    resized_annotations: list[Annotation] = []

    for annotation in annotations:
        image = get_img_from_id(images, annotation["image_id"])
        width, height = image["width"], image["height"]

        resized_image: Image = {
            "id": image["id"],
            "width": new_width,
            "height": new_height,
            "file_name": image["file_name"]
        }
        resized_images.append(resized_image)

        width_ratio, height_ratio = width / new_width, height / new_height
        new_segmentation = [v / width_ratio if i % 2 == 0 else v / height_ratio
                            for i, v in enumerate(annotation["segmentation"][0])]
        new_area = polygon_area([v for i, v in enumerate(new_segmentation) if i % 2 == 0],
                                [v for i, v in enumerate(new_segmentation) if i % 2 == 1])
        new_bbox = [annotation["bbox"][0] / width_ratio,
                    annotation["bbox"][1] / height_ratio,
                    annotation["bbox"][2] / width_ratio,
                    annotation["bbox"][3] / height_ratio]
        annotation: Annotation = {
            "id": annotation["id"],
            "image_id": annotation["image_id"],
            "category_id": annotation["category_id"],
            "segmentation": [new_segmentation],
            "area": new_area,
            "bbox": new_bbox,
            "iscrowd": annotation["iscrowd"]
        }
        resized_annotations.append(annotation)

    nb_imgs = len(resized_images)
    mp_args = [(image, data_path, output_path, new_width, new_height) for image in resized_images]
    nb_images_processed = 0
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for _ in pool.imap(worker, mp_args, chunksize=10):
            nb_images_processed += 1
            msg = f"Processing status: ({nb_images_processed}/{nb_imgs})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

    # Save the resized annotations
    resized_dataset = {
        "images": resized_images,
        "annotations": resized_annotations,
        "categories": categories
    }
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "annotations.json", 'w', encoding="utf-8") as json_file:
        json.dump(resized_dataset, json_file, indent=4)

    msg = "Finished resizing dataset."
    print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)))


if __name__ == "__main__":
    main()
