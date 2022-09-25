import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from src.types.coco_types import Annotation, Category, Image


def main():
    parser = argparse.ArgumentParser(description="Splits COCO annotations file into training and validation sets.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images.")
    parser.add_argument("annotations_path", type=Path, help="Path to COCO annotations json file.")
    parser.add_argument("output_path", type=Path, help="Where to store the new Train and Validation datasets.")
    parser.add_argument("--spec_file", "-sf", type=Path, default=None,
                        help="Path to a text file that specifies which images to use for val (one name per line)")
    parser.add_argument("--split", "-s", type=float, default=0.85, help="Train split ratio")
    args = parser.parse_args()

    data_path: Path = args.data_path
    annotations_path: Path = args.annotations_path
    output_path: Path = args.output_path
    spec_file_path: Optional[Path] = args.spec_file
    split: float = args.split

    # Load the dataset
    with open(annotations_path, encoding="utf-8") as annotations_file:
        coco_dataset = json.load(annotations_file)
    images = np.asarray(coco_dataset["images"])
    annotations: list[Annotation] = coco_dataset["annotations"]
    categories: list[Category] = coco_dataset["categories"]

    if spec_file_path is None:
        number_of_images = len(images)
        indexes = np.arange(number_of_images)
        np.random.shuffle(indexes)

        # Split val / train
        train_images: list[Image] = list(images[indexes[:int(number_of_images*split)]])
        val_images: list[Image] = list(images[indexes[int(number_of_images*split):]])
    else:
        val_img_names = []
        with open(spec_file_path, "r", encoding="utf-8") as spec_file:
            for line in spec_file:
                val_img_names.append(line.strip())

        train_images = []
        val_images = []
        for image in images:
            if image["file_name"] in val_img_names:
                val_images.append(image)
            else:
                train_images.append(image)

    train_image_ids = [image["id"] for image in train_images]
    train_annotations: list[Annotation] = []
    val_annotations: list[Annotation] = []

    for annotation in annotations:
        if annotation["image_id"] in train_image_ids:
            train_annotations.append(annotation)
        else:
            val_annotations.append(annotation)

    # Save new training annotations
    train_dataset = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }
    train_output_path: Path = output_path / "train"
    train_output_path.mkdir(parents=True, exist_ok=True)
    with open(train_output_path / "annotations.json", "w", encoding="utf-8") as json_file:
        json.dump(train_dataset, json_file, indent=4)

    # Save new validation annotations
    val_dataset = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }
    val_output_path: Path = output_path / "validation"
    val_output_path.mkdir(parents=True, exist_ok=True)
    with open(val_output_path / "annotations.json", "w", encoding="utf-8") as json_file:
        json.dump(val_dataset, json_file, indent=4)

    print(f"Saved {len(train_images)} entries to {train_output_path} and {len(val_images)} to {val_output_path}")

    print("Now moving images. . .", end="\r")
    for image in train_images:
        in_img_path: Path = data_path / image["file_name"]
        out_img_path: Path = train_output_path / "images"
        out_img_path.mkdir(exist_ok=True)
        shutil.copy(in_img_path, out_img_path)
    for image in val_images:
        in_img_path: Path = data_path / image["file_name"]
        out_img_path: Path = val_output_path / "images"
        out_img_path.mkdir(exist_ok=True)
        shutil.copy(in_img_path, out_img_path)

    print("Finished splitting dataset.")


if __name__ == "__main__":
    main()
