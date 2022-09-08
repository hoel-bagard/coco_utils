import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Splits COCO annotations file into training and validation sets.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images.")
    parser.add_argument("annotations", type=Path, help="Path to COCO annotations file.")
    parser.add_argument("output_path", type=Path, help="Where to store the new Train and Validation datasets")
    parser.add_argument("--spec_file", "-sf", type=Path, default=None,
                        help="Path to a text file that specifies which images to use for val (one name per line)")
    parser.add_argument("--split", "-s", type=float, default=0.85, help="Train split ratio")
    args = parser.parse_args()

    # Load the dataset
    with open(args.annotations, encoding="utf-8") as annotations:
        coco_dataset = json.load(annotations)
    images = np.asarray(coco_dataset["images"])
    annotations = coco_dataset["annotations"]
    categories = coco_dataset["categories"]

    if not args.spec_file:
        number_of_images = len(images)
        indexes = np.arange(number_of_images)
        np.random.shuffle(indexes)

        # Split val / train
        train_images = list(images[indexes[:int(number_of_images*args.split)]])
        val_images = list(images[indexes[int(number_of_images*args.split):]])
    else:
        val_img_names = []
        with open(args.spec_file, 'r', encoding="utf-8") as spec_file:
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
    train_annotations = []
    val_annotations = []

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
    train_output_path: Path = args.output_path / "train"
    train_output_path.mkdir(parents=True, exist_ok=True)
    with open(train_output_path / "annotations.json", 'w', encoding="utf-8") as json_file:
        json.dump(train_dataset, json_file, indent=4)

    # Save new validation annotations
    val_dataset = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }
    val_output_path: Path = args.output_path / "validation"
    val_output_path.mkdir(parents=True, exist_ok=True)
    with open(val_output_path / "annotations.json", 'w', encoding="utf-8") as json_file:
        json.dump(val_dataset, json_file, indent=4)

    print(f"Saved {len(train_images)} entries to {train_output_path} and {len(val_images)} to {val_output_path}")

    print("Now moving images. . .", end="\r")
    for image in train_images:
        in_img_path: Path = args.data_path / image["file_name"]
        out_img_path: Path = train_output_path / "images"
        out_img_path.mkdir(exist_ok=True)
        shutil.copy(in_img_path, out_img_path)
    for image in val_images:
        in_img_path: Path = args.data_path / image["file_name"]
        out_img_path: Path = val_output_path / "images"
        out_img_path.mkdir(exist_ok=True)
        shutil.copy(in_img_path, out_img_path)

    print("Finished splitting dataset.")


if __name__ == "__main__":
    main()
