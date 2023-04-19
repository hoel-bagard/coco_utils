"""Script to remove annotation entries and images that do not have annotations.

Use with 'python -m src.remove_imgs_without_annotations <path to json annotation file>'
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from src.types.coco_types import Annotation, Category, Image
from src.utils.misc import clean_print


def main() -> None:
    parser = argparse.ArgumentParser(description="Script to remove entries and images that do not have annotations.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("json_path", type=Path, help="Path to the json file with the coco annotations.")
    parser.add_argument("--output_path", "-o", type=Path, default=None,
                        help="Path for to where the edited json file will be saved. Defaults to inplace editing.")
    parser.add_argument("--data_path", "-d", type=Path, default=None,
                        help=("Path to the directory with the images."
                              "If given, the images without annotations will be deleted."))
    args = parser.parse_args()

    json_path: Path = args.json_path
    output_path: Path = args.output_path if args.output_path is not None else json_path
    data_path: Optional[Path] = args.data_path

    with json_path.open(encoding="utf-8") as annotations_file:
        coco_dataset = json.load(annotations_file)

    images: list[Image] = coco_dataset["images"]
    annotations: list[Annotation] = coco_dataset["annotations"]
    categories: list[Category] = coco_dataset["categories"]

    kept_images: list[Image] = []
    nb_imgs = len(images)
    nb_deleted_entries = 0
    print(f"Loaded a json file containing {nb_imgs} image entries. Filtering out the ones without annotation.")
    for i, img_entry in enumerate(images, start=1):
        clean_print(f"Processing entry: ({i+1}/{nb_imgs})", end="\r" if i+1 != nb_imgs else "\n")

        if len([ann for ann in annotations if ann["image_id"] == img_entry["id"]]) != 0:
            kept_images.append(img_entry)
        else:
            print(f"\nRemoving the annotation entry for {img_entry['file_name']}.")
            nb_deleted_entries += 1
            if data_path is not None:
                img_path = data_path / img_entry["file_name"]
                print(f"Deleting the corresponding image: {img_path}.")
                img_path.unlink()

    # Save the filtered annotations.
    edited_dataset = {
        "images": kept_images,
        "annotations": annotations,
        "categories": categories,
    }

    print(f"Saving the edited json to: '{output_path}'")
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(edited_dataset, json_file, indent=4)

    print(f"Finished processing the dataset, remove {nb_deleted_entries} entries")


if __name__ == "__main__":
    main()
