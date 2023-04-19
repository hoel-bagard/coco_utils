"""Script to reindex the category indices in a coco annotation file.

Run with: python -m src.reindex_class_indices <path to json file>
"""
import argparse
import json
from pathlib import Path

from src.types.coco_types import Annotation, Category, Image


def main() -> None:
    parser = argparse.ArgumentParser(description=("Script to reindex the category indices in a coco annotation file."),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("json_path", type=Path, help="Path to COCO annotations file.")
    parser.add_argument("--start_idx", "-s", type=int, default=0, help="Where the new indices will start.")
    parser.add_argument("--output_path", "-o", type=Path, default=None,
                        help="Path for to where the edited json file will be saved. Defaults to input file.")
    args = parser.parse_args()

    json_path: Path = args.json_path
    start_idx: int = args.start_idx
    output_path: Path = args.output_path if args.output_path is not None else json_path

    # Load the dataset
    with json_path.open(encoding="utf-8") as annotations_file:
        coco_dataset = json.load(annotations_file)
    images: list[Image] = coco_dataset["images"]
    annotations: list[Annotation] = coco_dataset["annotations"]
    categories: list[Category] = coco_dataset["categories"]

    print(f"Found the following classes: {categories}")

    # Create a conversion table that does old_id -> new_id
    id_conversion_dict: dict[int, int] = {}
    for i, cat in enumerate(categories, start=start_idx):
        id_conversion_dict[cat["id"]] = i
        categories[i]["id"] = i

    print(f"Changing the class indices using the following mapping: {id_conversion_dict}")

    for i, ann in enumerate(annotations):
        annotations[i]["category_id"] = id_conversion_dict[ann["category_id"]]

    # Save the altered annotations
    edited_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    print(f"Saving the edited json to {output_path}")
    # import shutil
    # shutil.move(json_path, json_path.parent / "original_ids_annotations.json")
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(edited_dataset, json_file, indent=4)

    print("Finished processing the dataset.")


if __name__ == "__main__":
    main()
