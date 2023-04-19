import argparse
import json
import shutil
from pathlib import Path

from src.types.coco_types import Annotation, Image


def is_duplicate(list_to_check: list[Image] | list[Annotation], key: str, elt_id: str) -> bool:
    """Checks if the given id is already in the list."""
    return any(entry[key] == elt_id for entry in list_to_check)


def main() -> None:
    parser = argparse.ArgumentParser(description=("Changes image ids in a coco dataset from strings to ints."
                                                  "Also removes duplicate entries."))
    parser.add_argument("annotations", type=Path, help="Path to COCO annotations file.")
    args = parser.parse_args()

    # Load the dataset
    with args.annotations.open(encoding="utf-8") as annotations:
        coco_dataset = json.load(annotations)
    images = coco_dataset["images"]
    annotations = coco_dataset["annotations"]
    categories = coco_dataset["categories"]

    # Could try to play around with sets and pops, but there's no time
    no_duplicate_annotations: list[Annotation] = []
    no_duplicate_images: list[Image] = []

    # Remove duplicates
    for annotation in annotations:
        if not is_duplicate(no_duplicate_annotations, "id", annotation["id"]):
            no_duplicate_annotations.append(annotation)
    for image in images:
        if not is_duplicate(no_duplicate_images, "id", image["id"]):
            no_duplicate_images.append(image)

    # Change image ids to ints and create a conversion table that does old_id -> new_id
    image_id_conversion_table = {}
    for i, img_entry in enumerate(no_duplicate_images):
        image_id_conversion_table[img_entry["id"]] = i
        img_entry["id"] = i

    for i, annotation_entry in enumerate(no_duplicate_annotations):
        annotation_entry["id"] = i
        annotation_entry["image_id"] = image_id_conversion_table[annotation_entry["image_id"]]

    # Save the corrected annotations
    corrected_dataset = {
        "images": no_duplicate_images,
        "annotations": no_duplicate_annotations,
        "categories": categories,
    }
    shutil.move(args.annotations, args.annotations.parent / "original_ids_annotations.json")
    with (args.annotations.parent / "annotations.json").open("w", encoding="utf-8") as json_file:
        json.dump(corrected_dataset, json_file, indent=4)

    msg = "Finished processing dataset."
    print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)))


if __name__ == "__main__":
    main()
