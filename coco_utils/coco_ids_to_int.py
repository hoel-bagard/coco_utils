import argparse
import json
import shutil
from pathlib import Path

from coco_types import Annotation, Image


def is_duplicate(list_to_check: list[Image], key: str, elt_id: str) -> bool:
    """Checks if the given id is already in the list."""
    for entry in list_to_check:
        if entry[key] == elt_id:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=("Changes image ids in a coco dataset from strings to ints."
                                                  "Also removes duplicate entries."))
    parser.add_argument("annotations", type=Path, help="Path to COCO annotations file.")
    args = parser.parse_args()

    # Load the dataset
    with open(args.annotations) as annotations:
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
    for i in range(len(no_duplicate_images)):
        image_id_conversion_table[no_duplicate_images[i]["id"]] = i
        no_duplicate_images[i]["id"] = i

    for i in range(len(no_duplicate_annotations)):
        no_duplicate_annotations[i]["id"] = i
        no_duplicate_annotations[i]["image_id"] = image_id_conversion_table[no_duplicate_annotations[i]["image_id"]]

    # Save the corrected annotations
    corrected_dataset = {
        "images": no_duplicate_images,
        "annotations": no_duplicate_annotations,
        "categories": categories
    }
    shutil.move(args.annotations, args.annotations.parent / "original_ids_annotations.json")
    with open(args.annotations.parent / "annotations.json", 'w') as json_file:
        json.dump(corrected_dataset, json_file, indent=4)

    msg = "Finished processing dataset."
    print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)))


if __name__ == "__main__":
    main()
