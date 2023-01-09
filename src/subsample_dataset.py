import argparse
import json
import shutil
from pathlib import Path

from src.types.coco_types import Annotation, Category, Image


def main():
    parser = argparse.ArgumentParser(description="Create a smaller dataset by keeping a smaller id range.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("annotations_path", type=Path, help="Path to the COCO annotations file.")
    parser.add_argument("--images_folder_path", "-i", type=Path, default=None,
                        help="Path to the folder with the images. Defaults to annotations_path/../images")
    parser.add_argument("--output_path", "-o", type=Path, default=None,
                        help="Where to store the new dataset, defaults to annotations_path/../smaller_dataset")
    parser.add_argument("--max_id", "-m", type=int, default=None, help="Images with id above max_id will be removed.")
    parser.add_argument("--keep_ids", "-k", nargs="+", type=int, default=None, help="Ids to be kept.")
    args = parser.parse_args()

    annotations_path: Path = args.annotations_path
    images_path: Path = args.images_folder_path if args.images_folder_path else annotations_path.parent / "images"
    output_path: Path = args.output_path if args.output_path else annotations_path.parent / "smaller_dataset"
    max_id: int | None = args.max_id
    ids_to_keep: list[int] | None = args.keep_ids

    assert max_id is not None or ids_to_keep is not None, "One of --max_id and --keep_ids must be used."

    output_path.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    with open(annotations_path, encoding="utf-8") as annotations_file:
        coco_dataset = json.load(annotations_file)
    images: list[Image] = coco_dataset["images"]
    annotations: list[Annotation] = coco_dataset["annotations"]
    categories: list[Category] = coco_dataset["categories"]

    new_annotations = [annotation
                       for annotation in annotations
                       if (max_id is not None and annotation["image_id"] < max_id)
                       or (ids_to_keep is not None and annotation["image_id"] in ids_to_keep)]
    new_images = [image
                  for image in images
                  if (max_id is not None and image["id"] < max_id)
                  or (ids_to_keep is not None and image["id"] in ids_to_keep)]

    # Save new newing annotations
    new_dataset = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories
    }
    with open(output_path / "annotations.json", "w", encoding="utf-8") as json_file:
        json.dump(new_dataset, json_file, indent=4)

    print(f"Saved {len(new_images)} entries to {output_path}")

    print("Now copying images. . .", end="\r")
    for image in new_images:
        in_img_path: Path = images_path / image["file_name"]
        out_img_path: Path = output_path / "images"
        out_img_path.mkdir(exist_ok=True)
        shutil.copy(in_img_path, out_img_path)

    print("Finished trimming dataset.")


if __name__ == "__main__":
    main()
