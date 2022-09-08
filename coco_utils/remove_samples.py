import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create a smaller dataset by keeping a smaller id range.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("annotations_path", type=Path, help="Path to COCO annotations file.")
    parser.add_argument("--output_path", "-o", type=Path, default=None,
                        help="Where to store the new dataset, defaults to annotations_path/../smaller_dataset")
    parser.add_argument("--max_id", "-m", type=float, default=100, help="All images with id above will be removed")
    args = parser.parse_args()

    annotations_path: Path = args.annotations_path
    max_id: int = args.max_id
    output_path: Path = args.output_path if args.output_path else annotations_path.parent / "smaller_dataset"

    output_path.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    with open(annotations_path, encoding="utf-8") as annotations:
        coco_dataset = json.load(annotations)
    images = coco_dataset["images"]
    annotations = coco_dataset["annotations"]
    categories = coco_dataset["categories"]

    new_annotations = [annotation for annotation in annotations if annotation["image_id"] < max_id]
    new_images = [image for image in images if image["id"] < max_id]

    # Save new newing annotations
    new_dataset = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories
    }
    with open(output_path / "annotations.json", 'w', encoding="utf-8") as json_file:
        json.dump(new_dataset, json_file, indent=4)

    print(f"Saved {len(new_images)} entries to {output_path}")

    print("Now removing images. . .", end="\r")
    for image in images:
        if image["id"] < max_id:
            continue
        in_img_path: Path = annotations_path.parent / "images" / image["file_name"]
        out_img_path: Path = output_path / "images"
        out_img_path.mkdir(exist_ok=True)
        shutil.copy(in_img_path, out_img_path)

    print("Finished trimming dataset.")


if __name__ == "__main__":
    main()
