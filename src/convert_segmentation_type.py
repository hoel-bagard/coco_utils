"""Script to convert one segmentation type to another.

There are 3 ways of representing a segmentation for a coco dataset:
- Polygon
- RLE
- Encoded RLE

Run with: python -m src.convert_segmentation_type <path to json file>
"""
import argparse
import json
from pathlib import Path

import src.utils.segmentation_conversions as cvt
from src.types.coco_types import Annotation, Category, Image


def main():
    parser = argparse.ArgumentParser(description="Script to convert one segmentation type to another..",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("json_path", type=Path, help="Path to COCO annotations file.")
    parser.add_argument("--format", "-f", choices=["polygon", "rle", "encoded_rle"], default="rle", type=str,
                        help="The destination format.")
    parser.add_argument("--output_path", "-o", type=Path, default=None,
                        help="Path for to where the edited json file will be saved. Defaults to inplace editing.")
    args = parser.parse_args()

    json_path: Path = args.json_path
    format: str = args.format
    output_path: Path = args.output_path if args.output_path is not None else json_path

    # Load the dataset
    with open(json_path, encoding="utf-8") as annotations_file:
        coco_dataset = json.load(annotations_file)
    images: list[Image] = coco_dataset["images"]
    annotations: list[Annotation] = coco_dataset["annotations"]
    categories: list[Category] = coco_dataset["categories"]

    print(f"Loaded a json file containing {len(annotations)} annotations. Converting them to {format} format.")

    for i, annotation in enumerate(annotations):
        assert "segmentation" in annotation, f"No segmentation found for annotation {annotation}"
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            if format == "rle":
                raise NotImplementedError("RLE -> Encoded RLE is not implemented yet.")
            if format == "encoded_rle":
                raise NotImplementedError("RLE -> Encoded RLE is not implemented yet.")
            raise NotImplementedError("Polygon segmentation is not implemented yet.")
        else:
            if isinstance(segmentation["counts"], list):
                if format == "polygon":
                    raise NotImplementedError("RLE -> Polygon is not implemented yet.")
                elif format == "encoded_rle":
                    raise NotImplementedError("RLE -> Encoded RLE is not implemented yet.")
            else:
                if format == "polygon":
                    raise NotImplementedError("Encoded RLE -> Polygon is not implemented yet.")
                elif format == "rle":
                    rle = cvt.encoded_rle_to_rle(segmentation["counts"]).tolist()
                    annotations[i]["segmentation"]["counts"] = rle  # type: ignore

    # Save the altered annotations
    edited_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    print(f"Saving the edited json to: '{output_path}'")
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(edited_dataset, json_file, indent=4)

    print("Finished processing the dataset.")


if __name__ == "__main__":
    main()
