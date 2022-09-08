import argparse
import json
import shutil
from pathlib import Path

from coco_types import Annotation, Category, Image


def main():
    parser = argparse.ArgumentParser("Merges several coco label files into one."
                                     "If also renaming the files, it is assumed the images are in an images folder "
                                     "next to the annotations file. The images will be copied to the output dir")
    parser.add_argument("annotations_paths", nargs="+", type=Path, help="Paths to COCO annotations files to merge.")
    parser.add_argument("--change_names", "--c", action="store_true",
                        help="Change image names by prefixing dataset name.")
    parser.add_argument("--output_path", "--o", type=Path, default=None,
                        help="Path for the resulting json, defaults to the folder of the first json path")
    args = parser.parse_args()

    annotations_paths: list[Path] = args.annotations_paths
    output_path: Path = args.output_path if args.output_path else args.annotations_paths[0].parent
    output_path.mkdir(parents=True, exist_ok=True)
    if args.change_names:
        (output_path / "images").mkdir(parents=True, exist_ok=True)

    merged_images: list[Image] = []
    merged_annotations: list[Annotation] = []
    merged_categories: list[Category] = []
    for i, annotation_path in enumerate(annotations_paths):
        with open(annotation_path) as annotations_file:
            coco_dataset = json.load(annotations_file)
        images: list[Image] = coco_dataset["images"]

        if args.change_names:
            assert (annotation_path.parent / "images").exists, f"No images found for annotations {annotation_path}"
            for i in range(len(images)):
                shutil.copy(annotation_path.parent / "images" / images[i]["file_name"],
                            output_path / "images" / (annotation_path.parent.name + "_" + images[i]["file_name"]))
                images[i]["file_name"] = annotation_path.parent.name + "_" + images[i]["file_name"]

        merged_images.extend(images)
        merged_annotations.extend(coco_dataset["annotations"])
        # For this project the categories should be the same in every file, no need to duplicate them
        if i == 0:
            merged_categories.extend(coco_dataset["categories"])

    merged_dataset = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories
    }

    with open(output_path / "merged_annotations.json", 'w') as json_file:
        json.dump(merged_dataset, json_file, indent=4)

    msg = "Finished processing dataset."
    print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)))


if __name__ == "__main__":
    main()
