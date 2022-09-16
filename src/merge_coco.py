import argparse
import json
import shutil
from pathlib import Path

from src.types.coco_types import Annotation, Category, Image


def main():
    parser = argparse.ArgumentParser(description=("Merges several coco label files into one."
                                                  "If also renaming the files, it is assumed the images are in an "
                                                  "'images' folder next to the annotations file. The images will "
                                                  "be copied to the output dir"),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("annotations_paths", nargs="+", type=Path, help="Paths to COCO annotations files to merge.")
    parser.add_argument("--change_names", "-c", action="store_true",
                        help="Change image names by prefixing the dataset name.")
    parser.add_argument("output_path", type=Path, help="Path for to where the merged dataset will be stored.")
    args = parser.parse_args()

    annotations_paths: list[Path] = args.annotations_paths
    change_names: bool = args.change_names
    output_path: Path = args.output_path

    output_path.mkdir(parents=True, exist_ok=True)
    (output_img_path := output_path / "images").mkdir(parents=True, exist_ok=False)

    merged_images: list[Image] = []
    merged_annotations: list[Annotation] = []
    merged_categories: list[Category] = []
    for i, annotation_path in enumerate(annotations_paths):
        print(f"Processing file {annotation_path}")
        with open(annotation_path, 'r', encoding="utf-8") as annotations_file:
            coco_dataset = json.load(annotations_file)
        images: list[Image] = coco_dataset["images"]
        nb_imgs = len(images)

        assert (annotation_path.parent / "images").exists, f"No images found for annotations {annotation_path}"
        for img_entry in images:
            msg = f"Processing entry: {img_entry['file_name']} ({i}/{nb_imgs})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)),
                  end='\r' if i != nb_imgs else '\n', flush=True)

            filename = img_entry["file_name"]
            if change_names:
                filename = f"{annotation_path.parent.name}_" + filename

            shutil.copy(annotation_path.parent / "images" / img_entry["file_name"], output_img_path / filename)
            img_entry["file_name"] = filename

        merged_images.extend(images)
        merged_annotations.extend(coco_dataset["annotations"])
        # Categories should be the same in every file, no need to duplicate them
        if i == 0:
            merged_categories.extend(coco_dataset["categories"])

    merged_dataset = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories
    }

    with open(output_path / "merged_annotations.json", 'w', encoding="utf-8") as json_file:
        json.dump(merged_dataset, json_file, indent=4)

    msg = "Finished processing dataset."
    print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)))


if __name__ == "__main__":
    main()
