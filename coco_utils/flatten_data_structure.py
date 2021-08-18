from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import shutil


def main():
    parser = ArgumentParser("Flattens a dataset by putting all the images in one folder.")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("--labels_too", "--l", action="store_true", help="Puts any xml/jsons found in a 'label' folder")
    parser.add_argument("--move", "--m", action="store_true", help="Move files instead of copying them")
    parser.add_argument("--output_path", "--o", type=Path, default=None,
                        help="Path for the resulting json, defaults to data_path.parent / 'flattened_dataset'")
    args = parser.parse_args()

    data_path: Path = args.data_path
    labels_too: bool = args.labels_too
    move: bool = args.move
    img_output_path: Path = args.output_path if args.output_path else args.data_path.parent / "flattened_dataset"

    moving_fn = shutil.move if move else shutil.copy  # Shortcut to avoid ifs later

    # Output paths are a bit diffenrent if also moving the labels
    if labels_too:
        img_output_path = img_output_path / "imgs"
        labels_output_path = img_output_path / "labels"
        labels_output_path.mkdir(parents=True, exist_ok=True)
    img_output_path.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".bmp")
    img_paths = list([path for path in data_path.rglob('*') if path.suffix in exts])
    nb_imgs = len(img_paths)
    for i, img_path in enumerate(img_paths, start=1):
        msg = f"Processing image: {img_path.name} ({i}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

        file_output_path = img_output_path / img_path.name
        xml_modified = False
        xml_path: Path = None

        # Check if xml exists (either next to the image or in an adjacent "labels" folder)
        if img_path.with_suffix(".xml").exists():
            xml_path = img_path.with_suffix(".xml")
        elif (img_path.parent.parent / "labels" / (img_path.stem + ".xml")).exists():
            xml_path = img_path.parent.parent / "labels" / (img_path.stem + ".xml")

        if file_output_path.exists():
            # There is already an image and xml label file with that name. Rename and the move.
            if xml_path:
                xml_modified = True
                tree = ET.parse(xml_path)

                while file_output_path.exists():
                    file_output_path = file_output_path.with_stem(file_output_path.stem + f"_{random.randint(0, 1000)}")
                    xml_path = xml_path.with_stem(file_output_path.stem)

                root: ET.Element = tree.getroot()
                root.find("filename").text = file_output_path.name

            # There is already an image with that name.
            # But no xml (maybe a coco file somewhere else, but then it's already too late)
            else:
                print(f"\nFile {file_output_path} already exists. Skipping. "
                      "You might want to look into that, as the coco file is probably 'wrong'.")
                continue

        moving_fn(img_path, file_output_path)
        if xml_path and not xml_modified:
            moving_fn(xml_path, labels_output_path / xml_path.name)
        elif xml_modified:
            with open(labels_output_path / xml_path.name, "wb") as f:
                tree.write(f)

    print("\nFinished!")


if __name__ == "__main__":
    main()
