from argparse import ArgumentParser
from pathlib import Path
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
        if file_output_path.exists():
            print(f"\nFile {file_output_path} already exists. Skipping.")

        if move:
            shutil.move(img_path, file_output_path)
        else:
            shutil.copy(img_path, file_output_path)
    print("\nFinished copying/moving images.")

    if labels_too:
        label_paths = list([path for path in data_path.rglob('*') if path.suffix in (".xml", ".json")])
        nb_label_files = len(label_paths)
        for i, label_path in enumerate(label_paths, start=1):
            msg = f"Processing file: {label_path.name} ({i}/{nb_label_files})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

            file_output_path = labels_output_path / label_path.name
            if file_output_path.exists():
                print(f"\nFile {file_output_path} already exists. Skipping.")

            if move:
                shutil.move(label_path, file_output_path)
            else:
                shutil.copy(label_path, file_output_path)
        print("\nFinished copying/moving labels.")

    print("\nFinished!")


if __name__ == "__main__":
    main()
