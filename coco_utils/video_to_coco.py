import json
from argparse import ArgumentParser
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET  # noqa

import cv2

from coco_types import Annotation, Category, Image


def main():
    parser = ArgumentParser("Tool to convert a video into a coco dataset with no classes")
    parser.add_argument("data_path", type=Path, help="Path to a folder with videos.")
    parser.add_argument("--frame_sampling", "--f", type=float, default=0.01,
                        help="How many frames of the video should be kept (fraction, from 0 to 1)")
    parser.add_argument("--output_path", "--o", type=Path, default=None,
                        help="Path for the resulting dataset, defaults to data_path.parent / 'coco_dataset'")
    args = parser.parse_args()

    data_path: Path = args.data_path
    frame_kept_ratio: float = args.frame_sampling
    output_path: Path = args.output_path if args.output_path else args.data_path.parent / "coco_dataset"
    (output_path / "images").mkdir(parents=True, exist_ok=True)

    images: list[Image] = []
    annotations: list[Annotation] = []
    categories: list[Category] = []

    image_id_counter: int = 0

    exts = (".mov", ".avi", ".mp4")
    video_paths = list([path for path in data_path.rglob('*') if path.suffix in exts])
    nb_videos = len(video_paths)
    for i, video_path in enumerate(video_paths, start=1):
        print(f"Processing video: {video_path.name} ({i}/{nb_videos})")

        cap = cv2.VideoCapture(str(video_path))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_nb = 0
        print(f"\tVideo has {video_length} frames, {video_length*frame_kept_ratio:.2f} frames will be kept")
        while frame_nb < video_length:
            msg = f"        Processing status for the current video: ({frame_nb+1}/{video_length})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)
            ret, img = cap.read()
            if ret:
                if frame_nb % (video_length // (video_length * frame_kept_ratio)) == 0:
                    filename = video_path.stem + f"_{str(image_id_counter).zfill(8)}.png"
                    cv2.imwrite(str(output_path / "images" / filename), img)
                    height, width, _ = img.shape
                    sample_image: Image = {
                        "id": image_id_counter,
                        "width": width,
                        "height": height,
                        "file_name": filename
                    }
                    images.append(sample_image)
                    image_id_counter += 1
            else:
                break
            frame_nb += 1
        cap.release()
        print()

    coco_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path / "annotations.json", 'w') as json_file:
        json.dump(coco_dataset, json_file, indent=4)

    print("\n")
    print(f"Saved the coco annotations and frames to {output_path}")


if __name__ == "__main__":
    main()
