import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def worker(args: tuple[Path]):
    """Worker in charge of converting an image into a 3 channels grayscale.

    Args:
        args: Path of the image to process.
    """
    img_path, = args
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, -1)
    img = np.concatenate((img, img, img), axis=-1)
    cv2.imwrite(img_path, img)


def main():
    parser = argparse.ArgumentParser(description=("Converts all images in a folder to grayscale (but still 3 channels)."
                                                  " Note: the transformation is done in place."))
    parser.add_argument("dir_path", type=Path, help="Path to the folder with the images.")
    args = parser.parse_args()

    exts = [".jpg", ".png"]
    image_paths = list([(str(p), ) for p in args.dir_path.rglob("*") if p.suffix in exts])
    nb_imgs = len(image_paths)

    nb_images_processed = 0
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for _result in pool.imap(worker, image_paths, chunksize=10):
            nb_images_processed += 1
            msg = f"Processing status: ({nb_images_processed}/{nb_imgs})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

    msg = "Finished processing images."
    print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)))


if __name__ == "__main__":
    main()
