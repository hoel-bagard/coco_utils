import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.types.solo_frame_data import Capture


def main():
    parser = argparse.ArgumentParser(description=("Tool to visualize solo labels. "
                                                  "Use with 'python -m src.visualize_solo <path to data folder> "),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the dataset.")
    args = parser.parse_args()

    data_path: Path = args.data_path

    with open(data_path, encoding="utf-8") as data_file:
        data_string = data_file.read()

    data = Capture.parse_raw(data_string)
    print(data)


if __name__ == "__main__":
    main()
