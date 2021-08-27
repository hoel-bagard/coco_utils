from argparse import ArgumentParser
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from typing import Union
import shutil

import numpy as np

from coco_types import Image, Annotation, Category


def parse_voc2007_annotation(xml_path: Union[str, Path]) -> tuple[str, int, int, list[str, tuple[int, int, int, int]]]:
    """ Takes a path to an xml file and parses it to return the relevant information.

    Args:
        xml_path (str): Path to the xml file to parse

    Returns:
        Filename of the image, width and height of the image, list of the objects in the image.
        Each object consists of a string (the class) and a list of 4 ints (xmin, ymin, xmax, ymax)
    """
    root: ET.Element = ET.parse(xml_path).getroot()

    img_name = root.find("filename").text

    size: ET.Element = root.find("size")
    width, height = int(size.find("width").text), int(size.find("height").text)

    objects: ET.Element = root.findall("object")

    labels = []
    for item in objects:
        difficult = int(item.find("difficult").text)
        if difficult:
            continue
        labels.append([])

        cls = item.find("name").text
        labels[-1].append(cls)
        bbox = ((int(item.find("bndbox").find("xmin").text)),
                (int(item.find("bndbox").find("ymin").text)),
                (int(item.find("bndbox").find("xmax").text)),
                (int(item.find("bndbox").find("ymax").text)))
        labels[-1].append(bbox)

    return img_name, width, height, labels


def get_all_classes(xmls_path: list[Path]) -> list[str]:
    """ Takes a list of PascalVOC format xmls, and return a list of all the classes.
        Not efficient, but sometimes better than hardcoding it.

    Args:
        xml_paths (list): List with all the xml paths

    Returns:
        List with all the classes
    """
    classes = set()
    for xml_path in xmls_path:
        root: ET.Element = ET.parse(xml_path).getroot()
        objects: ET.Element = root.findall("object")
        for item in objects:
            classes.add(item.find("name").text)

    return list(classes)


def polygon_area(x: list[float], y: list[float]) -> float:
    """ Returns the area of a polygon

    Taken from: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    Args:
        x (list): List of with x coordinates of the vertices
        y (list): List of with y coordinates of the vertices

    Exemple:
        If the polygon is defined by a list of [x1, y1, ...., x_n, y_n] coordinates, then split it like this:
        x = [239.97, 222.04, 199.84, 213.5, 259.62, 274.13, 277.55, 249.37, 237.41, 242.54, 228.87]
        y = [260.24, 270.49, 253.41, 227.79, 200.46, 202.17, 210.71, 253.41, 264.51, 261.95, 271.34]

        then:
            >>> PolyArea(x, y)
            2765.1486500000005

    Returns:
        (float): The area of the polygon
    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def flatten(t: list):
    return [item for sublist in t for item in sublist]


def get_unused_id(list_to_check: list[Annotation]) -> str:
    """ Returns an id that is not already in the list """
    while True:
        new_id = str(np.random.random())
        already_used = False
        for entry in list_to_check:
            if entry["id"] == new_id:
                already_used = True
                break
        if not already_used:
            return new_id


def main():
    parser = ArgumentParser("Tool to convert PascalVOC format to coco format")
    parser.add_argument("data_path", type=Path,
                        help="Path to the dataset to convert. Images are expected to be in the same "
                        "or in an adjacent folder to the xmls. Images and corresponding xml files are expected to "
                        "have the same name.")
    parser.add_argument("--output_path", "--o", type=Path, default=None,
                        help="Path for the resulting json, defaults to data_path.parent / 'coco_annotations.json'")
    args = parser.parse_args()

    data_path: Path = args.data_path

    images: list[Image] = []
    annotations: list[Annotation] = []
    categories: list[Category] = []

    xml_paths = list(data_path.rglob("*.xml"))

    classes = get_all_classes(xml_paths)
    cls_name_to_id: dict[str, int] = {}
    for cls_id, cls_name in enumerate(classes):
        categories.append({
            "id": cls_id,
            "name": cls_name,
            "supercategory": cls_name
        })
        cls_name_to_id[cls_name] = cls_id

    bb_id = 0  # Each bounding box as a unique ID
    nb_imgs = len(xml_paths)
    for i, xml_path in enumerate(xml_paths, start=0):
        msg = f"Processing image: {xml_path.name} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

        filename, width, height, objects = parse_voc2007_annotation(xml_path)

        sample_image: Image = {
            "id": i,
            "width": width,
            "height": height,
            "file_name": filename
        }
        images.append(sample_image)

        # Add the segmentation for that layer
        for (cls_name, bbox) in objects:
            width = bbox[2]-bbox[0]
            height = bbox[3]-bbox[1]

            annotation: Annotation = {
                "id": bb_id,
                "image_id": i,
                "category_id": cls_name_to_id[cls_name],
                # "segmentation": [flatten([[point["x"], point["y"]] for point in region["points"]])],
                "segmentation": [],
                # "area": polygon_area([point["x"] for point in region["points"]],
                #                      [point["y"] for point in region["points"]]),
                "area": width * height,
                "bbox": [bbox[0],  # Top left x position
                         bbox[1],   # Top left y position
                         width,  # Width
                         height],  # Height
                "iscrowd": 0
            }
            bb_id += 1
            annotations.append(annotation)

    coco_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    output_path: Path = args.output_path if args.output_path else args.data_path.parent / "coco_annotations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as json_file:
        json.dump(coco_dataset, json_file, indent=4)

    print(f"Saved the coco annotations to {output_path}")


if __name__ == "__main__":
    main()
