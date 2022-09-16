from typing import TypedDict


class Image(TypedDict):
    id: int
    width: int
    height: int
    file_name: str


class Annotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    # Segmentation could also be RLE, but not this is not handled for now.
    # Exemple of segmentation: "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]]
    segmentation: list[list[float]]
    area: float
    # The COCO bounding box format is [top left x position, top left y position, width, height].
    # bbox exemple:  "bbox": [473.07,395.93,38.65,28.67]
    bbox: list[float]
    iscrowd: int  # Either 1 or 0


class Category(TypedDict):
    id: int
    name: str
    supercategory: str
