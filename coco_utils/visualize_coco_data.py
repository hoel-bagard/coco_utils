import argparse
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def main():
    parser = argparse.ArgumentParser(description="Tool to visualize coco labels.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images")
    parser.add_argument("json_path", type=Path, help="Path to the json file with the coco annotations")
    parser.add_argument("--show_bbox", "--sb", action="store_true", help="Show the bounding boxes")
    args = parser.parse_args()

    data_path: Path = args.data_path
    json_path: Path = args.json_path

    coco = COCO(json_path)

    # Get all images containing given categories, select one at random
    cat_ids = coco.getCatIds(catNms=["Thread"])
    img_ids = coco.getImgIds(catIds=cat_ids)

    for i in range(len(img_ids)):
        img_data = coco.loadImgs([img_ids[i]])[0]
        msg = f"Showing image: {img_data['file_name']} ({i+1}/{len(img_ids)})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

        ann_ids = coco.getAnnIds(imgIds=[img_data["id"]])
        anns = coco.loadAnns(ann_ids)

        # Load an image and its corresponding instance annotations then display it
        img = cv2.imread(str(args.data_path / img_data["file_name"]))

        if args.show_bbox:
            # Add the bounding box to the image
            for annotation in anns:
                top_x, top_y, width, height = annotation["bbox"]
                img = cv2.rectangle(img, (int(top_x), int(top_y)), (int(top_x+width), int(top_y+height)),
                                    (255, 0, 0), 5)

        plt.imshow(img)
        plt.axis('off')
        coco.showAnns(anns)
        plt.show()


if __name__ == "__main__":
    main()


# Old code snipet to show the segmentation with OpenCV
# image_name = sample["asset"]["name"]
# for i, sample_2 in enumerate(json_object["assets"].values(), start=1):
#     if video_name == sample_2["asset"]["parent"]["name"] and image_name != sample_2["asset"]["name"]:
#         for region in sample_2["regions"]:
#             color = color_map[int(region["tags"][0])]
#             points: list[dict[str, float]] = region["points"]
#             pts = np.asarray([[point["x"], point["y"]] for point in points], dtype=np.int32)
#             pts = pts.reshape((-1, 1, 2))
#             img = cv2.fillPoly(img, [pts], color)
