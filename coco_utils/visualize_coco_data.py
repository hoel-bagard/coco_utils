import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from coco_types import Annotation, Category, Image


def decode_rle(encoded_count_rle: str, height: int = 480, width: int = 640) -> list[int]:
    """Decode incoded segmentation information into the standard format.

    See the (hard to read) implementation:
        https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c#L218
        https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/_mask.pyx#L145

    LEB128 wikipedia article: https://en.wikipedia.org/wiki/LEB128#Decode_signed_integer
    It is similar to LEB128, but here shift is incremented by 5 instead of 7 because the implementation uses
    6 bits per byte instead of 8...

    I'm reimplementing it in python here half by curiosity and half for debugging purposes.
    Performance is obviously going to be worse.
    """
    bytes_rle = str.encode(encoded_count_rle)

    print(len(encoded_count_rle))
    m, current_byte_idx, counts = 0, 0, np.zeros(len(encoded_count_rle), dtype=np.uint8)
    while current_byte_idx < len(bytes_rle):
        continuous_pixels, shift, high_order_bit = 0, 0, 1

        # When the high order bit of a byte becomes 0, we have decoded the integer and can move on to the next one.
        while high_order_bit:
            byte = bytes_rle[current_byte_idx] - 48  # The encoding uses the ascii chars 48-111.
            # 0x1f is 31, i.e. 001111 --> Here we select the first four bits of the byte.
            continuous_pixels |= (byte & 0x1f) << shift
            # 0x20 is 32 as int, i.e. 2**5, i.e 010000 --> Here we select the fifth bit of the byte.
            high_order_bit = byte & 0x20
            current_byte_idx += 1
            shift += 5
            if (not high_order_bit) and (byte & 0x10):  # 0x20 is 16 as int, i.e. 1000
                continuous_pixels |= (~0 << shift)
        if m > 2:
            continuous_pixels += int(counts[m-2])
        counts[m] = continuous_pixels
        m += 1

    mask = np.zeros(height * width, dtype=np.bool_)
    current_value, current_position = 0, 0
    for nb_pixels in counts:
        mask[current_position:current_position+nb_pixels] = current_value
        current_value = 0 if current_value else 1
        current_position += nb_pixels

    print(height * width)
    print(current_position)
    return mask.reshape(height, width)
    # void rleFrString( RLE *R, char *s, siz h, siz w ) {
    #     siz m=0, p=0, k; long x; int more; uint *cnts;
    #     while( s[m] ) m++; cnts=malloc(sizeof(uint)*m); m=0;
    #     while( s[p] ) {
    #         x=0; k=0; more=1;
    #         while( more ) {
    #             char c=s[p]-48;
    #             x |= (c & 0x1f) << 5*k;
    #             more = c & 0x20; p++; k++;
    #             if(!more && (c & 0x10)) x |= -1 << 5*k;
    #         }
    #         if(m>2) x+=(long) cnts[m-2]; cnts[m++]=(uint) x;
    #     }
    #     rleInit(R,h,w,m,cnts); free(cnts);
    # }
    # void rleInit( RLE *R, siz h, siz w, siz m, uint *cnts ) {
    #     R->h=h; R->w=w; R->m=m; R->cnts=(m==0)?0:malloc(sizeof(uint)*m);
    #     siz j; if(cnts) for(j=0; j<m; j++) R->cnts[j]=cnts[j];
    # }


    # def decode(rleObjs):
    #     cdef RLEs Rs = _frString(rleObjs)
    #     h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
    #     masks = Masks(h, w, n)
    #     rleDecode(<RLE*>Rs._R, masks._mask, n);
    #     return np.array(masks)

    pass

res = decode_rle("iX13a>i0^O:F9H7I8J4K6J6K4L4L4M3L3N3L3N3L3N3M2M4M2N3M2N2N2N2N2N2N2N2O1N2N2N1O2N2O1N2N101N2N101N2N2O0O2O0O2O1N101N101N101O0O2O001N101N10001N10001N10001O0O10001O0O1000001O000000001O0000000000000O1000000000000000000001O00000000000000000O2O00000O2O00000O2O00001N10001N101O0O101N101N101O0O2O0O1O2O1N101N101N2N101N2N101N2N2N1O2N2N2O1N2N2N2N2M3N2N2N2N3M2N2M3N3M2M4M2M4L3N3L4L4L4L5K4K6J6J7H8H9E?^OcVa6", 480, 640)

while True:
    cv2.imshow("a", 255 * res.astype(np.uint8))
    key = cv2.waitKey(10)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break

# def main():
#     parser = argparse.ArgumentParser(description="Tool to visualize coco labels.",
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("data_path", type=Path, help="Path to the directory with the images.")
#     parser.add_argument("json_path", type=Path, help="Path to the json file with the coco annotations.")
#     parser.add_argument("--show_bbox", "-sb", action="store_true", help="Show the bounding boxes.")
#     args = parser.parse_args()

#     data_path: Path = args.data_path
#     json_path: Path = args.json_path
#     show_bbox: bool = args.show_bbox

#     with open(json_path, 'r', encoding="utf-8") as annotations:
#         coco_dataset = json.load(annotations)

#     img_entries: list[Image] = coco_dataset["images"]
#     annotations: list[Annotation] = coco_dataset["annotations"]
#     categories: list[Category] = coco_dataset["categories"]

#     # Get all images containing given categories, select one at random
#     nb_imgs = len(img_entries)
#     for i, img_entry in enumerate(img_entries, start=1):
#         msg = f"Showing image: {img_entry['file_name']} ({i+1}/{nb_imgs})"
#         print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)),
#               end='\r' if i != nb_imgs else '\n', flush=True)

#         img = cv2.imread(str(data_path / img_entry["file_name"]))

#         img_annotations = [annotation for annotation in annotations
#                            if annotation["image_id"] == img_entry["id"]]
#         if show_bbox:
#             # Add the bounding boxes to the image
#             for annotation in img_annotations:
#                 top_x, top_y, width, height = annotation["bbox"]
#                 img = cv2.rectangle(img,
#                                     (int(top_x), int(top_y)),
#                                     (int(top_x+width), int(top_y+height)),
#                                     (255, 0, 0), 5)

#         # Add the segmentation masks
#         plt.imshow(img)
#         plt.axis('off')
#         coco.showAnns(anns)
#         plt.show()


# if __name__ == "__main__":
#     main()


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
