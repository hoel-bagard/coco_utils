# coco_utils
Utility script for handling COCO-style data

## Installation
Requires python version >= 3.9

### Dependencies
opencv-python\
matplotlib


### Clone the repository
```
git clone git@github.com:hoel-bagard/coco_utils.git
```

#### Convert
TODO

#### Merge
If necessary merge several coco json with:
```
python utils/merge_coco.py <path_to_first_json> <path_to_second_json>
python utils/merge_coco.py ../data/second_dataset/vott-json-export_brown/coco_annotations.json ../data/second_dataset/vott-json-export_blue/coco_annotations.json ../data/second_dataset/first_dataset/coco_annotations.json
```

#### Resize
Resize the dataset (if needed):
```
python utils/resize_coco.py <path_to_image_dir> <path_to_json_annotations> <output_path> <size1> <size2>
python utils/resize_coco.py ../data/original_dataset/train/images/ ../data/original_dataset/train/annotations.json ../data/resized_dataset/train 550 550
```

#### Rename
Change all the ids from strings to ints:
```
python utils/coco_ids_to_int.py <path_to_annotation_file>
python utils/coco_ids_to_int.py ../data/train/annotations.json
```

#### Grayscale (remove that part ?)
Convert images to grayscale if desired:
```
python utils/imgs_to_grayscale.py <path_to_image_folder>
python utils/imgs_to_grayscale.py ../data/validation/images/
```

#### Split
Split the dataset into train and validation datasets:
```
python utils/split_train_val.py <path to image folder> <path to annotation file> <output path>
python utils/split_train_val.py ../data/original_dataset ../data/original_dataset/coco_annotations.json ../data/split_dataset/
```

#### Visualize the data
Finally, check that everything works as expected by using:
```
python utils/visualize_coco_data.py <path to image folder> <path to annotation file>
python utils/visualize_coco_data.py ../data/train/images/ ../data/train/annotations.json
```
