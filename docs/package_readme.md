# coco_utils

## Loading COCO data

You can load COCO dataset labels into Pydantic objects by using the `Dataset` and `DatasetKP` classes.

Note: This packages loads the data as is and does not create dictionaries mapping ids to lists of annotations/categories.

#### Loading

For an object detection dataset:
```python
with open("path/to/json", encoding="utf-8") as data_file:
    dataset = DatasetKP.parse_raw(data_file.read())
```

For a keypoint detection dataset:
```python
with open("path/to/json", encoding="utf-8") as data_file:
    dataset = DatasetKP.parse_raw(data_file.read())
```


#### Using

```python
for image_entry in dataset.images:
    print(image_entry.file_name)
```
