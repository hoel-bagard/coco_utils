from pydantic import validator

from src.types.coco_object_detection import Annotation, Category, Dataset


class AnnotationKP(Annotation):
    keypoints: list[int]
    num_keypoints: int

    @validator("keypoints")
    def keypoints_length(cls, keypoints: list[int]):  # noqa: N805
        assert len(keypoints) % 3 == 0, (
            "Keypoints should be a length 3k array where k is the total number of keypoints defined for the category")
        return keypoints

    @validator("keypoints")
    def keypoints_visibility_flag(cls, keypoints: list[int]):  # noqa: N805
        for i in range(2, len(keypoints), 3):
            assert keypoints[i] in (0, 1, 2), (
                f"The visibility flag can only take the values 0 (nto labeled), 1 (labeled but not visible) "
                f"and 2 (labeled and visible). Got {keypoints[i]}")
        return keypoints

    @validator("num_keypoints")
    def num_keypoints_matches_keypoints_length(cls, num_keypoints: int, values: dict[str, list[int]]):  # noqa: N805
        if len(values["keypoints"]) // 3 != num_keypoints:
            raise ValueError(f"Length of the keypoints list ({len(values['keypoints'])}) does not match "
                             f"the number of keypoints ({num_keypoints}).")
        return num_keypoints


class CategoryKP(Category):
    keypoints: list[str]
    skeleton: list[tuple[int, int]]


class DatasetKP(Dataset):
    annotations: list[AnnotationKP]
    categories: list[CategoryKP]
