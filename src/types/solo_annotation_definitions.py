from pydantic import BaseModel, Field


class SpecEntry(BaseModel):
    label_id: int
    label_name: str


class BoundingBox2DAnnotation(BaseModel):
    type: str = Field(..., alias="@type")
    id: str
    desciption: str
    spec: list[SpecEntry]


class Keypoint(BaseModel):
    label: str
    index: int
    color: tuple[int, int, int, int]


class Template(BaseModel):
    template_id: str = Field(..., alias="templateId")
    template_name: str = Field(..., alias="templateName")
    keypoints: list[Keypoint]


class KeypointAnnotation(BaseModel):
    type: str = Field(..., alias="@type")
    id: str
    desciption: str
    template: Template


class InstanceSegmentationAnnotation(BaseModel):
    type: str = Field(..., alias="@type")
    id: str
    desciption: str
    spec: list[SpecEntry]


class annotationDefinitions(BaseModel):
    annotationDefinitions: list[BoundingBox2DAnnotation | KeypointAnnotation | InstanceSegmentationAnnotation]
