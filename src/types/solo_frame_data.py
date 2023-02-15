from pydantic import BaseModel, Field

class BBoxValue(BaseModel):
    instanceId: int
    labelId: int
    labelName: str
    origin: tuple[float, float]
    dimension: tuple[float, float]


class BoundingBox2DAnnotation(BaseModel):
    type: str = Field(..., alias="@type")
    id: str
    desciption: str
    values: list[BBoxValue]


class Instance(BaseModel):
    instanceId: int
    labelId: int
    labelName: str
    color: tuple[int, int, int, int]


class InstanceSegmentationAnnotation(BaseModel):
    type: str = Field(..., alias="@type")
    id: str
    sensorId: str
    description: str
    imageFormat: str
    dimension: tuple[float, float]
    filename: str
    instances: list[Instance]


class Keypoint(BaseModel):
    index: int
    location: tuple[float, float]
    color: tuple[int, int, int, int]
    cameraCartesianLocation: tuple[float, float, float]
    state: int


class KeypointValue(BaseModel):
    instanceId: int
    labelId: int
    pose: str
    keypoints: list[Keypoint]


class KeypointAnnotation(BaseModel):
    type: str = Field(..., alias="@type")
    id: str
    sensorId: str
    description: str
    templateId: str
    values: list[KeypointValue]


class Capture(BaseModel):
    type: str = Field(..., alias="@type")
    id: str
    desciption: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    filename: str
    imageFormat: str
    dimension: tuple[float, float]
    projection: str
    matrix: tuple[float, float, float, float, float, float, float, float, float]
    annotations: list[BoundingBox2DAnnotation]
