from setuptools import find_packages, setup

setup(
    name="coco_utils",
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "matplotlib",
        "pycocotools"
    ],
)
