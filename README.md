# Torch_yolov3

## About:
- A pytorch implementation of yoloV3.
- Inspired by [keras-yolo3](https://github.com/qqwweee/keras-yolo3)

## Installation:
- Download YoloV3 pytorch weight `yolo_body.pth` from [Google Drive](https://drive.google.com/file/d/1caOxmU3ZN7pcidamiVYjXjMPEQNqOCio/view?usp=sharing)
- Put it under `model_data` folder
- Run tests

## Requirements
- pytorch
- pytest
- torchvision>=0.4.2
- numpy
- Pillow>=6.2.1
- matplotlib

## Test
    python -m pytest src/tests/
