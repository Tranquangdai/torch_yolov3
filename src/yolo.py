import numpy as np
import torch
from src.yolo3 import PostProcess, YoloBody
from src.yolo3.utils import letterbox_image


class Yolo(object):

    _defaults = {
        "model_path": 'model_data/yolo_body.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self._load_model()

    def detect_image(self, image):
        image_shape, img = self._load_img(image)
        img = torch.Tensor(img).permute(0, 3, 1, 2)
        with torch.no_grad():
            out_boxes, out_scores, out_classes = self.post_processor.yolo_eval(
                self.yolo_body(img),
                self.anchors,
                self.num_classes,
                image_shape)
        return out_boxes.numpy(), out_scores.numpy(), out_classes.numpy()

    def _load_model(self):
        num_anchors = len(self.anchors)
        self.num_classes = len(self.class_names)
        self.yolo_body = YoloBody(num_anchors // 3, self.num_classes)
        state_dict = torch.load(self.model_path)
        self.yolo_body.load_state_dict(state_dict)
        self.yolo_body.eval()
        self.post_processor = PostProcess()

    def _load_img(self, image):
        image_shape = image.size[::-1]
        if self.model_image_size != (None, None):
            assert self.model_image_size[
                0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[
                1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image,
                                          tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        return torch.Tensor(image_shape), image_data

    def _get_anchors(self):
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_class(self):
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
