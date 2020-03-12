import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self._load_model()

    def detect_image(self, image, draw_bbox=True, draw_label=False):
        image_shape, img = self._load_img(image)
        img = torch.Tensor(img).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            out_boxes, out_scores, out_classes = self.post_processor.yolo_eval(
                self.yolo_body(img),
                self.anchors,
                self.num_classes,
                torch.Tensor(image_shape).to(self.device))

        out_boxes = out_boxes.cpu().numpy()
        out_scores = out_scores.cpu().numpy()
        out_classes = out_classes.cpu().numpy()

        if not draw_bbox:
            return out_boxes, out_scores, out_classes
        else:
            return self.draw_bbox(image,
                                  out_boxes, out_scores, out_classes,
                                  draw_label)

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

    def _load_model(self):
        num_anchors = len(self.anchors)
        self.num_classes = len(self.class_names)
        self.yolo_body = YoloBody(num_anchors // 3, self.num_classes)
        state_dict = torch.load(self.model_path)
        self.yolo_body.load_state_dict(state_dict)
        self.yolo_body.to(self.device)
        self.yolo_body.eval()
        self.post_processor = PostProcess()
        self.colors = self._get_colors()

    def _get_colors(self):
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255),
                                     int(x[1] * 255),
                                     int(x[2] * 255)), colors))
        return colors

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
        return image_shape, image_data

    def draw_bbox(self, image,
                  out_boxes, out_scores, out_classes,
                  draw_label=False):
        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            if draw_label:
                print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image
