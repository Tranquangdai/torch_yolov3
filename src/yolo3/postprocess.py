import torch
import torch.nn.functional as F
import torchvision


class PostProcess(object):

    @staticmethod
    def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
        num_anchors = len(anchors)
        anchors_tensor = torch.Tensor(anchors).view(1, 1, 1, num_anchors, 2)
        grid_shape = feats.shape[2:4]

        grid_x, grid_y = torch.meshgrid([torch.arange(0, grid_shape[0]),
                                         torch.arange(0, grid_shape[1])])
        grid = torch.cat([grid_x.view(1, 1, *grid_x.shape),
                          grid_y.view(1, 1, *grid_y.shape)], dim=1).permute(3, 2, 0, 1).float()

        feats = feats.view(-1, num_anchors, num_classes + 5,
                           *grid_shape).permute(0, 3, 4, 1, 2)

        box_xy = (torch.sigmoid(feats[..., :2]) +
                  grid) / torch.Tensor(list(grid_shape[::-1]))
        box_wh = torch.exp(feats[..., 2:4]) * \
            anchors_tensor / input_shape.flip(0)
        box_confidence = torch.sigmoid(feats[..., 4:5])
        box_class_probs = torch.sigmoid(feats[..., 5:])

        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    @staticmethod
    def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy.flip(-1)
        box_hw = box_wh.flip(-1)

        input_shape = input_shape.to(box_yx.dtype)
        image_shape = image_shape.to(box_yx.dtype)

        new_shape = torch.round(
            image_shape * torch.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / (2 * input_shape)
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        boxes = torch.cat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], dim=-1)

        boxes *= torch.cat([image_shape, image_shape])
        return boxes

    def yolo_boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(
            feats, anchors,
            num_classes,
            input_shape)

        boxes = self.yolo_correct_boxes(box_xy,
                                        box_wh,
                                        input_shape,
                                        image_shape)
        boxes = boxes.view(-1, 4)
        box_scores = box_confidence * box_class_probs
        box_scores = box_scores.view(-1, num_classes)
        return boxes, box_scores

    def yolo_eval(self,
                  yolo_outputs,
                  anchors,
                  num_classes,
                  image_shape,
                  score_threshold=.3,
                  iou_threshold=.45):

        num_layers = len(yolo_outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] \
            if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
        input_shape = torch.Tensor(list(yolo_outputs[0].shape[2:4])) * 32

        boxes = []
        box_scores = []
        for i in range(num_layers):
            _boxes, _box_scores = self.yolo_boxes_and_scores(
                yolo_outputs[i],
                anchors[anchor_mask[i]],
                num_classes,
                input_shape,
                image_shape
            )
            boxes.append(_boxes)
            box_scores.append(_box_scores)

        boxes = torch.cat(boxes, 0)
        box_scores = torch.cat(box_scores, 0)

        mask = box_scores > score_threshold
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            class_boxes = boxes[mask[:, c], :]
            class_box_scores = box_scores[mask[:, c], c]
            nms_index = torchvision.ops.nms(
                class_boxes, class_box_scores, iou_threshold)
            class_boxes = class_boxes[nms_index]
            class_box_scores = class_box_scores[nms_index]
            classes = torch.ones_like(class_box_scores).int() * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = torch.cat(boxes_, 0)
        scores_ = torch.cat(scores_, 0)
        classes_ = torch.cat(classes_, 0)
        return boxes_, scores_, classes_
