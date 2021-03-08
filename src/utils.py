import cv2
import numpy as np


def draw_box(img, box, color=(0, 255, 0), thickness=2):
    cv2.rectangle(
        img,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        [int(c) for c in color],
        thickness,
    )


def draw_vector(img, vector, color=(0, 255, 0), thickness=2):
    cv2.arrowedLine(
        img,
        (int(vector[0]), int(vector[1])),
        (int(vector[2]), int(vector[3])),
        [int(c) for c in color],
        thickness,
    )


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def iou(box1, box2):
    """Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims([box1], 0)
    bb_test = np.expand_dims([box2], 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o[0, 0]


def codirected_vectors(vector, t_vector):
    vector = [vector[0], vector[1], vector[2], vector[3]]
    t_vector = [t_vector[0], t_vector[1], t_vector[2], t_vector[3]]
    if np.dot(vector, t_vector) < 0:
        return False
    return True
