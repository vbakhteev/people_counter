from pathlib import Path
from typing import Union

import cv2
import numpy as np


def imread(path: Union[Path, str]):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwrite(path: Union[Path, str], img: np.array):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)


def get_video_capture(path: Union[Path, str]):
    cap = cv2.VideoCapture(str(path))
    meta = dict(
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    return cap, meta


def get_video_writer(path: Union[Path, str], meta):
    shape = (meta['width'], meta['height'])
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MP4V'), meta['fps'], shape)
    return out
