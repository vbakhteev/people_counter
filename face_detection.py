import argparse

import cv2
import torch
from tqdm import tqdm

from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine

from src.sort import Sort
from src.utils import iou, get_color, draw_box
from src.io import get_video_capture, get_video_writer


def parse_args():
    parser = argparse.ArgumentParser(description='People counter based on face detection')
    parser.add_argument('video', type=str, help='Path to the video')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('weights', type=str, help='Path to the weights of face detector')
    parser.add_argument('--output', type=str, default='outputs/face_detection.mp4', help='Path to the output')
    parser.add_argument('--start_sec', type=int, default=0, help='Process file from this second')
    parser.add_argument('--duration', type=int, default=20, help='How much seconds to process')

    return parser.parse_args()


def main():
    args = parse_args()
    door_boxes = [(1250, 70, 1440, 310)]

    cfg = Config.fromfile(args.config)
    engine, data_pipeline, device = prepare(cfg, weights_path=args.weights)

    vidcap, video_meta = get_video_capture(args.video)
    fps = video_meta['fps']
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * args.start_sec))
    out = get_video_writer(args.output, video_meta)

    tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.05)
    entered_doors = set()

    n_frames = int(args.duration * fps)
    for _ in tqdm(range(n_frames)):
        it_worked, img = vidcap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not it_worked:
            break

        # Detection
        result = infer_image(img, engine, data_pipeline, device)
        boxes = result[0][:, :-1]

        # Tracking
        tracks = tracker.update(boxes).astype(int)
        boxes = tracks[:, :4].astype(int)
        track_ids = tracks[:, 4]

        # Check if person enters doors
        for box, track_id in zip(boxes, track_ids):
            enters = False
            for door_box in door_boxes:
                enters = (iou(door_box, box) > 0)
                if enters:
                    break

            if enters:
                entered_doors.add(track_id)

        # Draw doors
        for box in door_boxes:
            draw_box(img, box, color=(255, 0, 0), thickness=5)
        # Draw faces
        for box, track_id in zip(boxes, track_ids):
            draw_box(img, tuple(box), color=get_color(track_id))
        # Write number of entrances
        cv2.putText(img, str(len(entered_doors)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10)

        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def prepare(cfg, weights_path):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, weights_path)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def infer_image(img, engine, data_pipeline, device):
    data = dict(
        filename='',
        ori_filename='',
        img=img,
        img_shape=img.shape,
        ori_shape=img.shape,
        img_fields=['img'],
        img_prefix=None,
    )
    data = data_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if device != 'cpu':
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data

    result = engine.infer(data['img'], data['img_metas'])[0]

    return result


if __name__ == '__main__':
    main()
