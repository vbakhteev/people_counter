import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from configs.fairmot import opt
from src.io import get_video_capture, get_video_writer
from src.lib.datasets.dataset.jde import letterbox
from src.lib.tracker.multitracker import JDETracker
from src.utils import draw_box, get_color, iou


def parse_args():
    parser = argparse.ArgumentParser(description='People counter based on people detection and tracking')
    parser.add_argument('video', type=str, help='Path to the video')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('weights', type=str, help='Path to the weights of people detector')
    parser.add_argument('--output', type=str, default='outputs/body_detection.mp4', help='Path to the output')
    parser.add_argument('--start_sec', type=int, default=0, help='Process file from this second')
    parser.add_argument('--duration', type=int, default=20, help='How much seconds to process')

    return parser.parse_args()


def main():
    args = parse_args()
    door_boxes = [(615, 40, 718, 160)]
    # door_boxes = [(1250, 70, 1440, 310)]

    cap, meta = get_video_capture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(meta['fps'] * args.start_sec))
    out = get_video_writer(args.output, meta)

    tracker = JDETracker(opt, frame_rate=meta['fps'])
    entered_doors_tracks = dict()  # track_id -> [list of coords]
    entered_doors = 0

    n_frames = int(args.duration * meta['fps'])
    for _ in tqdm(range(n_frames)):
        it_worked, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not it_worked:
            break

        # Detect and track
        boxes, track_ids = predict_image(img, tracker, (1088, 608))

        # Check if person is in doors
        for box, track_id in zip(boxes, track_ids):
            in_doors = False
            for door_box in door_boxes:
                in_doors = (iou(door_box, box) > 0)
                if in_doors:
                    break

            if not in_doors:
                # if person just left a door, then check his direction and remove his track
                if track_id in entered_doors_tracks:
                    track = entered_doors_tracks[track_id]
                    diff = track[-1][0] - track[0][0]
                    if diff > 50:
                        entered_doors += 1

                    del entered_doors_tracks[track_id]

                continue

            if track_id in entered_doors_tracks:
                entered_doors_tracks[track_id].append((box[0], box[1]))
            else:
                entered_doors_tracks[track_id] = [(box[0], box[1])]

        # TODO if track in entered_doors_tracks long enough then check its direction

        # Draw doors
        for box in door_boxes:
            draw_box(img, box, color=(255, 0, 0), thickness=5)
        # Draw boxes
        for box, track_id in zip(boxes, track_ids):
            draw_box(img, box, color=get_color(track_id))
        # Write number of entrances
        cv2.putText(img, str(entered_doors), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10)

        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print('Finished!')


def predict_image(img0, tracker, img_shape, w=1920, h=1080):
    # img0 = cv2.resize(img0, (w, h))
    img, _, _, _ = letterbox(img0, height=img_shape[0], width=img_shape[1])
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    # run tracking
    blob = torch.from_numpy(img).cuda().unsqueeze(0)
    online_targets = tracker.update(blob, img0)
    online_tlwhs = []
    online_ids = []

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)

    boxes = np.array(online_tlwhs).astype(int)
    if len(boxes):
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

    return boxes, online_ids


if __name__ == '__main__':
    main()
