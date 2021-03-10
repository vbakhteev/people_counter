import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from configs.fairmot import opt
from src.io import get_video_capture, get_video_writer
from src.lib.datasets.dataset.jde import letterbox
from src.lib.tracker.multitracker import JDETracker
from src.sort import Sort
from src.utils import draw_box, get_color, iou, draw_vector, draw_polygon, codirected_vectors, create_door_vectors


def parse_args():
    parser = argparse.ArgumentParser(description='People counter based on people detection and tracking')
    parser.add_argument('video', type=str, help='Path to the video')
    parser.add_argument('weights', type=str, help='Path to the weights of people detector')
    parser.add_argument('--output', type=str, default='outputs/body_detection.mp4', help='Path to the output')
    parser.add_argument('--start_sec', type=int, default=0, help='Process file from this second')
    parser.add_argument('--duration', type=int, default=20, help='How much seconds to process')

    return parser.parse_args()


def main():
    args = parse_args()

    # door_boxes = [(1250, 70, 1240, 300, 1430, 310, 1450, 80)]
    # door_boxes = [(615, 40, 615, 160, 718, 160, 718, 40)]
    door_boxes = [(625, 35, 615, 155, 710, 160, 718, 44)]
    door_vectors = create_door_vectors(door_boxes)

    cap, meta = get_video_capture(args.video)
    num_frames = cv2.CAP_PROP_FRAME_COUNT
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(meta['fps'] * args.start_sec))
    out = get_video_writer(args.output, meta)

    tracker = JDETracker(opt, frame_rate=meta['fps'])
    sort_tracker = None
    # sort_tracker = Sort(4)
    entered_doors_tracks = dict()  # track_id -> [list of coords]
    entered_doors = 0

    n_frames = min(int(args.duration * meta['fps']), int(cap.get(num_frames) - meta['fps'] * args.start_sec))
    for _ in tqdm(range(n_frames)):
        it_worked, img = cap.read()
        if not it_worked:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect and track
        boxes, track_ids = predict_image(img, tracker, (1088, 608), sort_tracker=sort_tracker)

        # Check if person is in doors
        for box, track_id in zip(boxes, track_ids):
            in_doors = False
            for door_box in door_boxes:
                d_b = (door_box[0], door_box[1], door_box[4], door_box[5])
                in_doors = (iou(d_b, box) > 0)
                if in_doors:
                    break

            if not in_doors:
                # if person just left a door, then check his direction and remove his track
                if track_id in entered_doors_tracks:
                    track = entered_doors_tracks[track_id]
                    track_vector = (track[0][0], track[0][1], track[-1][0], track[-1][1])
                    draw_vector(img, track_vector, color=(0, 255, 0), thickness=3)
                    for vector in door_vectors:
                        if codirected_vectors(vector, track_vector):
                            entered_doors += 1
                            break
                    del entered_doors_tracks[track_id]

                continue

            if track_id in entered_doors_tracks:
                entered_doors_tracks[track_id].append((box[0], box[1]))
            else:
                entered_doors_tracks[track_id] = [(box[0], box[1])]

        # TODO if track in entered_doors_tracks long enough then check its direction

        # Draw doors
        for box in door_boxes:
            draw_polygon(img, box, color=(255, 0, 0), thickness=5)
        # Draw door vectors
        for vector in door_vectors:
            draw_vector(img, vector, color=(0, 255, 0), thickness=3)
        # Draw boxes
        for box, track_id in zip(boxes, track_ids):
            draw_box(img, box, color=get_color(track_id))
        # Write number of entrances
        cv2.putText(img, str(entered_doors), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10)

        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f'Finished! Total entered: {entered_doors}')


def predict_image(img0, mot_tracker, img_shape, sort_tracker=None):
    img = letterbox(img0, height=img_shape[0], width=img_shape[1])[0]
    img = (img.transpose(2, 0, 1) / 255).astype('float32')

    # run tracking
    if opt.gpus[0] >= 0:
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
    else:
        blob = torch.from_numpy(img).unsqueeze(0)
    dets, id_feature = mot_tracker.predict(blob, img0)
    if sort_tracker is None:
        online_targets = mot_tracker.update(dets, id_feature)
        online_targets = [(t.tlwh, t.track_id) for t in online_targets]
    else:
        online_targets = sort_tracker.update(dets)
        online_targets = [([t[0], t[1], t[2] - t[0], t[3] - t[1]], t[4]) for t in online_targets]

    online_tlwhs = []
    online_ids = []

    for tlwh, tid in online_targets:
        # tlwh = t.tlwh
        # tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)

    boxes = np.array(online_tlwhs, dtype=int)
    if len(boxes):
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

    return boxes, online_ids


if __name__ == '__main__':
    main()
