'''
    time:2022-11-16
    author:xwd
'''

import argparse
import os
import glob
import cv2

import numpy as np
from iou_tracker import *
from lib.utils import *

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='IOU-Tracker demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true', default=True)
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--sigma_l",
                        help="lowest ehreshold score of detections",
                        type=float, default=0.3)
    parser.add_argument("--sigma_h",
                        help="highest threshold score of detections",
                        type=float, default=0.8)
    parser.add_argument("--sigma_iou", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--t_min", help="Minimum detections.", type=int, default=3)
    parser.add_argument("--disappear_time", help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=3)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase

    mot_dir = '/media/calyx/Windy/wdy/data_online/MOT15'

    colors = create_colors()

    if (display):
        if not os.path.exists(mot_dir):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = IouTracker(sigma_l=args.sigma_l,
                                 sigma_h=args.sigma_h,
                                 sigma_iou = args.sigma_iou,
                                 t_min = args.t_min,
                                 disappear_time = args.disappear_time)  # create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        print("Processing %s." % (seq))
        for frame in range(int(seq_dets[:, 0].max())):
            frame += 1  # detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
            dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

            if (display):
                fn = os.path.join(mot_dir, phase, seq, 'img1', '%06d.jpg' % (frame))
                img = cv2.imread(fn)

            trackers = mot_tracker.update(dets)

            for d in trackers:
                if (display):
                    for i, track in enumerate(trackers):
                        bbox = track['bbox']
                        score = track['score']
                        idx = track['id']
                        life = track['life']
                        draw_image(img, bbox, text='ID:%s,life:%s' % (idx, life), color=colors[idx])

            if(display):
                cv2.imshow('img', img)
                cv2.waitKey(30)