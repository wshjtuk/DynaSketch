# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
import sys


def track(target, mask, grid_size, fps1, fps2, device):
    # DEFAULT_DEVICE = ('cuda' if device else 'cpu')
    DEFAULT_DEVICE = ('cpu')
    # frames = [np.array(np.asarray(img1)), np.array(np.asarray(img2))]
    # video = np.stack(frames)
    # # load the input video frame by frame
    video = read_video_from_path(target)[fps1:fps2 + 1]
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    mask = np.mean(mask, axis=2, keepdims=True)
    mask = np.squeeze(mask, axis=-1)
    segm_mask = np.array(mask)
    segm_mask = torch.from_numpy(segm_mask)[None, None]
    model = CoTrackerPredictor(
        checkpoint="/data/wujk2022/2023/CLIPasso/tracker/checkpoints/cotracker_stride_4_wind_8.pth")
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)
    pred_tracks, pred_visibility = model(
        video,
        grid_size=grid_size,
        grid_query_frame=0,
        backward_tracking=True,
        segm_mask=segm_mask
    )
    print("computed track")
    vis = Visualizer(save_dir="/data/wujk2022/2023/CLIPasso/tracker/saved_videos", pad_value=120, linewidth=3)
    vis.visualize(video, pred_tracks, pred_visibility, query_frame=0)
    return pred_tracks[:, -1, :, :].squeeze().to('cuda'), pred_visibility[:, -1, :].squeeze().to('cuda'), pred_tracks[:, 0, :, :].squeeze().to('cuda')