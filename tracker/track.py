import os
import torch
from cotracker.predictor import CoTrackerPredictor
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
video = read_video_from_path('./assets/apple.mp4')
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()


model = CoTrackerPredictor(
    checkpoint=os.path.join(
        './checkpoints/cotracker_stride_4_wind_8.pth'
    )
)
if torch.cuda.is_available():
    model = model.cuda()
    video = video.cuda()

pred_tracks, pred_visibility = model(video, grid_size=30)
vis = Visualizer(save_dir='./videos', pad_value=100)
vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='teaser')