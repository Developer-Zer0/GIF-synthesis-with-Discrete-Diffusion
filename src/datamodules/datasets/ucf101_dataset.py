import os
import os.path as osp
import math
import random
import pickle
import warnings

import glob
import numpy as np

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights


class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']
    class_names = ['BreastStroke', 'BaseballPitch']
    # class_names = ['FrisbeeCatch','Swing','Mixing','SkateBoarding','CricketBowling','Punch','BreastStroke','Rowing',
    # 'CuttingInKitchen','PlayingFlute','FloorGymnastics','BoxingPunchingBag',
    # 'IceDancing','TaiChi','Nunchucks','ThrowDiscus',
    # 'BenchPress','Biking','BalanceBeam','BodyWeightSquats','ApplyEyeMakeup','BaseballPitch','HighJump','Typing','JugglingBalls',
    # 'SalsaSpin','VolleyballSpiking','PlayingCello','SumoWrestling','BrushingTeeth','Skijet','PlayingTabla','Hammering','Archery',
    # 'HorseRiding','LongJump','MilitaryParade','BasketballDunk','ApplyLipstick','HammerThrow','Fencing','RockClimbingIndoor',
    # 'Knitting','HeadMassage','PoleVault','CricketShot','HorseRace','PushUps','StillRings','Billiards','BlowingCandles']

    def __init__(self, data_folder, sequence_length, split="train", resolution=64, **kwargs):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.split = split
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.frame_preprocess = weights.transforms()

        folder = osp.join(data_folder, split)
        files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])

        files = [f for f in files if get_parent_dir(f) in self.class_names]

        # hacky way to compute # of classes (count # of unique parent directories)
        self.classes = list(set([get_parent_dir(f) for f in files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file) or True:
            clips = VideoClips(files, sequence_length, 100, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, sequence_length, 100,
                               _precomputed_metadata=metadata)
        self._clips = clips

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        video, _, _, idx = self._clips.get_clip(idx)
        orig_length = video.shape[0]

        class_name = get_parent_dir(self._clips.video_paths[idx])
        label = self.class_to_label[class_name]

        video = preprocess(video, resolution)
        
        Extract frame
        frame = video.permute(1, 0, 2, 3)[0]
        processed_frame = self.frame_preprocess(frame)
        frame_feats = self.resnet(processed_frame)

        if video.shape[2] == 8:
            video = torch.repeat_interleave(video, 2, dim=2)
        elif video.shape[2] == 4:
            video = torch.repeat_interleave(video, 4, dim=2)

        return dict(video=video, label=label, length=len(video), orig_length=orig_length, text=class_name, frame=frame_feats)


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.float() / 255.

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    video = (video - mean) / std

    video = video.permute(0, 3, 1, 2) # TCHW

    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    # video -= 0.5

    return video
