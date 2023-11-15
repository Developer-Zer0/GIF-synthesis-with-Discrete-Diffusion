from __future__ import annotations
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
from torchvision.models import resnet50, ResNet50_Weights
# import pytorch_lightning as pl
import json

class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

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

        if split == 'val':
            split = 'validate'

        if split != "test":
            split_folder = "train_val_videos"
            annon_file = "train_val_videodatainfo.json"
        else:
            split_folder = "test_videos"

        self.video_id_to_sentence = {}
        videos_split_list = []
        annotations = open(osp.join(data_folder, "train_val_annotation", annon_file))
        a = json.load(annotations)
        sent_list = a['sentences']
        for sent in sent_list:
            try:
                self.video_id_to_sentence[sent['video_id']].append(sent['caption'])
            except KeyError:
                self.video_id_to_sentence[sent['video_id']] = [sent['caption']]
        vid_list = a['videos']
        for vid in vid_list:
            if split == vid['split']:
                videos_split_list.append(osp.join(data_folder, split_folder, "TrainValVideo", vid['video_id'])+'.mp4')
        annotations.close()

        warnings.filterwarnings('ignore')
        cache_file = osp.join(osp.join(data_folder, split_folder), f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file) or True:
            clips = VideoClips(videos_split_list[:len(videos_split_list)//4], sequence_length, 100, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(videos_split_list, sequence_length, 100,
                               _precomputed_metadata=metadata)
        self._clips = clips

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        video, _, _, idx = self._clips.get_clip(idx)
        orig_length = video.shape[0]

        sent_list = self.video_id_to_sentence[self._clips.video_paths[idx].split('/')[-1].replace(".mp4", "")]
        rand = random.randint(0, len(sent_list)-1)
        text = sent_list[rand]

        video = preprocess(video, resolution)

        # Extract frame
        frame = video.permute(1, 0, 2, 3)[0]
        processed_frame = self.frame_preprocess(frame)
        frame_feats = self.resnet(processed_frame)

        return dict(video=video, label=None, length=len(video), orig_length=orig_length, text=text, frame=frame_feats)


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