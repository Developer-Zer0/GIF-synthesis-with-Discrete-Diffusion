import numpy as np
import pandas
import torch
import logging
from torch import nn
from typing import Dict, Optional
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from pathlib import Path
import os

from src.datamodules.datasets.data_utils import get_split_keyids, smpl_data_to_matrix_and_trans, subsample
from src.datamodules.datasets.transforms import Transform

from src.datamodules.datasets.word_vectorizer import WordVectorizer, POS_enumerator
import codecs as cs
import spacy
import os
from os.path import join as pjoin
import random

logger = logging.getLogger(__name__)

class HumanML3D(Dataset):
    dataname = "HumanML3D"
    def __init__(self, datapath: str,
                splitpath: str,
                transforms: Transform,
                split: str = "train",
                sampler=None,
                framerate: float = 20,
                joints_num: int = 22,
                max_motion_length: int = 196,
                feat_bias: int = 5,
                deps: str = None,
                progress_bar: bool = True,
                tiny: bool = False,
                devices: str = 'cpu', **kwargs):

        self.gpu_device = ''

        if devices == 'cpu':
            self.gpu_device = devices
        else:
            self.gpu_device = 'cuda:' + str(devices[0])

        self.motion_dir = pjoin(datapath, 'new_joint_vecs')
        self.text_dir = pjoin(datapath, 'texts')
        self.nlp = spacy.load('en_core_web_sm')
        vectorizer_path = deps + '/word_vectorizer'
        self.w_vectorizer = WordVectorizer(vectorizer_path, 'our_vab')
        test = len(self.w_vectorizer)
        print(test)
        self.split = split
        self.transforms = transforms
        self.max_length = 196
        self.pointer = 0
        min_motion_len = 40

        self.cf_drop_caption_rate = 0.15

        # joints_num = 22
        # feat_bias = 5
        # same as above issue. Should either refactor it or make sure the meta data comes with code checkout.
        meta_dir = pjoin('data/ankur', 'meta')
        os.makedirs(meta_dir, exist_ok=True)
        mean = np.load(pjoin(datapath, 'mean.npy'))
        std = np.load(pjoin(datapath, 'std.npy'))
        self.max_motion_length = max_motion_length
        self.unit_length = 4
        self.max_text_len = 20

        data_dict = {}
        # id_list = []
        # with cs.open(split_file, 'r') as f:
        #     for line in f.readlines():
        #         id_list.append(line.strip())
        id_list = get_split_keyids(path=splitpath, split=split+'.txt')

        if progress_bar:
            enumerator = enumerate(tqdm(id_list, f"Loading HumanML3D {split}"))
        else:
            enumerator = enumerate(id_list)

        if tiny:
            maxdata = 2
        else:
            maxdata = np.inf

        new_name_list = []
        length_list = []
        for name in tqdm(id_list, f"Loading HumanML3D {split}"):
            if len(data_dict) >= maxdata:
                break

            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20): int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if False:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(meta_dir, 'mean.npy'), mean)
            np.save(pjoin(meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        # self.reset_max_len(self.max_length)

        self.nfeats = len(self[0]["datastruct"].features[0])

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        try:
            data['pred_data'].features = data['pred_data'].features.cpu() * self.std + self.mean
            data['gt_data'].features = data['gt_data'].features.cpu() * self.std + self.mean
            data['pred_single_step'].features = data['pred_single_step'].features.cpu() * self.std + self.mean
            data['datastruct_test'].features = data['datastruct_test'].features.cpu() * self.std + self.mean
        except:
            pass
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item

        # Selecting a particular sample for rendering/testing
        # if len(self.name_list) > 1000:
        #     idx = 3135

        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        # Disable random caption selecting for testing
        # text_data = text_list[0]
        # Randomly select a caption
        text_data = random.choice(text_list)

        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.unit_length

        if False:
            if m_length != self.max_length:
                # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx + self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.unit_length - 1) * self.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        #
        # print(m_length, motion.shape)

        motion = torch.tensor(motion).to(self.gpu_device)
        datastruct = self.transforms.Datastruct(features=motion)

        if self.split != 'train' or random.random() > self.cf_drop_caption_rate:
            text_dict = {'word_embs': word_embeddings, 'pos_onehot': pos_one_hots, 'caption': caption, 'cap_lens': sent_len}
        else:
            empty_tokens = ['sos/OTHER'] + ['eos/OTHER'] + ['unk/OTHER'] * (self.max_text_len)
            pos_one_hots = []
            word_embeddings = []
            for e_token in empty_tokens:
                word_emb, pos_oh = self.w_vectorizer[e_token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            text_dict = {'word_embs': word_embeddings, 'pos_onehot': pos_one_hots, 'caption': '', 'cap_lens': 2}

        element = {"datastruct": datastruct, "text": text_dict,
                   "length": len(datastruct), "keyid": idx, "orig_length": m_length}

        return element
