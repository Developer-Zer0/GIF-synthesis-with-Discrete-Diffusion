from pathlib import Path
from DetUtil.UtilRotations import axis_angle_to
from src.datamodules.datasets.transforms.smpl import RotTransDatastruct
import numpy as np
import torch

def get_split_keyids(path: str, split: str):
    filepath = Path(path) / split
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")



def smpl_data_to_matrix_and_trans(data, nohands=True):
    trans = data["trans"]
    nframes = len(trans)

    axis_angle_poses = data["poses"]
    axis_angle_poses = data["poses"].reshape(nframes, -1, 3)

    if nohands:
        axis_angle_poses = axis_angle_poses[:, :22]

    matrix_poses = axis_angle_to("matrix", axis_angle_poses)

    return RotTransDatastruct(rots=matrix_poses, trans=trans)

# TODO: use a real subsampler..
def subsample(num_frames, last_framerate, new_framerate):
    step = int(last_framerate / new_framerate)
    assert step >= 1
    frames = np.arange(0, num_frames, step)
    return frames


# TODO: use a real upsampler..
def upsample(motion, last_framerate, new_framerate):
    step = int(new_framerate / last_framerate)
    assert step >= 1

    # Alpha blending => interpolation
    alpha = np.linspace(0, 1, step+1)
    last = np.einsum("l,...->l...", 1-alpha, motion[:-1])
    new = np.einsum("l,...->l...", alpha, motion[1:])

    chuncks = (last + new)[:-1]
    output = np.concatenate(chuncks.swapaxes(1, 0))
    # Don't forget the last one
    output = np.concatenate((output, motion[[-1]]))
    return output

# collate functions
from typing import List, Dict
from torch import Tensor
def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_datastruct_and_text(lst_elements: List) -> Dict:
    collate_datastruct = lst_elements[0]["datastruct"].transforms.collate

    if True:
        batch = {
            # Collate with padding for the datastruct
            "datastruct": collate_datastruct([x["datastruct"] for x in lst_elements]),
            # Collate normally for the length
            "length": [x["length"] for x in lst_elements],
            "orig_length": [x["orig_length"] for x in lst_elements],
            # Collate the text
            "caption": [x["text"]["caption"] for x in lst_elements],
            "word_embs": [x["text"]["word_embs"] for x in lst_elements],
            "cap_lens": [x["text"]["cap_lens"] for x in lst_elements],
            "pos_onehot": [x["text"]["pos_onehot"] for x in lst_elements]}
    else:
        batch = {
            # Collate with padding for the datastruct
            "datastruct": collate_datastruct([x["datastruct"] for x in lst_elements]),
            # Collate normally for the length
            "length": [x["length"] for x in lst_elements],
            # Collate the text
            "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch


def collate_text_and_length(lst_elements: Dict) -> Dict:
    batch = {"length": [x["length"] for x in lst_elements],
             "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch and x != "datastruct"]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]
    return batch

######## Tevet Collate_fn

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text']["caption"] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    lengthCollate = [b["length"] for b in batch]
    collate_datastruct = batch[0]["datastruct"].transforms.collate
    datastructCollate = collate_datastruct([b["datastruct"] for b in batch])
    textCollate = [b["text"] for b in batch]
    origlensCollate = [b["orig_length"] for b in batch]
    wordembsCollate = [b["text"]['word_embs'] for b in batch]
    caplensCollate = [b["text"]['cap_lens'] for b in batch]
    posohotCollate = [b["text"]['pos_onehot'] for b in batch]

    return {'model_input': (motion, cond), 'length': lengthCollate, 'datastruct': datastructCollate, 'text': textCollate,
            'orig_length': origlensCollate, 'word_embs': wordembsCollate, 'cap_lens': caplensCollate, 'pos_onehot': posohotCollate}

def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b["datastruct"].features.T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]       # check if this is correct
        'text': b["text"], #b[0]['caption']
        'orig_length': b["orig_length"],
        # 'lengths': b[5],
        "length": b["length"],
        "datastruct": b["datastruct"],
        'word_embs': b["text"]["word_embs"],
        'cap_lens': b["text"]["cap_lens"],
        'pos_onehot': b["text"]["pos_onehot"],
    } for b in batch]
    return collate(adapted_batch)