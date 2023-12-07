import numpy as np
import torch

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

def ucf101_collate(lst_elements):
    batch = {
        "length": [x["length"] for x in lst_elements],
        "orig_length": [x["orig_length"] for x in lst_elements],
        "video": collate_tensors([x["video"] for x in lst_elements]),
        "frame": collate_tensors([x["frame"] for x in lst_elements]),
        "label": [x["label"] for x in lst_elements],
        "text": [x["text"] for x in lst_elements],
    }
    return batch

def msrvtt_collate(lst_elements):
    batch = {
        "length": [x["length"] for x in lst_elements],
        "orig_length": [x["orig_length"] for x in lst_elements],
        "video": collate_tensors([x["video"] for x in lst_elements]),
        # "frame": collate_tensors([x["frame"] for x in lst_elements]),
        "label": [x["label"] for x in lst_elements],
        "text": [x["text"] for x in lst_elements],
    }
    return batch