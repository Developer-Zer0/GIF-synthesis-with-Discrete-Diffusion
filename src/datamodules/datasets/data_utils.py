import numpy as np
import torch

def ufc101_collate(batch):
    batch = {
            "length": [x["length"] for x in lst_elements],
            "orig_length": [x["orig_length"] for x in lst_elements],
            "video": [x["video"] for x in lst_elements],
            "label": [x["label"] for x in lst_elements],
            "text": [x["text"] for x in lst_elements],
    return batch
