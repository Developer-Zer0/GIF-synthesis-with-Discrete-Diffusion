from src.datamodules.base import BaseTextVideoDataModule
from src.datamodules.datasets.msrvtt_dataset import VideoDataset
from hydra.utils import instantiate


class MSRVTTDataModule(BaseTextVideoDataModule):
    def __init__(self, collate_fn=None, data_dir: str = "",
                 batch_size: int = 32,
                 num_workers: int = 16,
                 **kwargs):

        from .datasets.data_utils import msrvtt_collate
        self.collate_fn = ucf101_collate
        
        super().__init__(collate_fn=self.collate_fn,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         )
        self.save_hyperparameters(logger=False)
        self.Dataset = VideoDataset
        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)

        # Get additional info of the dataset
        # TODO Add resolution and sequence length
        # self.nfeats = self._sample_set.nfeats
