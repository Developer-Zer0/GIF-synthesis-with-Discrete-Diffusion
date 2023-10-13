from src.datamodules.base import BaseTextMotionDataModule
from src.datamodules.datasets.humanml3d_dataset import HumanML3D
from hydra.utils import instantiate


class HumanMLDataModule(BaseTextMotionDataModule):
    def __init__(self, collate_fn=None, data_dir: str = "",
                 batch_size: int = 32,
                 num_workers: int = 16,
                 **kwargs):

        if collate_fn == 't2m_collate':
            from .datasets.data_utils import t2m_collate
            self.collate_fn = t2m_collate
        else:
            from .datasets.data_utils import collate_datastruct_and_text
            self.collate_fn = collate_datastruct_and_text

        super().__init__(collate_fn=self.collate_fn,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         )
        self.save_hyperparameters(logger=False)
        self.Dataset = HumanML3D
        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)

        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        self.transforms = self._sample_set.transforms