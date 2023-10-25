import os
import sys
from typing import Any, List
import hydra
from torchmetrics import MetricCollection

from src.models.base import BaseModel
from hydra.utils import instantiate
from omegaconf import DictConfig

# from src.models.metrics.metrics import ComputeMetrics
import random

from src.utils.basic_video_renderer import render_animation

if os.name != "nt":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch

# from src.utils.evaluator import Evaluator

class TextMotionModel(BaseModel):
    """
    Speech Gesture model for 3D gesture motion synthesis from speech audio
    For simple baseline, we use only the MFCC audio features as input.
    For further experiments, we will add text semantics to improve gesture results.
    """

    def __init__(
        self,
        generator: DictConfig,
        losses: DictConfig,
        # nfeats: int,
        # checkpoint_paths: dict,  # Dict of checkpoint paths for either autoencoder or length_estimator or generator
        # evaluator: DictConfig = None,
        lr_args = {},
        render_animations: bool = True,
        do_evaluation: bool = False,
        devices: str = 'cpu',
        **kwargs
    ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.gpu_device = ''

        if devices == 'cpu':
            self.gpu_device = devices
        else:
            self.gpu_device = 'cuda:' + str(devices[0])

        self.save_hyperparameters(ignore=['generator'])
        self.generator =  instantiate(generator, device=self.gpu_device, _recursive_=False)

        # Temporary code to load model

        # state_dict = torch.load(checkpoint_paths)['state_dict']
        # for param_key in list(state_dict.keys()):
        #     new_param_key = param_key[10:]
        #     state_dict[new_param_key] = state_dict.pop(param_key)
        # self.generator.load_state_dict(state_dict)

        # state_dict = torch.load(checkpoint_paths)['state_dict']
        # self.load_state_dict(state_dict)

        self._losses = MetricCollection(
            {split: instantiate(losses,_recursive_ = False)
             for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        # self.metrics = ComputeMetrics()
        self.lr_args = lr_args
        self.render_animations = render_animations

        self.do_evaluation = do_evaluation
        if self.do_evaluation:
            self.evaluator = instantiate(evaluator, nfeats=nfeats, device=self.gpu_device, _recursive_=False)

    def generator_step(self, batch: Any):
        generator_outputs = self.generator(batch)
        #pred_poses = generator_outputs['pred_poses']
        outputs = {}
        outputs.update(generator_outputs)
        outputs['length'] = batch['length']
        return outputs

    # Added inference step for generator
    def sample_generator_step(self, batch: Any):
        generator_outputs = self.generator(batch, do_inference=True)
        #pred_poses = generator_outputs['pred_poses']
        outputs = {}
        outputs.update(generator_outputs)
        outputs['length'] = batch['length']
        return outputs

    def allsplit_step(self, split: str, batch, batch_idx):
        prefix = '%s/'%split
        s = sys.getsizeof(self)
        outputs = self.generator_step(batch)

        if self.do_evaluation and (split != 'train' and len(batch['length']) != 1):
            eval_outputs = self.sample_generator_step(batch)
            self.evaluator.push_vals(batch, batch_idx, eval_outputs['pred_data'].features)
            # self.evaluator.push_vals(batch, batch_idx, outputs['pred_data'].features)

        compute_loss = self.losses[split]
        loss = compute_loss.update(outputs)
        # if split == "val": # compute metrics for validation data
        #     self.metrics.update(outputs)
        return loss

    def allsplit_epoch_end(self, split: str, outputs):
        losses = self.losses[split]
        loss_dict = losses.compute()
        dico = {losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items()}

        # if split == "val":
        #     metrics_dict = self.metrics.compute()
        #     dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items()})

        if self.do_evaluation and (split != 'train'):
            metrics = self.evaluator.evaluate_metrics(self.trainer.datamodule, self.generator)
            dico.update({f"Metrics/{metric}-{split}": value for metric, value in metrics.items()})
            self.evaluator.reset()

        dico.update({"epoch": float(self.trainer.current_epoch),
                     "step": float(self.trainer.current_epoch)})

        if split == 'val' and self.current_epoch % 5 == 0:
            self.render_sample_results()

        self.log_dict(dico)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        b1 = 0.5
        b2 = 0.999
        out_opts = []
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_args['gen_lr'], betas=(b1, b2))
        out_opts.append(opt_g)
        return out_opts, []

    def render_sample_results(self):

        if not self.render_animations:
            return
        
        self.generator.eval()
        # output to current logging dir
        output_dir = self.trainer.log_dir
        sample_dataloader = self.trainer.datamodule.val_dataloader()
        sample_dataset = sample_dataloader.dataset

        device = torch.device(self.trainer.root_gpu if torch.cuda.is_available() else "cpu")
        # randomly get a sample from val dataset
        sample_idx = random.randint(0, len(sample_dataset)-1)
        # create sample batch
        sample_data_batch = sample_dataloader.collate_fn([sample_dataset[sample_idx]])
        # pred_output = self.generator_step(sample_data_batch)
        inference_output = self.sample_generator_step(sample_data_batch)
        
        output_video_file = os.path.join(output_dir, 'epoch%d_synthesis.mp4' % self.current_epoch)
        render_animation(inference_output['pred_data'][0].permute(1, 2, 3, 0).cpu().numpy(), output_path = output_video_file, fps=1)

        output_video_file = os.path.join(output_dir, 'epoch%d_original.mp4' % self.current_epoch)
        render_animation(inference_output['gt_data'][0].permute(1, 2, 3, 0).cpu().numpy(), output_path = output_video_file, fps=1)

        self.generator.train()

        # if sample_dataset.dataname == "HumanML3D":
        #     inference_output = sample_dataset.inv_transform(inference_output)

        # try:
        #     joints_np = inference_output['pred_m2m'].joints.cpu().numpy()[0] # only one batch
        #     text = 'No Title'
        # except KeyError:
        #     joints_np = inference_output['pred_data'].joints.cpu().numpy()[0] # only one batch
        #     # x_t_joints_np = inference_output['x_t'].joints.cpu().numpy()[0] # only one batch
        #     # output_of_x_t_joints_np = inference_output['output_of_x_t'].joints.cpu().numpy()[0] # only one batch

        #     text = sample_data_batch['text'][0]['caption']
        # og_joints_np = inference_output['gt_data'].joints.cpu().numpy()[0] # only one batch

        # output_video_file = os.path.join(
        #     output_dir, 'epoch%d_synthesis.mp4' % self.current_epoch
        #     )  # get subset of audio track

        # if sample_dataset.nfeats == 263:
        #     dataset_name = 'HumanML3D'
        # else:
        #     dataset_name = 'KIT-ML'

        # render_animation(joints_np, title = text, output = output_video_file, dataset_name=dataset_name)

        # # output_video_file = os.path.join(
        # #     output_dir, 'epoch%d_x_t.mp4' % self.current_epoch
        # #     )  # get subset of audio track

        # # render_animation(x_t_joints_np, title = text, output = output_video_file)
        
        # # output_video_file = os.path.join(
        # #     output_dir, 'epoch%d_output_of_x_t.mp4' % self.current_epoch
        # #     )  # get subset of audio track

        # # render_animation(output_of_x_t_joints_np, title = text, output = output_video_file)

        # og_output_video_file = os.path.join(
        #     output_dir, 'epoch%d_original.mp4' % self.current_epoch
        #     )  # get subset of audio track

        # render_animation(og_joints_np, title = text, output = og_output_video_file, dataset_name=dataset_name)

        # self.generator.train()


