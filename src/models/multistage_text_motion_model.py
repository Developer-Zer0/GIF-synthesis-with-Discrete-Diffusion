import os
from typing import Any, List
import hydra
from torchmetrics import MetricCollection

from src.models.base import BaseModel
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig

import random
from src.utils.basic_video_renderer import render_animation

# from src.utils.evaluator import Evaluator

import numpy as np

if os.name != "nt":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch

class MultistageTextMotionModel(BaseModel):
    """
    Text Motion model for 3D motion synthesis from tset
    Model can have multiple stages
    Motion to motion synthesis
    Text to motion synthesis
    Text length predictor
    """

    def __init__(
        self,
        autoencoder: DictConfig,
        generator: DictConfig,
        freeze_models_dict: dict,              # Dict of model params to be frozen
        checkpoint_paths: dict,                # Dict of checkpoint paths for either autoencoder or length_estimator or generator
        # nfeats: int,
        # losses: dict,                         # Use this for the onitialize_losses function
        evaluator: DictConfig = None,
        generator_losses: DictConfig = None,
        autoencoder_losses: DictConfig = None,
        length_estimator_losses: DictConfig = None,
        lr_args = {},
        render_animations: bool = True,
        do_evaluation: bool = False,
        devices: str = 'cpu',
        **kwargs
    ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(ignore=['generator'])

        self.gpu_device = ''

        if devices == 'cpu':
            self.gpu_device = devices
        else:
            self.gpu_device = 'cuda:' + str(devices[0])

        self.generator = instantiate(generator, device=self.gpu_device, _recursive_=False)
        self.autoencoder = instantiate(autoencoder, device=self.gpu_device, _recursive_=False)
        self.length_estimator = None
        self.modules = [self.generator, self.autoencoder, self.length_estimator]
        self.keys = ['generator', 'autoencoder', 'length_estimator']

        # self.load_checkpoints(checkpoint_paths)
        
        state_dict = torch.load(checkpoint_paths, map_location=self.gpu_device)['state_dict']
        self.autoencoder.load_state_dict(state_dict)

        ## This function not working for some reason
        # self.losses = self.initialize_losses(losses)

        if generator_losses:
            self._generator_losses = MetricCollection(
                {split: instantiate(generator_losses,_recursive_ = False)
                for split in ["generator_losses_train", "generator_losses_test", "generator_losses_val"]})
            self.generator_losses = {key: self._generator_losses["generator_losses_" + key] for key in ["train", "test", "val"]}
        else:
            self.generator_losses = None

        if autoencoder_losses:
            self._autoencoder_losses = MetricCollection(
                {split: instantiate(autoencoder_losses,_recursive_ = False)
                for split in ["autoencoder_losses_train", "autoencoder_losses_test", "autoencoder_losses_val"]})
            self.autoencoder_losses = {key: self._autoencoder_losses["autoencoder_losses_" + key] for key in ["train", "test", "val"]}
        else:
            self.autoencoder_losses = None

        if length_estimator_losses:
            self._length_estimator_losses = MetricCollection(
                {split: instantiate(length_estimator_losses,_recursive_ = False)
                for split in ["length_estimator_losses_train", "length_estimator_losses_test", "length_estimator_losses_val"]})
            self.length_estimator_losses = {key: self._length_estimator_losses["length_estimator_losses_" + key] for key in ["train", "test", "val"]}
        else:
            self.length_estimator_losses = None

        self.losses = (self.generator_losses, self.autoencoder_losses, self.length_estimator_losses)

        self.lr_args = lr_args
        self.render_animations = render_animations

        # self.freeze_params(freeze_models_dict)
        self.automatic_optimization = False
        self.loss_dict = {}

        self.output_test_motions = False

        self.do_evaluation = do_evaluation
        if self.do_evaluation:
            self.evaluator = instantiate(evaluator, device=self.gpu_device, _recursive_=False)

    def load_checkpoints(self, checkpoint_paths):
        for key, module in zip(self.keys, self.modules):
            try:
                state_dict = torch.load(checkpoint_paths[key])['state_dict']
                for param_key in list(state_dict.keys()):
                    new_param_key = param_key[10:]
                    state_dict[new_param_key] = state_dict.pop(param_key)
                module.load_state_dict(state_dict)
            except KeyError:
                pass

    def freeze_params(self, freeze_models_dict):
        # Freeze selected Generator weights
        for key, module in zip(self.keys, self.modules):
            for layer_name, params in module.named_parameters():
                for layer in freeze_models_dict[key]:
                    if layer in layer_name:
                        params.requires_grad = False


    # def initialize_losses(self, losses_dict):
    #     all_losses = []
    #     new_keys = [k + '_losses' for k in self.keys]
    #     for new_key in new_keys:
    #         try:
    #             self._losses = MetricCollection(
    #                 {split: instantiate(losses_dict[new_key],_recursive_ = False)
    #                 for split in ["losses_train", "losses_test", "losses_val"]})
    #             losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
    #             all_losses.append(losses)
    #         except KeyError:
    #             all_losses.append(None)
    #     return tuple(all_losses)


    def generator_step(self, batch: Any):
        generator_outputs = self.generator(batch, self.autoencoder, self.length_estimator)
        #pred_poses = generator_outputs['pred_poses']
        #pred_axis_angle = self.output_poses_to_axis_angles(pred_poses)
        outputs = {}
        outputs.update(generator_outputs)
        # add length entry into output
        outputs['length'] = batch['length']
        return outputs

    # Added inference step for generator
    def sample_generator_step(self, batch: Any):
        generator_outputs = self.generator(batch, self.autoencoder, self.length_estimator, do_inference=True)
        # pred_poses = generator_outputs['pred_poses']
        # pred_axis_angle = self.output_poses_to_axis_angles(pred_poses)
        outputs = {}
        outputs.update(generator_outputs)
        # add length entry into output
        outputs['length'] = batch['length']
        return outputs

    def allsplit_step(self, split: str, batch, batch_idx):

        # opt_gen, opt_auto, opt_len_est = self.optimizers()
        # optimizers are (opt_gen, opt_auto, opt_len_est) in that order
        optimizers = self.optimizers()

        prefix = '%s/'%split
        outputs = self.generator_step(batch)

        if self.do_evaluation and (split != 'train' and len(batch['length']) != 1) and self.current_epoch % 5 == 0:
            eval_outputs = self.sample_generator_step(batch)
            self.evaluator.push_vals(batch, batch_idx, eval_outputs['pred_data'])
            # self.evaluator.push_vals(batch, batch_idx, batch['datastruct'].features)

        new_keys = [k + '_loss' for k in self.keys]

        for opt, loss, new_key in zip(optimizers, self.losses, new_keys):
            if loss:
                compute_loss = loss[split]
                self.loss_dict[new_key] = compute_loss.update(outputs)
                if split == "train":
                    opt.zero_grad()
                    torch.autograd.set_detect_anomaly(True)
                    self.manual_backward(self.loss_dict[new_key], retain_graph=False)

        # check if this is how it is supposed to be done
        for opt in optimizers:
            opt.step()

        # if split == "val":  # compute metrics for validation data
        #     self.metrics.update(outputs)

        if self.output_test_motions and split == "test" and batch_idx%100 == 0:
            eval_outputs = self.sample_generator_step(batch)
            self.render_test_results(eval_outputs, batch_idx, batch)

        return self.loss_dict

    def allsplit_epoch_end(self, split: str, outputs):
        s = 'total/' + split
        total_dico = {s: 0.0}
        new_keys = [k + '_dico' for k in self.keys]

        for loss, new_key in zip(self.losses, new_keys):
            if loss:
                losses = loss[split]
                loss_dict = losses.compute()
                dico = {losses.loss2logname(loss, split): value.item()
                        for loss, value in loss_dict.items()}

                # if split == "val":
                #     metrics_dict = self.metrics.compute()
                #     dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items()})
                dico.update({"epoch": float(self.trainer.current_epoch),
                            "step": float(self.trainer.current_epoch)})

                total_dico.update(dico)
                total_dico[s] += dico[s]

        if self.do_evaluation and (split != 'train') and self.current_epoch % 5 == 0:
            metrics = self.evaluator.evaluate_metrics(self.trainer.datamodule, self.generator)
            # print(metrics)
            total_dico.update({f"Metrics/{metric}-{split}": value for metric, value in metrics.items()})
            self.evaluator.reset()

        if split == 'val' and self.current_epoch % 10 == 0:
            self.render_sample_results()

        self.log_dict(total_dico)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        b1 = 0.5
        b2 = 0.999
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr_args['gen_lr'], betas=(b1, b2))
        opt_auto = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr_args['auto_lr'], betas=(b1, b2))
        # opt_len_est = torch.optim.Adam(self.length_estimator.parameters(), lr=self.lr_args['len_est_lr'], betas=(b1, b2))
        return opt_gen, opt_auto

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
        inference_output = self.sample_generator_step(sample_data_batch)

        output_video_file = os.path.join(output_dir, 'epoch%d_synthesis_%s.mp4' % (self.current_epoch, sample_data_batch['text'][0]))
        render_animation(inference_output['pred_data'][0].permute(1, 2, 3, 0).cpu().numpy(), output_path = output_video_file, fps=1)

        output_video_file = os.path.join(output_dir, 'epoch%d_single_step.mp4' % self.current_epoch)
        render_animation(inference_output['pred_single_step'][0].permute(1, 2, 3, 0).cpu().numpy(), output_path = output_video_file, fps=1)

        output_video_file = os.path.join(output_dir, 'epoch%d_original.mp4' % self.current_epoch)
        render_animation(inference_output['gt_data'][0].permute(1, 2, 3, 0).cpu().numpy(), output_path = output_video_file, fps=1)

        self.generator.train()
