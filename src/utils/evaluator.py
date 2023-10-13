import torch
import torch.nn as nn
from src.models.metrics.metrics import *
from hydra.utils import instantiate
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Evaluator:
    def __init__(self, nfeats, device, motionencoder, autoencoder, checkpoint_paths, diversity_times=100):
        # self.generator = generator
        self.device = device
        self.motionencoder = instantiate(motionencoder, _recursive_=False)
        self.autoencoder = instantiate(autoencoder, pose_dim=nfeats, _recursive_=False)
        self.motionencoder.to(self.device)
        self.autoencoder.to(self.device)
        self.diversity_times = diversity_times

        self.all_size = 0
        self.matching_score_sum = 0
        self.top_k_count = 0

        self.all_motion_embeds_generated = []
        self.all_motion_embeds_gt = []
        self.curr_motion_embs_generated = None
        self.curr_motion_embs_gt = None
        self.curr_text_embs_gt = None

        self.mm_num_samples = 100
        self.mm_num_repeat = 30
        self.mm_num_times = 10

        # self.batch_size = None
        # self.dataset = None
        # self.mm_generated_motions = []
        # self.mm_m_lens = []

        # FIXME Combine code for loading our model and Guo model

        # state_dict = torch.load(checkpoint_paths)['state_dict']
        # auto_state_dict = {}
        # for param_key in list(state_dict.keys()):
        #     if 'generator' in param_key:
        #         new_param_key = param_key[10:]
        #         state_dict[new_param_key] = state_dict.pop(param_key)
        #     elif 'autoencoder' in param_key:
        #         new_param_key = param_key[12:]
        #         auto_state_dict[new_param_key] = state_dict.pop(param_key)
        #     elif 'length_estimator' in param_key:
        #         state_dict.pop(param_key)
        # self.motionencoder.load_state_dict(state_dict)
        # self.autoencoder.load_state_dict(auto_state_dict)

        checkpoint = torch.load(checkpoint_paths,
                                map_location=self.device)
        self.autoencoder.motionencoder.load_state_dict(checkpoint['movement_encoder'])
        self.motionencoder.textencoder.load_state_dict(checkpoint['text_encoder'])
        self.motionencoder.motion_encoder.load_state_dict(checkpoint['motion_encoder'])

    # def post_init(self, batch_size, val_dataset):
    #     # self.mm_batch_idxs = np.random.choice(len(val_dataset) // batch_size, self.mm_num_samples, replace=False)
    #     # self.mm_batch_idxs = np.sort(self.mm_batch_idxs)
    #     self.batch_size = batch_size
    #     self.mm_idxs = np.random.randint(len(val_dataset), size=self.mm_num_samples)
    #     self.mm_idxs = np.sort(self.mm_idxs)

    def reset(self):
        self.all_motion_embeds_gt = []
        self.all_motion_embeds_generated = []
        self.curr_motion_embs_generated = None
        self.curr_motion_embs_gt = None
        self.curr_text_embs_gt = None
        self.all_size = 0
        self.matching_score_sum = 0
        self.top_k_count = 0

        # self.mm_generated_motions = []
        # self.mm_m_lens = []

    def push_vals(self, batch, batch_idx, outputs):

        # Initialize variables
        # if batch_idx == 0:
        #     self.post_init(batch_size, val_dataset)

        self.push_generated_outputs(outputs.to(self.device), batch["orig_length"])
        self.push_gt(batch['datastruct'].features.to(self.device), batch['orig_length'])
        self.push_text(batch)
        self.update_r_precision()
        # self.update_multimodality(batch, batch_idx, generator)

    def push_generated_outputs(self, outputs, m_lens):
        self.curr_motion_embs_generated = self.get_motion_embeddings(outputs, m_lens)
        self.all_motion_embeds_generated.append(self.curr_motion_embs_generated.cpu())

    def push_gt(self, features, m_lens):
        self.curr_motion_embs_gt = self.get_motion_embeddings(features, m_lens)
        self.all_motion_embeds_gt.append(self.curr_motion_embs_gt.cpu())

    def push_text(self, batch):
        self.curr_text_embs_gt = self.motionencoder.get_text_embeddings(batch["word_embs"],
                                                                        batch["pos_onehot"],
                                                                        batch["cap_lens"],
                                                                        batch["orig_length"])

    def get_motion_embeddings(self, inpt, m_lens):
        return self.motionencoder.get_motion_embeddings(self.autoencoder, inpt.float(), m_lens)

    def evaluate_metrics(self, val_dataset, generator):
        generated_activations = np.concatenate(self.all_motion_embeds_generated, axis=0)
        gt_activations = np.concatenate(self.all_motion_embeds_gt, axis=0)
        generated_activations = np.reshape(generated_activations, (generated_activations.shape[0], -1))
        gt_activations = np.reshape(gt_activations, (gt_activations.shape[0], -1))
        fid = self.evaluate_fid(generated_activations, gt_activations)
        diversity = self.evaluate_diversity(generated_activations)
        matching_score, r_precision = self.evaluate_r_precision()
        # multimodality = self.evaluate_multimodality(val_dataset, generator)
        metrics = {
            'fid': fid,
            'diversity': diversity,
            'matching_score': matching_score,
            'R-precision-Top-1': r_precision[0],
            'R-precision-Top-2': r_precision[1],
            'R-precision-Top-3': r_precision[2],
            # 'multimodality': multimodality,
        }
        return metrics

    def evaluate_fid(self, generated_activations, gt_activations):
        gen_mu, gen_cov = calculate_activation_statistics(generated_activations)
        gt_mu, gt_cov = calculate_activation_statistics(gt_activations)
        fid = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
        return fid

    def evaluate_diversity(self, activations):
        if activations.shape[0] > self.diversity_times:
            diversity = calculate_diversity(activations, self.diversity_times)
        else:
            diversity = 0
        return diversity

    def update_r_precision(self):
        # Calculating Matching and R-precision in every step. Totaled at epoch_end
        t1 = self.curr_text_embs_gt.cpu().numpy()
        t2 = self.curr_motion_embs_gt.cpu().numpy()
        dist_mat = euclidean_distance_matrix(self.curr_text_embs_gt.cpu().numpy(),
                                             self.curr_motion_embs_generated.cpu().numpy())
        self.matching_score_sum += dist_mat.trace()

        argsmax = np.argsort(dist_mat, axis=1)
        top_k_mat = calculate_top_k(argsmax, top_k=3)
        self.top_k_count += top_k_mat.sum(axis=0)

        self.all_size += self.curr_text_embs_gt.shape[0]

    def evaluate_r_precision(self):
        matching_score = self.matching_score_sum / self.all_size
        r_precision = self.top_k_count / self.all_size
        return matching_score, r_precision

    # def update_multimodality(self, batch, batch_idx, generator):
    #     ele_list = []
    #     for i in range(self.batch_size):
    #         if batch_idx*self.batch_size + i in self.mm_idxs:
    #             ele_list.append(batch_idx*self.batch_size + i)
    #
    #         # if self.mm_idxs[self.mm_num_now] == batch_idx:
    #             # sample = batch[i].unsqueeze(0)
    #             # for key, val in batch[i].items():
    #             #     sample[key] =
    #     for t in range(self.mm_num_repeat):
    #         outputs = generator(batch, do_inference=True)
    #         for ele in ele_list:
    #             self.mm_generated_motions.append(outputs['pred_data'].features[ele])
    #             self.mm_m_lens.append(batch["length"][ele])
    #         # self.mm_generated_motions.append({
    #         #     'motion': outputs['pred_data'][0].cpu().numpy(),
    #         #     'length': sample[0]["length"]
    #         # })
    #     self.mm_num_now = len(self.mm_generated_motions)

    def evaluate_multimodality(self, datamodule, generator):
        val_dataset = datamodule.val_dataloader().dataset
        all_mm_generated_motions = []
        mm_idxs = np.random.choice(len(val_dataset), self.mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        mm_num_now = 0
        dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=True, collate_fn=datamodule.dataloader_options['collate_fn'])
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            if batch_idx == mm_idxs[mm_num_now]:
                batch_mm_motions = []
                for t in range(self.mm_num_repeat):
                    outputs = generator(batch, do_inference=True)
                    batch_mm_motions.append({
                        'motion': outputs['pred_data'].features[0].cpu().numpy(),
                        'length': batch["length"][0],
                    })
                all_mm_generated_motions.append(batch_mm_motions)
                mm_num_now = min(self.mm_num_samples-1, mm_num_now+1)

        all_mm_motion_embeddings = []
        for batch_mm_motion in all_mm_generated_motions:
            motions = []
            m_lens = []
            for mm_motion in batch_mm_motion:
                m_lens.append(mm_motion['length'])
                motion = mm_motion['motion']
                motion = motion[None, :]
                motions.append(motion)
            m_lens = np.array(m_lens, dtype=np.int)
            motions = np.concatenate(motions, axis=0)
            sort_indx = np.argsort(m_lens)[::-1].copy()
            m_lens = m_lens[sort_indx]
            motions = torch.tensor(motions[sort_indx]).to(self.device)
            all_mm_motion_embeddings.append(self.get_motion_embeddings(motions, m_lens).unsqueeze(0))

        mm_motion_embeddings = torch.cat(all_mm_motion_embeddings, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(mm_motion_embeddings, self.mm_num_times)

        # for idx in range(len(self.mm_generated_motions)):
        #     mm_motion_embeddings.append(self.get_motion_embeddings(self.mm_generated_motions[idx], self.mm_m_lens[idx]))
        # mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
        # multimodality = calculate_multimodality(mm_motion_embeddings, self.mm_num_times)
        return multimodality
