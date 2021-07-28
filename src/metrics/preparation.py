# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/preparation.py

import os

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
import numpy as np

from metrics.inception_net import InceptionV3
import metrics.fid as fid
import metrics.ins as ins
import utils.misc as misc


class LoadEvalModel(object):
    def __init__(self, eval_backbone, world_size, distributed_data_parallel, local_rank):
        super(LoadEvalModel, self).__init__()
        self.eval_backbone = eval_backbone
        self.save_output = misc.SaveOutput()

        if self.eval_backbone == "Inception_V3":
            self.model = InceptionV3().to(local_rank)
        elif self.eval_backbone == "SwAV":
            self.model = torch.hub.load('facebookresearch/swav', 'resnet50').to(local_rank)
            hook_handles = []
            for name, layer in self.model.named_children():
                if name == "fc":
                    handle = layer.register_forward_pre_hook(self.save_output)
                    hook_handles.append(handle)
        else:
            raise NotImplementedError

        if world_size > 1 and distributed_data_parallel:
            misc.toggle_grad(self.model, on=True)
            self.model = DDP(self.model,
                             device_ids=[local_rank],
                             broadcast_buffers=False,
                             find_unused_parameters=True)
        elif world_size > 1 and distributed_data_parallel is False:
            self.model = DataParallel(self.model,
                                      output_device=local_rank)
        else:
            pass

    def eval(self):
        self.model.eval()

    def get_outputs(self, x):
        if self.eval_backbone == "Inception_V3":
            repres, logits = self.model(x)
        else:
            logits = self.model(x)
            repres = self.save_output.outputs[0][0]
            self.save_output.clear()
        return repres, logits

def prepare_moments_calculate_ins(data_loader, eval_model, splits, cfgs, logger, local_rank):
    eval_model.eval()
    save_path = os.path.abspath(os.path.join("./data", cfgs.DATA.name + "_" + cfgs.RUN.ref_dataset + "_" + \
                                             "inception_moments.npz"))
    is_file = os.path.isfile(save_path)

    if is_file:
        mu = np.load(save_path)["mu"]
        sigma = np.load(save_path)["sigma"]
    else:
        if local_rank == 0: logger.info("Calculate moments of {ref} dataset.".format(ref=cfgs.RUN.ref_dataset))
        mu, sigma = fid.calculate_moments(data_loader=data_loader,
                                          Gen="N/A",
                                          eval_model=eval_model,
                                          is_generate=False,
                                          num_generate="N/A",
                                          y_sampler="N/A",
                                          batch_size=cfgs.OPTIMIZER.batch_size,
                                          z_prior="N/A",
                                          truncation_th="N/A",
                                          z_dim="N/A",
                                          num_classes=cfgs.DATA.num_classes,
                                          LOSS="N/A",
                                          local_rank=local_rank,
                                          disable_tqdm=False)

        if local_rank == 0: logger.info("Save calculated means and covariances to disk.")
        np.savez(save_path, **{"mu": mu, "sigma": sigma})

    if is_file:
        pass
    else:
        if local_rank == 0: logger.info("Calculate inception score of the {ref} dataset.".format(ref=cfgs.RUN.ref_dataset))
        is_score, is_std = ins.eval_dataset(data_loader=data_loader,
                                            eval_model=eval_model,
                                            splits=splits,
                                            batch_size=cfgs.OPTIMIZER.batch_size,
                                            local_rank=local_rank,
                                            disable_tqdm=False)
        if local_rank == 0: logger.info("Inception score={is_score}-Inception_std={is_std}".format(is_score=is_score, is_std=is_std))
    return mu, sigma
