# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/preparation.py

from os.path import exists, join
import os

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from metrics.inception_net import InceptionV3
import metrics.fid as fid
import metrics.ins as ins
import utils.misc as misc
import utils.ops as ops
import utils.resize as resize


class LoadEvalModel(object):
    def __init__(self, eval_backbone, resize_fn, world_size, distributed_data_parallel, device):
        super(LoadEvalModel, self).__init__()
        self.eval_backbone = eval_backbone
        self.resize_fn = resize_fn
        self.save_output = misc.SaveOutput()

        if self.eval_backbone == "Inception_V3":
            self.res, mean, std = 299, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            self.model = InceptionV3(resize_input=False, normalize_input=False).to(device)
        elif self.eval_backbone == "SwAV":
            self.res, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.model = torch.hub.load("facebookresearch/swav", "resnet50").to(device)
            hook_handles = []
            for name, layer in self.model.named_children():
                if name == "fc":
                    handle = layer.register_forward_pre_hook(self.save_output)
                    hook_handles.append(handle)
        else:
            raise NotImplementedError

        self.resizer = resize.build_resizer(mode=resize_fn, size=self.res)
        self.trsf = transforms.Compose([transforms.ToTensor()])
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to("cuda")
        self.std = torch.Tensor(std).view(1, 3, 1, 1).to("cuda")

        if world_size > 1 and distributed_data_parallel:
            misc.make_model_require_grad(self.model)
            self.model = DDP(self.model, device_ids=[device], broadcast_buffers=True)
        elif world_size > 1 and distributed_data_parallel is False:
            self.model = DataParallel(self.model, output_device=device)
        else:
            pass

    def eval(self):
        self.model.eval()

    def get_outputs(self, x, quantize=False):
        if quantize:
            x = ops.quantize_images(x)
        else:
            x = x.detach().cpu().numpy().astype(np.uint8)
        x = ops.resize_images(x, self.resizer, self.trsf, self.mean, self.std)

        if self.eval_backbone == "Inception_V3":
            repres, logits = self.model(x)
        elif self.eval_backbone == "SwAV":
            logits = self.model(x)
            repres = self.save_output.outputs[0][0]
            self.save_output.clear()
        return repres, logits


def prepare_moments(data_loader, eval_model, quantize, cfgs, logger, device):
    disable_tqdm = device != 0
    eval_model.eval()
    moment_dir = join(cfgs.RUN.save_dir, "moments")
    if not exists(moment_dir):
        os.makedirs(moment_dir)
    moment_path = join(moment_dir, cfgs.DATA.name + "_" + cfgs.RUN.ref_dataset + "_" + \
                       cfgs.RUN.resize_fn + "_" + str(cfgs.DATA.img_size) + "_" + cfgs.RUN.eval_backbone + "_moments.npz")
    is_file = os.path.isfile(moment_path)
    if is_file:
        mu = np.load(moment_path)["mu"]
        sigma = np.load(moment_path)["sigma"]
    else:
        if device == 0:
            logger.info("Calculate moments of {ref} dataset using {eval_backbone} model.".\
                        format(ref=cfgs.RUN.ref_dataset, eval_backbone=cfgs.RUN.eval_backbone))
        mu, sigma = fid.calculate_moments(data_loader=data_loader,
                                          eval_model=eval_model,
                                          num_generate="N/A",
                                          batch_size=cfgs.OPTIMIZATION.batch_size,
                                          quantize=quantize,
                                          world_size=cfgs.OPTIMIZATION.world_size,
                                          DDP=cfgs.RUN.distributed_data_parallel,
                                          disable_tqdm=disable_tqdm,
                                          fake_feats=None)

        if device == 0:
            logger.info("Save calculated means and covariances to disk.")
        np.savez(moment_path, **{"mu": mu, "sigma": sigma})
    return mu, sigma


def calculate_ins(data_loader, eval_model, quantize, splits, cfgs, logger, device):
    disable_tqdm = device != 0
    if device == 0:
        logger.info("Calculate inception score of the {ref} dataset uisng pre-trained {eval_backbone} model.".\
                    format(ref=cfgs.RUN.ref_dataset, eval_backbone=cfgs.RUN.eval_backbone))
    is_score, is_std = ins.eval_dataset(data_loader=data_loader,
                                        eval_model=eval_model,
                                        quantize=quantize,
                                        splits=splits,
                                        batch_size=cfgs.OPTIMIZATION.batch_size,
                                        world_size=cfgs.OPTIMIZATION.world_size,
                                        DDP=cfgs.RUN.distributed_data_parallel,
                                        disable_tqdm=disable_tqdm)
    if device == 0:
        logger.info("Inception score={is_score}-Inception_std={is_std}".format(is_score=is_score, is_std=is_std))
