# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/utils.py


from utils.biggan_utils import set_bn_train, set_deterministic_op_train, apply_accumulate_stat

import torch
import torch.nn.functional as F
from torch.nn import DataParallel


import numpy as np
import random
import os
from datetime import datetime



# fix python, numpy, torch seed
def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def count_parameters(module):
    return 'Number of parameters: {}'.format(sum([p.data.nelement() for p in module.parameters()]))


def define_sampler(dataset_name, conditional_strategy):
    if conditional_strategy != "no":
        if dataset_name == "cifar10":
            sampler = "class_order_all"
        else:
            sampler = "class_order_some"
    else:
        sampler = "default"
    return sampler

def check_flag_0(batch_size, n_gpus, fused_optimization, mixed_precision, acml_bn, ema, freeze_dis, checkpoint_folder):
    assert batch_size % n_gpus == 0, "batch_size should be divided by the number of gpus "
    assert int(fused_optimization)*int(mixed_precision) == 0.0, "can't turn on fused_optimization and mixed_precision together."
    if acml_bn is True:
        assert ema, "turning on accumulated batch_norm needs EMA update of the generator"
    if freeze_dis:
        assert checkpoint_folder is not None, "freezing discriminator needs a pre-trained model."


def check_flag_1(tempering_type, pos_collected_numerator, conditional_strategy, diff_aug, ada, mixed_precision,
                 gradient_penalty_for_dis, cr, bcr, zcr):
    assert int(diff_aug)*int(ada) == 0, \
        "you can't simultaneously apply differentiable Augmentation (DiffAug) and adaptive augmentation (ADA)"

    assert int(mixed_precision)*int(gradient_penalty_for_dis) == 0, \
        "you can't simultaneously apply mixed precision training (mpc) and gradient penalty for WGAN-GP"

    assert int(cr)*int(bcr) == 0 and int(cr)*int(zcr) == 0, \
        "you can't simultaneously turn on Consistency Reg. (CR) and Improved Consistency Reg. (ICR)"

    if conditional_strategy == "ContraGAN":
        assert tempering_type == "constant" or tempering_type == "continuous" or tempering_type == "discrete", \
            "tempering_type should be one of constant, continuous, or discrete"

    if pos_collected_numerator:
        assert conditional_strategy == "ContraGAN", "pos_collected_numerator option is not appliable except for ContraGAN."


def elapsed_time(start_time):
    now = datetime.now()
    elapsed = now - start_time
    return str(elapsed).split('.')[0]  # remove milliseconds


def reshape_weight_to_matrix(weight):
    weight_mat = weight
    dim =0
    if dim != 0:
        # permute dim to front
        weight_mat = weight_mat.permute(dim, *[d for d in range(weight_mat.dim()) if d != dim])
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


def find_string(list_, string):
    for i, s in enumerate(list_):
        if string == s:
            return i

def find_and_remove(path):
    if os.path.isfile(path):
        os.remove(path)

def calculate_all_sn(model):
    sigmas = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "bn" not in name and "shared" not in name and "deconv" not in name:
                if "blocks" in name:
                    splited_name = name.split('.')
                    idx = find_string(splited_name, 'blocks')
                    block_idx = int(splited_name[int(idx+1)])
                    module_idx = int(splited_name[int(idx+2)])
                    operation_name = splited_name[idx+3]
                    if isinstance(model, DataParallel):
                        operations = model.module.blocks[block_idx][module_idx]
                    else:
                        operations = model.blocks[block_idx][module_idx]
                    operation = getattr(operations, operation_name)
                else:
                    splited_name = name.split('.')
                    idx = find_string(splited_name, 'module') if isinstance(model, DataParallel) else -1
                    operation_name = splited_name[idx+1]
                    if isinstance(model, DataParallel):
                        operation = getattr(model.module, operation_name)
                    else:
                        operation = getattr(model, operation_name)

                weight_orig = reshape_weight_to_matrix(operation.weight_orig)
                weight_u = operation.weight_u
                weight_v = operation.weight_v
                sigmas[name] = torch.dot(weight_u, torch.mv(weight_orig, weight_v))
    return sigmas


def change_generator_mode(gen, gen_copy, acml_bn, acml_stat_step, prior, batch_size, z_dim, num_classes, device, training):
    if training:
        gen.train()
        if gen_copy is not None:
            gen_copy.train()
            return gen_copy
        return gen
    else:
        gen.eval()
        gen.apply(set_deterministic_op_train)
        if gen_copy is not None:
            gen_copy.eval()
            gen_copy.apply(set_deterministic_op_train)
            if acml_bn:
                apply_accumulate_stat(gen_copy, acml_stat_step, prior, batch_size, z_dim, num_classes, device)
                return gen_copy
            else:
                gen_copy.apply(set_bn_train)
                return gen_copy
        else:
            return gen
