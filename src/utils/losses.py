# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/loss.py

from torch.nn import DataParallel
from torch import autograd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.ops as ops


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, cls_output, label, **_):
        return self.ce_loss(cls_output, label).mean()


class CrossEntropyLossMI(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLossMI, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, mi_cls_output, label, **_):
        return self.ce_loss(mi_cls_output, label).mean()


class ConditionalContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, global_rank):
        super(ConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.global_rank = global_rank
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.global_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.global_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, label, **_):
        sim_matrix = self.calculate_similarity_matrix(embed, embed)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()


class ConditionalContrastiveLossMI(torch.nn.Module):
    def __init__(self, num_classes, temperature, global_rank):
        super(ConditionalContrastiveLossMI, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.global_rank = global_rank
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.global_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.global_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, mi_embed, mi_proxy, label, **_):
        sim_matrix = self.calculate_similarity_matrix(mi_embed, mi_embed)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(mi_embed, mi_proxy) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()


def d_vanilla(d_logit_real, d_logit_fake):
    device = d_logit_real.get_device()
    ones = torch.ones_like(d_logit_real, device=device, requires_grad=False)
    d_loss = -torch.mean(nn.LogSigmoid()(d_logit_real) + nn.LogSigmoid()(ones - d_logit_fake))
    return d_loss


def g_vanilla(g_logit_fake):
    return -torch.mean(nn.LogSigmoid()(g_logit_fake))


def d_ls(d_logit_real, d_logit_fake):
    d_loss = 0.5 * (d_logit_real - torch.ones_like(d_logit_real))**2 + 0.5 * (d_logit_fake)**2
    return d_loss.mean()


def g_ls(d_logit_fake):
    gen_loss = 0.5 * (d_logit_fake - torch.ones_like(d_logit_fake))**2
    return gen_loss.mean()


def d_hinge(d_logit_real, d_logit_fake):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))


def g_hinge(g_logit_fake):
    return -torch.mean(g_logit_fake)


def d_wasserstein(d_logit_real, d_logit_fake):
    return torch.mean(d_logit_fake - d_logit_real)


def g_wasserstein(g_logit_fake):
    return -torch.mean(g_logit_fake)


def crammer_singer_loss(adv_output, label, **_):
    # https://github.com/ilyakava/BigGAN-PyTorch/blob/master/train_fns.py
    # crammer singer criterion
    num_real_classes = adv_output.shape[1] - 1
    mask = torch.ones_like(adv_output).to(adv_output.device)
    mask.scatter_(1, label.unsqueeze(-1), 0)
    wrongs = torch.masked_select(adv_output, mask.bool()).reshape(adv_output.shape[0], num_real_classes)
    max_wrong, _ = wrongs.max(1)
    max_wrong = max_wrong.unsqueeze(-1)
    target = adv_output.gather(1, label.unsqueeze(-1))
    return torch.mean(F.relu(1 + max_wrong - target))


def feature_matching_loss(real_embed, fake_embed):
    # https://github.com/ilyakava/BigGAN-PyTorch/blob/master/train_fns.py
    # feature matching criterion
    fm_loss = torch.mean(torch.abs(torch.mean(fake_embed, 0) - torch.mean(real_embed, 0)))
    return fm_loss


def cal_deriv(inputs, outputs, device):
    grads = autograd.grad(outputs=outputs,
                          inputs=inputs,
                          grad_outputs=torch.ones(outputs.size()).to(device),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    return grads


def latent_optimise(zs, fake_labels, generator, discriminator, batch_size, lo_rate, lo_steps, lo_alpha, lo_beta,
                    cal_trsf_cost, device):
    for step in range(lo_steps):
        drop_mask = (torch.FloatTensor(batch_size, 1).uniform_() > 1 - lo_rate).to(device)

        zs = autograd.Variable(zs, requires_grad=True)
        fake_images = generator(zs, fake_labels)
        output_dict = discriminator(fake_images, fake_labels, eval=False)
        z_grads = cal_deriv(inputs=zs, outputs=output_dict["adv_output"], device=device)
        z_grads_norm = torch.unsqueeze((z_grads.norm(2, dim=1)**2), dim=1)
        delta_z = lo_alpha * z_grads / (lo_beta + z_grads_norm)
        zs = torch.clamp(zs + drop_mask * delta_z, -1.0, 1.0)

        if cal_trsf_cost:
            if step == 0:
                trsf_cost = (delta_z.norm(2, dim=1)**2).mean()
            else:
                trsf_cost += (delta_z.norm(2, dim=1)**2).mean()
        else:
            trsf_cost = None
        return zs, trsf_cost


def cal_grad_penalty(real_images, real_labels, fake_images, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_images.nelement() // batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    output_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=output_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    grads_penalty = ((grads.norm(2, dim=1) - 1)**2).mean()
    return grads_penalty


def cal_dra_penalty(real_images, real_labels, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    differences = 0.5 * real_images.std() * torch.rand(real_images.size()).to(device)
    interpolates = real_images + (alpha * differences)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    output_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=output_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    grads_penalty = ((grads.norm(2, dim=1) - 1)**2).mean()
    return grads_penalty


def cal_r1_reg(adv_output, images, device):
    batch_size = images.size(0)
    grad_dout = cal_deriv(inputs=images, outputs=adv_output.sum(), device=device)
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == images.size())
    r1_reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return r1_reg

def adjust_k(current_k, topk_gamma, sup_k):
    current_k = max(int(current_k*topk_gamma), sup_k)
    return current_k
