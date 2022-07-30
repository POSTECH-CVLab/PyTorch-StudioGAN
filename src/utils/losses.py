# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/loss.py

from torch.nn import DataParallel
from torch import autograd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from utils.style_ops import conv2d_gradfix
import utils.ops as ops


class GatherLayer(torch.autograd.Function):
    """
    This file is copied from
    https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
    Gather tensors from all process, supporting backward propagation
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, cls_output, label, **_):
        return self.ce_loss(cls_output, label).mean()


class ConditionalContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, master_rank, DDP):
        super(ConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, label, **_):
        if self.DDP:
            embed = torch.cat(GatherLayer.apply(embed), dim=0)
            proxy = torch.cat(GatherLayer.apply(proxy), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)

        sim_matrix = self.calculate_similarity_matrix(embed, embed)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()


class Data2DataCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, m_p, master_rank, DDP):
        super(Data2DataCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.m_p = m_p
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def make_index_matrix(self, labels):
        labels = labels.detach().cpu().numpy()
        num_samples = labels.shape[0]
        mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0

        for c in range(self.num_classes):
            c_indices = np.where(labels==c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def forward(self, embed, proxy, label, **_):
        # If train a GAN throuh DDP, gather all data on the master rank
        if self.DDP:
            embed = torch.cat(GatherLayer.apply(embed), dim=0)
            proxy = torch.cat(GatherLayer.apply(proxy), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)

        # calculate similarities between sample embeddings
        sim_matrix = self.calculate_similarity_matrix(embed, embed) + self.m_p - 1
        # remove diagonal terms
        sim_matrix = self.remove_diag(sim_matrix/self.temperature)
        # for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = F.relu(sim_matrix) - sim_max.detach()

        # calculate similarities between sample embeddings and the corresponding proxies
        smp2proxy = self.cosine_similarity(embed, proxy)
        # make false negative removal
        removal_fn = self.remove_diag(self.make_index_matrix(label)[label])
        # apply the negative removal to the similarity matrix
        improved_sim_matrix = removal_fn*torch.exp(sim_matrix)

        # compute positive attraction term
        pos_attr = F.relu((self.m_p - smp2proxy)/self.temperature)
        # compute negative repulsion term
        neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum(dim=1))
        # compute data to data cross-entropy criterion
        criterion = pos_attr + neg_repul
        return criterion.mean()


class PathLengthRegularizer:
    def __init__(self, device, pl_decay=0.01, pl_weight=2, pl_no_weight_grad=False):
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.pl_no_weight_grad = pl_no_weight_grad

    def cal_pl_reg(self, fake_images, ws):
        #ws refers to weight style
        #receives new fake_images of original batch (in original implementation, fakes_images used for calculating g_loss and pl_loss is generated independently)
        pl_noise = torch.randn_like(fake_images) / np.sqrt(fake_images.shape[2] * fake_images.shape[3])
        with conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
            pl_grads = torch.autograd.grad(outputs=[(fake_images * pl_noise).sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]
        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        loss_Gpl = (pl_penalty * self.pl_weight).mean(0)
        return loss_Gpl


def enable_allreduce(dict_):
    loss = 0
    for key, value in dict_.items():
        if value is not None and key != "label":
            loss += value.mean()*0
    return loss


def d_vanilla(d_logit_real, d_logit_fake, DDP):
    d_loss = torch.mean(F.softplus(-d_logit_real)) + torch.mean(F.softplus(d_logit_fake))
    return d_loss


def g_vanilla(d_logit_fake, DDP):
    return torch.mean(F.softplus(-d_logit_fake))


def d_logistic(d_logit_real, d_logit_fake, DDP):
    d_loss = F.softplus(-d_logit_real) + F.softplus(d_logit_fake)
    return d_loss.mean()


def g_logistic(d_logit_fake, DDP):
    # basically same as g_vanilla.
    return F.softplus(-d_logit_fake).mean()


def d_ls(d_logit_real, d_logit_fake, DDP):
    d_loss = 0.5 * (d_logit_real - torch.ones_like(d_logit_real))**2 + 0.5 * (d_logit_fake)**2
    return d_loss.mean()


def g_ls(d_logit_fake, DDP):
    gen_loss = 0.5 * (d_logit_fake - torch.ones_like(d_logit_fake))**2
    return gen_loss.mean()


def d_hinge(d_logit_real, d_logit_fake, DDP):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))


def g_hinge(d_logit_fake, DDP):
    return -torch.mean(d_logit_fake)


def d_wasserstein(d_logit_real, d_logit_fake, DDP):
    return torch.mean(d_logit_fake - d_logit_real)


def g_wasserstein(d_logit_fake, DDP):
    return -torch.mean(d_logit_fake)


def crammer_singer_loss(adv_output, label, DDP, **_):
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


def lecam_reg(d_logit_real, d_logit_fake, ema):
    reg = torch.mean(F.relu(d_logit_real - ema.D_fake).pow(2)) + \
          torch.mean(F.relu(ema.D_real - d_logit_fake).pow(2))
    return reg


def cal_deriv(inputs, outputs, device):
    grads = autograd.grad(outputs=outputs,
                          inputs=inputs,
                          grad_outputs=torch.ones(outputs.size()).to(device),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    return grads


def latent_optimise(zs, fake_labels, generator, discriminator, batch_size, lo_rate, lo_steps, lo_alpha, lo_beta, eval,
                    cal_trsp_cost, device):
    for step in range(lo_steps - 1):
        drop_mask = (torch.FloatTensor(batch_size, 1).uniform_() > 1 - lo_rate).to(device)

        zs = autograd.Variable(zs, requires_grad=True)
        fake_images = generator(zs, fake_labels, eval=eval)
        fake_dict = discriminator(fake_images, fake_labels, eval=eval)
        z_grads = cal_deriv(inputs=zs, outputs=fake_dict["adv_output"], device=device)
        z_grads_norm = torch.unsqueeze((z_grads.norm(2, dim=1)**2), dim=1)
        delta_z = lo_alpha * z_grads / (lo_beta + z_grads_norm)
        zs = torch.clamp(zs + drop_mask * delta_z, -1.0, 1.0)

        if cal_trsp_cost:
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
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean() + interpolates[:,0,0,0].mean()*0
    return grad_penalty


def cal_dra_penalty(real_images, real_labels, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    differences = 0.5 * real_images.std() * torch.rand(real_images.size()).to(device)
    interpolates = real_images + (alpha * differences)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean() + interpolates[:,0,0,0].mean()*0
    return grad_penalty


def cal_maxgrad_penalty(real_images, real_labels, fake_images, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_images.nelement() // batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    maxgrad_penalty = torch.max(grads.norm(2, dim=1)**2) + interpolates[:,0,0,0].mean()*0
    return maxgrad_penalty


def cal_r1_reg(adv_output, images, device):
    batch_size = images.size(0)
    grad_dout = cal_deriv(inputs=images, outputs=adv_output.sum(), device=device)
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == images.size())
    r1_reg = 0.5 * grad_dout2.contiguous().view(batch_size, -1).sum(1).mean(0) + images[:,0,0,0].mean()*0
    return r1_reg


def adjust_k(current_k, topk_gamma, inf_k):
    current_k = max(current_k * topk_gamma, inf_k)
    return current_k


def normal_nll_loss(x, mu, var):
    # https://github.com/Natsu6767/InfoGAN-PyTorch/blob/master/utils.py
    # Calculate the negative log likelihood of normal distribution.
    # Needs to be minimized in InfoGAN. (Treats Q(c]x) as a factored Gaussian)
    logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
    nll = -(logli.sum(1).mean())
    return nll


def stylegan_cal_r1_reg(adv_output, images):
    with conv2d_gradfix.no_weight_gradients():
        r1_grads = torch.autograd.grad(outputs=[adv_output.sum()], inputs=[images], create_graph=True, only_inputs=True)[0]
    r1_penalty = r1_grads.square().sum([1,2,3]) / 2
    return r1_penalty.mean()
