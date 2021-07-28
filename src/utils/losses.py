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


def d_vanilla(d_logit_real, d_logit_fake):
    device = d_logit_real.get_device()
    ones = torch.ones_like(d_logit_real, device=device, requires_grad=False)
    d_loss = -torch.mean(nn.LogSigmoid()(d_logit_real) + nn.LogSigmoid()(ones - d_logit_fake))
    return d_loss

def g_vanilla(g_logit_fake):
    return -torch.mean(nn.LogSigmoid()(g_logit_fake))

def d_ls(d_logit_real, d_logit_fake):
    d_loss = 0.5*(d_logit_real - torch.ones_like(d_logit_real))**2 + 0.5*(d_logit_fake)**2
    return d_loss.mean()

def g_ls(d_logit_fake):
    gen_loss = 0.5*(d_logit_fake - torch.ones_like(d_logit_fake))**2
    return gen_loss.mean()

def d_hinge(d_logit_real, d_logit_fake):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))

def g_hinge(g_logit_fake):
    return -torch.mean(g_logit_fake)

def d_wasserstein(d_logit_real, d_logit_fake):
    return torch.mean(d_logit_fake - d_logit_real)

def g_wasserstein(g_logit_fake):
    return -torch.mean(g_logit_fake)

def latent_optimise(zs, fake_labels, gen_model, dis_model, conditional_strategy, latent_op_step, latent_op_rate,
                    latent_op_alpha, latent_op_beta, trans_cost, default_device):
    batch_size = zs.shape[0]
    for step in range(latent_op_step):
        drop_mask = (torch.FloatTensor(batch_size, 1).uniform_() > 1 - latent_op_rate).to(default_device)
        z_gradients, z_gradients_norm = calc_derv(zs, fake_labels, dis_model, conditional_strategy, default_device, gen_model)
        delta_z = latent_op_alpha*z_gradients/(latent_op_beta + z_gradients_norm)
        zs = torch.clamp(zs + drop_mask*delta_z, -1.0, 1.0)

        if trans_cost:
            if step == 0:
                transport_cost = (delta_z.norm(2, dim=1)**2).mean()
            else:
                transport_cost += (delta_z.norm(2, dim=1)**2).mean()

    if trans_cost:
        return zs, trans_cost
    else:
        return zs


def set_temperature(conditional_strategy, tempering_type, start_temperature, end_temperature, step_count, tempering_step, total_step):
    if conditional_strategy == 'ContraGAN':
        if tempering_type == 'continuous':
            t = start_temperature + step_count*(end_temperature - start_temperature)/total_step
        elif tempering_type == 'discrete':
            tempering_interval = total_step//(tempering_step + 1)
            t = start_temperature + \
                (step_count//tempering_interval)*(end_temperature-start_temperature)/tempering_step
        else:
            t = start_temperature
    else:
        t = 'no'
    return t


class Cross_Entropy_loss(torch.nn.Module):
    def __init__(self, in_features, out_features, spectral_norm=True):
        super(Cross_Entropy_loss, self).__init__()

        if spectral_norm:
            self.layer =  snlinear(in_features=in_features, out_features=out_features, bias=True)
        else:
            self.layer =  linear(in_features=in_features, out_features=out_features, bias=True)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, embeds, labels):
        logits = self.layer(embeds)
        return self.ce_loss(logits, labels)

class ConditionalContrastive(torch.nn.Module):
    def __init__(self, device):
        super(ConditionalContrastive, self).__init__()
        self.device = device
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, mask, labels, temperature):
        sim_matrix = self.calculate_similarity_matrix(embed, embed)
        sim_matrix = torch.exp(self.remove_diag(sim_matrix)/temperature)
        neg_removal_mask = self.remove_diag(mask[labels])
        sim_btw_pos = neg_removal_mask*sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy)/temperature)

        numerator = emb2proxy + sim_btw_pos.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        criterion = -torch.log(numerator/denomerator)
        return criterion.mean()

def calc_derv4gp(netD, conditional_strategy, real_data, fake_data, real_labels, device):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(batch_size,c,h,w)
    alpha = alpha.to(device)

    real_data = real_data.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
            disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy == 'ACGAN':
        _, disc_interpolates = netD(interpolates, real_labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_derv4dra(netD, conditional_strategy, real_data, real_labels, device):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.to(device)

    real_data = real_data.to(device)
    differences  = 0.5*real_data.std()*torch.rand(real_data.size()).to(device)

    interpolates = real_data + (alpha*differences)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
            disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy == 'ACGAN':
        _, disc_interpolates = netD(interpolates, real_labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_derv(inputs, labels, netD, conditional_strategy, device, netG=None):
    zs = autograd.Variable(inputs, requires_grad=True)
    fake_images = netG(zs, labels)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, dis_out_fake = netD(fake_images, labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
        dis_out_fake = netD(fake_images, labels)
    elif conditional_strategy == 'ACGAN':
        _, dis_out_fake = netD(fake_images, labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=dis_out_fake, inputs=zs,
                              grad_outputs=torch.ones(dis_out_fake.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients_norm = torch.unsqueeze((gradients.norm(2, dim=1) ** 2), dim=1)
    return gradients, gradients_norm
