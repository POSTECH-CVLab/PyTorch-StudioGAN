# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/losses.py


from models.model_ops import snlinear, linear

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import autograd

import numpy as np 



# DCGAN loss
def loss_dcgan_dis(dis_out_real, dis_out_fake):
    device = dis_out_real.get_device()
    ones = torch.ones_like(dis_out_real, device=device, requires_grad=False)
    dis_loss = -torch.mean(nn.LogSigmoid()(dis_out_real) + nn.LogSigmoid()(ones - dis_out_fake))
    return dis_loss


def loss_dcgan_gen(gen_out_fake):
    return -torch.mean(nn.LogSigmoid()(gen_out_fake))


def loss_hinge_dis(dis_out_real, dis_out_fake):
    return torch.mean(F.relu(1. - dis_out_real)) + torch.mean(F.relu(1. + dis_out_fake))


def loss_hinge_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)


def loss_wgan_dis(dis_out_real, dis_out_fake):
    return torch.mean(dis_out_fake - dis_out_real)


def loss_wgan_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)


def latent_optimise(z, fake_labels, gen_model, dis_model, latent_op_step, latent_op_rate, 
                    latent_op_alpha, latent_op_beta, trans_cost, default_device):
    batch_size = z.shape[0]
    for step in range(latent_op_step):
        drop_mask = (torch.FloatTensor(batch_size, 1).uniform_() > 1 - latent_op_rate).to(default_device)
        z_gradients, z_gradients_norm = calc_derv(z, fake_labels, dis_model, default_device, gen_model)
        delta_z = latent_op_alpha*z_gradients/(latent_op_beta + z_gradients_norm)
        z = torch.clamp(z + drop_mask*delta_z, -1.0, 1.0)

        if trans_cost:
            if step == 0:
                transport_cost = (delta_z.norm(2, dim=1)**2).mean()
            else:
                transport_cost += (delta_z.norm(2, dim=1)**2).mean()
            return z, trans_cost
        else:
            return z


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


class Conditional_Embedding_Contrastive_loss(torch.nn.Module):
    def __init__(self, device, batch_size):
        super(Conditional_Embedding_Contrastive_loss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.hard_positive_mask = self._get_hard_positive_mask().type(torch.bool)
        self.tp = nn.Parameter(torch.zeros(1))
        self.ti = nn.Parameter(torch.zeros(1))

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

    def _get_hard_positive_mask(self):
        hard_positive = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((hard_positive)).type(torch.bool)
        return mask.to(self.device)

    def forward(self, inst_embed, proxy, negative_mask, labels, temperature):
        similarity_matrix = self.calculate_similarity_matrix(inst_embed, inst_embed)
        instance_zone = torch.exp(self.remove_diag(similarity_matrix)/self.ti)

        mask_4_remove_negatives = negative_mask[labels]
        mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)

        inst2inst_positives = instance_zone*mask_4_remove_negatives
        inst2embed_positive = torch.exp(self.cosine_similarity(inst_embed, proxy)/self.tp)

        numerator = (inst2inst_positives.sum(dim=1)+inst2embed_positive)
        denomerator = torch.cat([torch.unsqueeze(inst2embed_positive, dim=1), instance_zone], dim=1).sum(dim=1)
        criterion = -torch.log(numerator/denomerator).mean()
        return criterion

def calc_derv4gp(netD, real_data, fake_data, real_labels, device):
    # print "real_data: ", real_data.size(), fake_data.size()
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(batch_size,c,h,w)
    alpha = alpha.to(device)

    real_data = real_data.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    _, _, disc_interpolates = netD(interpolates, real_labels)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_derv4lo(z, fake_labels, netG, netD, device):
    z_rg = autograd.Variable(z, requires_grad=True)
    fake_images = netG(z_rg, fake_labels)

    _, _, dis_fake_out = netD(fake_images, fake_labels)

    gradients = autograd.grad(outputs=dis_fake_out, inputs=z_rg,
                              grad_outputs=torch.ones(dis_fake_out.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients_norm = torch.unsqueeze((gradients.norm(2, dim=1) ** 2), dim=1)
    return gradients, gradients_norm


def calc_derv(inputs, labels, netD, device, netG=None):
    if netG is None:
        netD.eval()     

        X = autograd.Variable(inputs, requires_grad=True)

        _, _, outputs = netD(X, labels)

        gradients = autograd.grad(outputs=outputs, inputs=X, 
                                grad_outputs=torch.ones(outputs.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradients_norm = torch.unsqueeze((gradients.norm(2, dim=1) ** 2), dim=1)
        
        netD.train()
        return gradients_norm
    else:
        netD.eval()
        netG.eval()

        Z = autograd.Variable(inputs, requires_grad=True)
        fake_images = netG(Z, labels)
        
        _, _, dis_fake_out = netD(fake_images, labels)

        gradients = autograd.grad(outputs=dis_fake_out, inputs=Z,
                                 grad_outputs=torch.ones(dis_fake_out.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients_norm = torch.unsqueeze((gradients.norm(2, dim=1) ** 2), dim=1)

        netD.train()
        netG.train()
        return gradients, gradients_norm


### Differentiable Augmentation for Data-Efficient GAN Training (https://arxiv.org/abs/2006.10738)
### Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
### https://github.com/mit-han-lab/data-efficient-gans
def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_color(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=[1, 8]):
    shift_x, shift_y = x.size(2) // 8, x.size(3) // 8
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad[grid_batch, :, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=[1, 2]):
    cutout_size = x.size(2) // 2, x.size(3) // 2
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_color, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

### ---------------------------------------------------------------------------
