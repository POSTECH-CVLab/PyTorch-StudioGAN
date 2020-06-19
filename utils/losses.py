import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import autograd
from models.model_ops import snlinear, linear
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

    def forward(self, inst_embed, anchor, cls_mask, labels, temperature, augmentation=None):
        if augmentation is not None:
            representations = torch.cat([inst_embed, augmentation], dim=0)
            candidates = self.calculate_similarity_matrix(representations, representations)
            inst2aug_positive = torch.exp(candidates[self.hard_positive_mask]/temperature)
            instance_zone = torch.exp(self.remove_diag(candidates[:self.batch_size, :self.batch_size])/temperature)

            mask_4_remove_negatives = cls_mask[labels]
            mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)

            inst2inst_positives = instance_zone*mask_4_remove_negatives
            inst2embed_positive = torch.exp(self.cosine_similarity(inst_embed, anchor)/temperature)

            numerator = (inst2inst_positives.sum(dim=1)+inst2embed_positive + inst2aug_positive)
            denomerator = torch.cat([torch.unsqueeze(inst2aug_positive, dim=1), torch.unsqueeze(inst2embed_positive, dim=1),
                                     instance_zone], dim=1).sum(dim=1)

            criterion = -torch.log(numerator/denomerator).mean()
        else:
            similarity_matrix = self.calculate_similarity_matrix(inst_embed, inst_embed)
            instance_zone = torch.exp(self.remove_diag(similarity_matrix)/temperature)

            mask_4_remove_negatives = cls_mask[labels]
            mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)

            inst2inst_positives = instance_zone*mask_4_remove_negatives
            inst2embed_positive = torch.exp(self.cosine_similarity(inst_embed, anchor)/temperature)

            numerator = (inst2inst_positives.sum(dim=1)+inst2embed_positive)
            denomerator = torch.cat([torch.unsqueeze(inst2embed_positive, dim=1), instance_zone], dim=1).sum(dim=1)

            criterion = -torch.log(numerator/denomerator).mean()
        return criterion


class Class_Conditional_Repulsion_loss(torch.nn.Module):
    def __init__(self, device, batch_size):
        super(Class_Conditional_Repulsion_loss, self).__init__()
        self.batch_size = batch_size
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

    def forward(self, images, cls_mask, labels):
        channel_mean = torch.mean(images, dim=(2,3))
        similarity_matrix = self.calculate_similarity_matrix(channel_mean, channel_mean)
        mask_4_remove_negatives = cls_mask[labels]
        mask_4_remove_negatives = mask_4_remove_negatives.type(torch.bool).to(self.device)
        criterion = similarity_matrix[mask_4_remove_negatives].mean()
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