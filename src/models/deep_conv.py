# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/deep_conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.ops as ops
import utils.misc as misc


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_cond_mtd, g_info_injection, affine_input_dim, MODULES):
        super(GenBlock, self).__init__()
        self.g_cond_mtd = g_cond_mtd
        self.g_info_injection = g_info_injection

        self.deconv0 = MODULES.g_deconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

        if self.g_cond_mtd == "W/O" and self.g_info_injection in ["N/A", "concat"]:
            self.bn0 = MODULES.g_bn(in_features=out_channels)
        elif self.g_cond_mtd == "cBN" or self.g_info_injection == "cBN":
            self.bn0 = MODULES.g_bn(affine_input_dim, out_channels, MODULES)
        else:
            raise NotImplementedError

        self.activation = MODULES.g_act_fn

    def forward(self, x, affine):
        x = self.deconv0(x)
        if self.g_cond_mtd == "W/O" and self.g_info_injection in ["N/A", "concat"]:
            x = self.bn0(x)
        elif self.g_cond_mtd == "cBN" or self.g_info_injection == "cBN":
            x = self.bn0(x, affine)
        out = self.activation(x)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim, g_shared_dim, img_size, g_conv_dim, apply_attn, attn_g_loc, g_cond_mtd, num_classes, g_init, g_depth,
                 mixed_precision, MODULES, MODEL):
        super(Generator, self).__init__()
        self.in_dims = [512, 256, 128]
        self.out_dims = [256, 128, 64]

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.g_cond_mtd = g_cond_mtd
        self.mixed_precision = mixed_precision
        self.MODEL = MODEL
        self.affine_input_dim = 0

        info_dim = 0
        if self.MODEL.info_type in ["discrete", "both"]:
            info_dim += self.MODEL.info_num_discrete_c*self.MODEL.info_dim_discrete_c
        if self.MODEL.info_type in ["continuous", "both"]:
            info_dim += self.MODEL.info_num_conti_c

        self.g_info_injection = self.MODEL.g_info_injection
        if self.MODEL.info_type != "N/A":
            if self.g_info_injection == "concat":
                self.info_mix_linear = MODULES.g_linear(in_features=self.z_dim + info_dim, out_features=self.z_dim, bias=True)
            elif self.g_info_injection == "cBN":
                self.affine_input_dim += self.z_dim
                self.info_proj_linear = MODULES.g_linear(in_features=info_dim, out_features=self.z_dim, bias=True)

        if self.g_cond_mtd != "W/O" and self.g_cond_mtd == "cBN":
            self.affine_input_dim += self.num_classes

        self.linear0 = MODULES.g_linear(in_features=self.z_dim, out_features=self.in_dims[0]*4*4, bias=True)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[
                GenBlock(in_channels=self.in_dims[index],
                         out_channels=self.out_dims[index],
                         g_cond_mtd=self.g_cond_mtd,
                         g_info_injection=self.g_info_injection,
                         affine_input_dim=self.affine_input_dim,
                         MODULES=MODULES)
            ]]

            if index + 1 in attn_g_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=True, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.conv4 = MODULES.g_conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        ops.init_weights(self.modules, g_init)

    def forward(self, z, label, shared_label=None, eval=False):
        affine_list = []
        if self.g_cond_mtd != "W/O":
            label = F.one_hot(label, num_classes=self.num_classes).to(torch.float32)
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            if self.MODEL.info_type != "N/A":
                if self.g_info_injection == "concat":
                    z = self.info_mix_linear(z)
                elif self.g_info_injection == "cBN":
                    z, z_info = z[:, :self.z_dim], z[:, self.z_dim:]
                    affine_list.append(self.info_proj_linear(z_info))

            if self.g_cond_mtd != "W/O":
                affine_list.append(label)
            if len(affine_list) > 0:
                affines = torch.cat(affine_list, 1)
            else:
                affines = None

            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], 4, 4)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, ops.SelfAttention):
                        act = block(act)
                    else:
                        act = block(act, affines)

            act = self.conv4(act)
            out = self.tanh(act)
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn, MODULES):
        super(DiscBlock, self).__init__()
        self.apply_d_sn = apply_d_sn

        self.conv0 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = MODULES.d_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

        if not apply_d_sn:
            self.bn0 = MODULES.d_bn(in_features=out_channels)
            self.bn1 = MODULES.d_bn(in_features=out_channels)

        self.activation = MODULES.d_act_fn

    def forward(self, x):
        x = self.conv0(x)
        if not self.apply_d_sn:
            x = self.bn0(x)
        x = self.activation(x)

        x = self.conv1(x)
        if not self.apply_d_sn:
            x = self.bn1(x)
        out = self.activation(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size, d_conv_dim, apply_d_sn, apply_attn, attn_d_loc, d_cond_mtd, aux_cls_type, d_embed_dim, normalize_d_embed,
                 num_classes, d_init, d_depth, mixed_precision, MODULES, MODEL):
        super(Discriminator, self).__init__()
        self.in_dims = [3] + [64, 128]
        self.out_dims = [64, 128, 256]

        self.apply_d_sn = apply_d_sn
        self.d_cond_mtd = d_cond_mtd
        self.aux_cls_type = aux_cls_type
        self.normalize_d_embed = normalize_d_embed
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.MODEL= MODEL

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[
                DiscBlock(in_channels=self.in_dims[index], out_channels=self.out_dims[index], apply_d_sn=self.apply_d_sn, MODULES=MODULES)
            ]]

            if index + 1 in attn_d_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=False, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = MODULES.d_act_fn
        self.conv1 = MODULES.d_conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        if not self.apply_d_sn:
            self.bn1 = MODULES.d_bn(in_features=512)

        # linear layer for adversarial training
        if self.d_cond_mtd == "MH":
            self.linear1 = MODULES.d_linear(in_features=512, out_features=1 + num_classes, bias=True)
        elif self.d_cond_mtd == "MD":
            self.linear1 = MODULES.d_linear(in_features=512, out_features=num_classes, bias=True)
        else:
            self.linear1 = MODULES.d_linear(in_features=512, out_features=1, bias=True)

        # double num_classes for Auxiliary Discriminative Classifier
        if self.aux_cls_type == "ADC":
            num_classes = num_classes * 2

        # linear and embedding layers for discriminator conditioning
        if self.d_cond_mtd == "AC":
            self.linear2 = MODULES.d_linear(in_features=512, out_features=num_classes, bias=False)
        elif self.d_cond_mtd == "PD":
            self.embedding = MODULES.d_embedding(num_classes, 512)
        elif self.d_cond_mtd in ["2C", "D2DCE"]:
            self.linear2 = MODULES.d_linear(in_features=512, out_features=d_embed_dim, bias=True)
            self.embedding = MODULES.d_embedding(num_classes, d_embed_dim)
        else:
            pass

        # linear and embedding layers for evolved classifier-based GAN
        if self.aux_cls_type == "TAC":
            if self.d_cond_mtd == "AC":
                self.linear_mi = MODULES.d_linear(in_features=512, out_features=num_classes, bias=False)
            elif self.d_cond_mtd in ["2C", "D2DCE"]:
                self.linear_mi = MODULES.d_linear(in_features=512, out_features=d_embed_dim, bias=True)
                self.embedding_mi = MODULES.d_embedding(num_classes, d_embed_dim)
            else:
                raise NotImplementedError

        # Q head network for infoGAN
        if self.MODEL.info_type in ["discrete", "both"]:
            out_features = self.MODEL.info_num_discrete_c*self.MODEL.info_dim_discrete_c
            self.info_discrete_linear = MODULES.d_linear(in_features=512, out_features=out_features, bias=False)
        if self.MODEL.info_type in ["continuous", "both"]:
            out_features = self.MODEL.info_num_conti_c
            self.info_conti_mu_linear = MODULES.d_linear(in_features=512, out_features=out_features, bias=False)
            self.info_conti_var_linear = MODULES.d_linear(in_features=512, out_features=out_features, bias=False)

        if d_init:
            ops.init_weights(self.modules, d_init)

    def forward(self, x, label, eval=False, adc_fake=False):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            embed, proxy, cls_output = None, None, None
            mi_embed, mi_proxy, mi_cls_output = None, None, None
            info_discrete_c_logits, info_conti_mu, info_conti_var = None, None, None
            h = x
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.conv1(h)
            if not self.apply_d_sn:
                h = self.bn1(h)
            bottom_h, bottom_w = h.shape[2], h.shape[3]
            h = self.activation(h)
            h = torch.sum(h, dim=[2, 3])

            # adversarial training
            adv_output = torch.squeeze(self.linear1(h))

            # make class labels odd (for fake) or even (for real) for ADC
            if self.aux_cls_type == "ADC":
                if adc_fake:
                    label = label*2 + 1
                else:
                    label = label*2

            # forward pass through InfoGAN Q head
            if self.MODEL.info_type in ["discrete", "both"]:
                info_discrete_c_logits = self.info_discrete_linear(h/(bottom_h*bottom_w))
            if self.MODEL.info_type in ["continuous", "both"]:
                info_conti_mu = self.info_conti_mu_linear(h/(bottom_h*bottom_w))
                info_conti_var = torch.exp(self.info_conti_var_linear(h/(bottom_h*bottom_w)))

            # class conditioning
            if self.d_cond_mtd == "AC":
                if self.normalize_d_embed:
                    for W in self.linear2.parameters():
                        W = F.normalize(W, dim=1)
                    h = F.normalize(h, dim=1)
                cls_output = self.linear2(h)
            elif self.d_cond_mtd == "PD":
                adv_output = adv_output + torch.sum(torch.mul(self.embedding(label), h), 1)
            elif self.d_cond_mtd in ["2C", "D2DCE"]:
                embed = self.linear2(h)
                proxy = self.embedding(label)
                if self.normalize_d_embed:
                    embed = F.normalize(embed, dim=1)
                    proxy = F.normalize(proxy, dim=1)
            elif self.d_cond_mtd == "MD":
                idx = torch.LongTensor(range(label.size(0))).to(label.device)
                adv_output = adv_output[idx, label]
            elif self.d_cond_mtd in ["W/O", "MH"]:
                pass
            else:
                raise NotImplementedError

            # extra conditioning for TACGAN and ADCGAN
            if self.aux_cls_type == "TAC":
                if self.d_cond_mtd == "AC":
                    if self.normalize_d_embed:
                        for W in self.linear_mi.parameters():
                            W = F.normalize(W, dim=1)
                    mi_cls_output = self.linear_mi(h)
                elif self.d_cond_mtd in ["2C", "D2DCE"]:
                    mi_embed = self.linear_mi(h)
                    mi_proxy = self.embedding_mi(label)
                    if self.normalize_d_embed:
                        mi_embed = F.normalize(mi_embed, dim=1)
                        mi_proxy = F.normalize(mi_proxy, dim=1)
        return {
            "h": h,
            "adv_output": adv_output,
            "embed": embed,
            "proxy": proxy,
            "cls_output": cls_output,
            "label": label,
            "mi_embed": mi_embed,
            "mi_proxy": mi_proxy,
            "mi_cls_output": mi_cls_output,
            "info_discrete_c_logits": info_discrete_c_logits,
            "info_conti_mu": info_conti_mu,
            "info_conti_var": info_conti_var
        }
