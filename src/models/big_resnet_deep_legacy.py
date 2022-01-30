# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/deep_big_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.ops as ops
import utils.misc as misc


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_cond_mtd, affine_input_dim, upsample,
                 MODULES, channel_ratio=4):
        super(GenBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.g_cond_mtd = g_cond_mtd
        self.upsample = upsample
        self.hidden_channels = self.in_channels // channel_ratio

        self.bn1 = MODULES.g_bn(affine_input_dim, self.in_channels, MODULES)
        self.bn2 = MODULES.g_bn(affine_input_dim, self.hidden_channels, MODULES)
        self.bn3 = MODULES.g_bn(affine_input_dim, self.hidden_channels, MODULES)
        self.bn4 = MODULES.g_bn(affine_input_dim, self.hidden_channels, MODULES)

        self.activation = MODULES.g_act_fn

        self.conv2d1 = MODULES.g_conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d2 = MODULES.g_conv2d(in_channels=self.hidden_channels,
                                        out_channels=self.hidden_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.conv2d3 = MODULES.g_conv2d(in_channels=self.hidden_channels,
                                        out_channels=self.hidden_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.conv2d4 = MODULES.g_conv2d(in_channels=self.hidden_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, affine):
        if self.in_channels != self.out_channels:
            x0 = x[:, :self.out_channels]
        else:
            x0 = x

        x = self.bn1(x, affine)
        x = self.conv2d1(self.activation(x))

        x = self.bn2(x, affine)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # upsample
        x = self.conv2d2(x)

        x = self.bn3(x, affine)
        x = self.conv2d3(self.activation(x))

        x = self.bn4(x, affine)
        x = self.conv2d4(self.activation(x))

        if self.upsample:
            x0 = F.interpolate(x0, scale_factor=2, mode="nearest")  # upsample
        out = x + x0
        return out


class Generator(nn.Module):
    def __init__(self, z_dim, g_shared_dim, img_size, g_conv_dim, apply_attn, attn_g_loc, g_cond_mtd, num_classes, g_init, g_depth,
                 mixed_precision, MODULES, MODEL):
        super(Generator, self).__init__()
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
        }

        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
        }

        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.g_shared_dim = g_shared_dim
        self.g_cond_mtd = g_cond_mtd
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.MODEL = MODEL
        self.in_dims = g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.num_blocks = len(self.in_dims)
        self.affine_input_dim = self.z_dim

        info_dim = 0
        if self.MODEL.info_type in ["discrete", "both"]:
            info_dim += self.MODEL.info_num_discrete_c*self.MODEL.info_dim_discrete_c
        if self.MODEL.info_type in ["continuous", "both"]:
            info_dim += self.MODEL.info_num_conti_c

        if self.MODEL.info_type != "N/A":
            if self.MODEL.g_info_injection == "concat":
                self.info_mix_linear = MODULES.g_linear(in_features=self.z_dim + info_dim, out_features=self.z_dim, bias=True)
            elif self.MODEL.g_info_injection == "cBN":
                self.affine_input_dim += self.g_shared_dim
                self.info_proj_linear = MODULES.g_linear(in_features=info_dim, out_features=self.g_shared_dim, bias=True)

        if self.g_cond_mtd != "W/O":
            self.affine_input_dim += self.g_shared_dim
            self.shared = ops.embedding(num_embeddings=self.num_classes, embedding_dim=self.g_shared_dim)

        self.linear0 = MODULES.g_linear(in_features=self.affine_input_dim, out_features=self.in_dims[0]*self.bottom*self.bottom, bias=True)


        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [[
                GenBlock(in_channels=self.in_dims[index],
                         out_channels=self.in_dims[index] if g_index == 0 else self.out_dims[index],
                         g_cond_mtd=g_cond_mtd,
                         affine_input_dim=self.affine_input_dim,
                         upsample=True if g_index == (g_depth - 1) else False,
                         MODULES=MODULES)
            ] for g_index in range(g_depth)]

            if index + 1 in attn_g_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=True, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = ops.batchnorm_2d(in_features=self.out_dims[-1])
        self.activation = MODULES.g_act_fn
        self.conv2d5 = MODULES.g_conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        ops.init_weights(self.modules, g_init)

    def forward(self, z, label, shared_label=None, eval=False):
        affine_list = []
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            if self.MODEL.info_type != "N/A":
                if self.MODEL.g_info_injection == "concat":
                    z = self.info_mix_linear(z)
                elif self.MODEL.g_info_injection == "cBN":
                    z, z_info = z[:, :self.z_dim], z[:, self.z_dim:]
                    affine_list.append(self.info_proj_linear(z_info))

            if self.g_cond_mtd != "W/O":
                if shared_label is None:
                    shared_label = self.shared(label)
                affine_list.append(shared_label)
            if len(affine_list) > 0:
                z = torch.cat(affine_list + [z], 1)

            affine = z
            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, ops.SelfAttention):
                        act = block(act)
                    else:
                        act = block(act, affine)

            act = self.bn4(act)
            act = self.activation(act)
            act = self.conv2d5(act)
            out = self.tanh(act)
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, MODULES, downsample=True, channel_ratio=4):
        super(DiscBlock, self).__init__()
        self.downsample = downsample
        hidden_channels = out_channels // channel_ratio

        self.activation = MODULES.d_act_fn
        self.conv2d1 = MODULES.d_conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d2 = MODULES.d_conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d3 = MODULES.d_conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d4 = MODULES.d_conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.learnable_sc = True if (in_channels != out_channels) else False
        if self.learnable_sc:
            self.conv2d0 = MODULES.d_conv2d(in_channels=in_channels,
                                            out_channels=out_channels - in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)

        if self.downsample:
            self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x

        x = self.conv2d1(self.activation(x))
        x = self.conv2d2(self.activation(x))
        x = self.conv2d3(self.activation(x))
        x = self.activation(x)

        if self.downsample:
            x = self.average_pooling(x)

        x = self.conv2d4(x)

        if self.downsample:
            x0 = self.average_pooling(x0)
        if self.learnable_sc:
            x0 = torch.cat([x0, self.conv2d0(x0)], 1)

        out = x + x0
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size, d_conv_dim, apply_d_sn, apply_attn, attn_d_loc, d_cond_mtd, aux_cls_type, d_embed_dim, normalize_d_embed,
                 num_classes, d_init, d_depth, mixed_precision, MODULES, MODEL):
        super(Discriminator, self).__init__()
        d_in_dims_collection = {
            "32": [d_conv_dim * 4, d_conv_dim * 4, d_conv_dim * 4],
            "64": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8],
            "128": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "256": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16],
            "512": [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16]
        }

        d_out_dims_collection = {
            "32": [d_conv_dim * 4, d_conv_dim * 4, d_conv_dim * 4],
            "64": [d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "128": [d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "256": [d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "512":
            [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16]
        }

        d_down = {
            "32": [True, True, False, False],
            "64": [True, True, True, True, False],
            "128": [True, True, True, True, True, False],
            "256": [True, True, True, True, True, True, False],
            "512": [True, True, True, True, True, True, True, False]
        }

        self.d_cond_mtd = d_cond_mtd
        self.aux_cls_type = aux_cls_type
        self.normalize_d_embed = normalize_d_embed
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.in_dims = d_in_dims_collection[str(img_size)]
        self.out_dims = d_out_dims_collection[str(img_size)]
        self.MODEL = MODEL
        down = d_down[str(img_size)]

        self.input_conv = MODULES.d_conv2d(in_channels=3, out_channels=self.in_dims[0], kernel_size=3, stride=1, padding=1)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[
                DiscBlock(in_channels=self.in_dims[index] if d_index == 0 else self.out_dims[index],
                          out_channels=self.out_dims[index],
                          MODULES=MODULES,
                          downsample=True if down[index] and d_index == 0 else False)
            ] for d_index in range(d_depth)]

            if (index+1) in attn_d_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=False, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = MODULES.d_act_fn

        # linear layer for adversarial training
        if self.d_cond_mtd == "MH":
            self.linear1 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=1 + num_classes, bias=True)
        elif self.d_cond_mtd == "MD":
            self.linear1 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=num_classes, bias=True)
        else:
            self.linear1 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=1, bias=True)

        # double num_classes for Auxiliary Discriminative Classifier
        if self.aux_cls_type == "ADC":
            num_classes = num_classes * 2

        # linear and embedding layers for discriminator conditioning
        if self.d_cond_mtd == "AC":
            self.linear2 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=num_classes, bias=False)
        elif self.d_cond_mtd == "PD":
            self.embedding = MODULES.d_embedding(num_classes, self.out_dims[-1])
        elif self.d_cond_mtd in ["2C", "D2DCE"]:
            self.linear2 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=d_embed_dim, bias=True)
            self.embedding = MODULES.d_embedding(num_classes, d_embed_dim)
        else:
            pass

        # linear and embedding layers for evolved classifier-based GAN
        if self.aux_cls_type == "TAC":
            if self.d_cond_mtd == "AC":
                self.linear_mi = MODULES.d_linear(in_features=self.out_dims[-1], out_features=num_classes, bias=False)
            elif self.d_cond_mtd in ["2C", "D2DCE"]:
                self.linear_mi = MODULES.d_linear(in_features=self.out_dims[-1], out_features=d_embed_dim, bias=True)
                self.embedding_mi = MODULES.d_embedding(num_classes, d_embed_dim)
            else:
                raise NotImplementedError

        # Q head network for infoGAN
        if self.MODEL.info_type in ["discrete", "both"]:
            out_features = self.MODEL.info_num_discrete_c*self.MODEL.info_dim_discrete_c
            self.info_discrete_linear = MODULES.d_linear(in_features=self.out_dims[-1], out_features=out_features, bias=False)
        if self.MODEL.info_type in ["continuous", "both"]:
            out_features = self.MODEL.info_num_conti_c
            self.info_conti_mu_linear = MODULES.d_linear(in_features=self.out_dims[-1], out_features=out_features, bias=False)
            self.info_conti_var_linear = MODULES.d_linear(in_features=self.out_dims[-1], out_features=out_features, bias=False)

        if d_init:
            ops.init_weights(self.modules, d_init)

    def forward(self, x, label, eval=False, adc_fake=False):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            embed, proxy, cls_output = None, None, None
            mi_embed, mi_proxy, mi_cls_output = None, None, None
            info_discrete_c_logits, info_conti_mu, info_conti_var = None, None, None
            h = self.input_conv(x)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
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
