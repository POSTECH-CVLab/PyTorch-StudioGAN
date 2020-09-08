"""
MIT License

Copyright (c) 2019 Kim Seonghyeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import math

from utils.ada_op import upfirdn2d

import torch
from torch.nn import functional as F



SYM6 = (
    0.015404109327027373,
    0.0034907120842174702,
    -0.11799011114819057,
    -0.048311742585633,
    0.4910559419267466,
    0.787641141030194,
    0.3379294217276218,
    -0.07263752278646252,
    -0.021060292512300564,
    0.04472490177066578,
    0.0017677118642428036,
    -0.007800708325034148,
)


def translate_mat(t_x, t_y):
    batch = t_x.shape[0]

    mat = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    translate = torch.stack((t_x, t_y), 1)
    mat[:, :2, 2] = translate

    return mat


def rotate_mat(theta):
    batch = theta.shape[0]

    mat = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    rot = torch.stack((cos_t, -sin_t, sin_t, cos_t), 1).view(batch, 2, 2)
    mat[:, :2, :2] = rot

    return mat


def scale_mat(s_x, s_y):
    batch = s_x.shape[0]

    mat = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y

    return mat


def translate3d_mat(t_x, t_y, t_z):
    batch = t_x.shape[0]

    mat = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    translate = torch.stack((t_x, t_y, t_z), 1)
    mat[:, :3, 3] = translate

    return mat


def rotate3d_mat(axis, theta):
    batch = theta.shape[0]

    u_x, u_y, u_z = axis

    eye = torch.eye(3).unsqueeze(0)
    cross = torch.tensor([(0, -u_z, u_y), (u_z, 0, -u_x), (-u_y, u_x, 0)]).unsqueeze(0)
    outer = torch.tensor(axis)
    outer = (outer.unsqueeze(1) * outer).unsqueeze(0)

    sin_t = torch.sin(theta).view(-1, 1, 1)
    cos_t = torch.cos(theta).view(-1, 1, 1)

    rot = cos_t * eye + sin_t * cross + (1 - cos_t) * outer

    eye_4 = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    eye_4[:, :3, :3] = rot

    return eye_4


def scale3d_mat(s_x, s_y, s_z):
    batch = s_x.shape[0]

    mat = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y
    mat[:, 2, 2] = s_z

    return mat


def luma_flip_mat(axis, i):
    batch = i.shape[0]

    eye = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    axis = torch.tensor(axis + (0,))
    flip = 2 * torch.ger(axis, axis) * i.view(-1, 1, 1)

    return eye - flip


def saturation_mat(axis, i):
    batch = i.shape[0]

    eye = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    axis = torch.tensor(axis + (0,))
    axis = torch.ger(axis, axis)
    saturate = axis + (eye - axis) * i.view(-1, 1, 1)

    return saturate


def lognormal_sample(size, mean=0, std=1):
    return torch.empty(size).log_normal_(mean=mean, std=std)


def category_sample(size, categories):
    category = torch.tensor(categories)
    sample = torch.randint(high=len(categories), size=(size,))

    return category[sample]


def uniform_sample(size, low, high):
    return torch.empty(size).uniform_(low, high)


def normal_sample(size, mean=0, std=1):
    return torch.empty(size).normal_(mean, std)


def bernoulli_sample(size, p):
    return torch.empty(size).bernoulli_(p)


def random_mat_apply(p, transform, prev, eye):
    size = transform.shape[0]
    select = bernoulli_sample(size, p).view(size, 1, 1)
    select_transform = select * transform + (1 - select) * eye

    return select_transform @ prev


def sample_affine(p, size, height, width):
    G = torch.eye(3).unsqueeze(0).repeat(size, 1, 1)
    eye = G

    # flip
    param = category_sample(size, (0, 1))
    Gc = scale_mat(1 - 2.0 * param, torch.ones(size))
    G = random_mat_apply(p, Gc, G, eye)
    # print('flip', G, scale_mat(1 - 2.0 * param, torch.ones(size)), sep='\n')

    # 90 rotate
    param = category_sample(size, (0, 3))
    Gc = rotate_mat(-math.pi / 2 * param)
    G = random_mat_apply(p, Gc, G, eye)
    # print('90 rotate', G, rotate_mat(-math.pi / 2 * param), sep='\n')

    # integer translate
    param = uniform_sample(size, -0.125, 0.125)
    param_height = torch.round(param * height) / height
    param_width = torch.round(param * width) / width
    Gc = translate_mat(param_width, param_height)
    G = random_mat_apply(p, Gc, G, eye)
    # print('integer translate', G, translate_mat(param_width, param_height), sep='\n')

    # isotropic scale
    param = lognormal_sample(size, std=0.2 * math.log(2))
    Gc = scale_mat(param, param)
    G = random_mat_apply(p, Gc, G, eye)
    # print('isotropic scale', G, scale_mat(param, param), sep='\n')

    p_rot = 1 - math.sqrt(1 - p)

    # pre-rotate
    param = uniform_sample(size, -math.pi, math.pi)
    Gc = rotate_mat(-param)
    G = random_mat_apply(p_rot, Gc, G, eye)
    # print('pre-rotate', G, rotate_mat(-param), sep='\n')

    # anisotropic scale
    param = lognormal_sample(size, std=0.2 * math.log(2))
    Gc = scale_mat(param, 1 / param)
    G = random_mat_apply(p, Gc, G, eye)
    # print('anisotropic scale', G, scale_mat(param, 1 / param), sep='\n')

    # post-rotate
    param = uniform_sample(size, -math.pi, math.pi)
    Gc = rotate_mat(-param)
    G = random_mat_apply(p_rot, Gc, G, eye)
    # print('post-rotate', G, rotate_mat(-param), sep='\n')

    # fractional translate
    param = normal_sample(size, std=0.125)
    Gc = translate_mat(param, param)
    G = random_mat_apply(p, Gc, G, eye)
    # print('fractional translate', G, translate_mat(param, param), sep='\n')

    return G


def sample_color(p, size):
    C = torch.eye(4).unsqueeze(0).repeat(size, 1, 1)
    eye = C
    axis_val = 1 / math.sqrt(3)
    axis = (axis_val, axis_val, axis_val)

    # brightness
    param = normal_sample(size, std=0.2)
    Cc = translate3d_mat(param, param, param)
    C = random_mat_apply(p, Cc, C, eye)

    # contrast
    param = lognormal_sample(size, std=0.5 * math.log(2))
    Cc = scale3d_mat(param, param, param)
    C = random_mat_apply(p, Cc, C, eye)

    # luma flip
    param = category_sample(size, (0, 1))
    Cc = luma_flip_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    # hue rotation
    param = uniform_sample(size, -math.pi, math.pi)
    Cc = rotate3d_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    # saturation
    param = lognormal_sample(size, std=1 * math.log(2))
    Cc = saturation_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    return C


def make_grid(shape, x0, x1, y0, y1, device):
    n, c, h, w = shape
    grid = torch.empty(n, h, w, 3, device=device)
    grid[:, :, :, 0] = torch.linspace(x0, x1, w, device=device)
    grid[:, :, :, 1] = torch.linspace(y0, y1, h, device=device).unsqueeze(-1)
    grid[:, :, :, 2] = 1

    return grid


def affine_grid(grid, mat):
    n, h, w, _ = grid.shape
    return (grid.view(n, h * w, 3) @ mat.transpose(1, 2)).view(n, h, w, 2)


def get_padding(G, height, width):
    extreme = (
        G[:, :2, :]
        @ torch.tensor([(-1.0, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]).t()
    )

    size = torch.tensor((width, height))

    pad_low = (
        ((extreme.min(-1).values + 1) * size)
        .clamp(max=0)
        .abs()
        .ceil()
        .max(0)
        .values.to(torch.int64)
        .tolist()
    )
    pad_high = (
        (extreme.max(-1).values * size - size)
        .clamp(min=0)
        .ceil()
        .max(0)
        .values.to(torch.int64)
        .tolist()
    )

    return pad_low[0], pad_high[0], pad_low[1], pad_high[1]


def try_sample_affine_and_pad(img, p, pad_k, G=None):
    batch, _, height, width = img.shape

    G_try = G

    while True:
        if G is None:
            G_try = sample_affine(p, batch, height, width)

        pad_x1, pad_x2, pad_y1, pad_y2 = get_padding(
            torch.inverse(G_try), height, width
        )

        try:
            img_pad = F.pad(
                img,
                (pad_x1 + pad_k, pad_x2 + pad_k, pad_y1 + pad_k, pad_y2 + pad_k),
                mode="reflect",
            )

        except RuntimeError:
            continue

        break

    return img_pad, G_try, (pad_x1, pad_x2, pad_y1, pad_y2)


def random_apply_affine(img, p, G=None, antialiasing_kernel=SYM6):
    kernel = antialiasing_kernel
    len_k = len(kernel)
    pad_k = (len_k + 1) // 2

    kernel = torch.as_tensor(kernel)
    kernel = torch.ger(kernel, kernel).to(img)
    kernel_flip = torch.flip(kernel, (0, 1))

    img_pad, G, (pad_x1, pad_x2, pad_y1, pad_y2) = try_sample_affine_and_pad(
        img, p, pad_k, G
    )

    p_ux1 = pad_x1
    p_ux2 = pad_x2 + 1
    p_uy1 = pad_y1
    p_uy2 = pad_y2 + 1
    w_p = img_pad.shape[3] - len_k + 1
    h_p = img_pad.shape[2] - len_k + 1
    h_o = img.shape[2]
    w_o = img.shape[3]

    img_2x = upfirdn2d(img_pad, kernel_flip, up=2)

    grid = make_grid(
        img_2x.shape,
        -2 * p_ux1 / w_o - 1,
        2 * (w_p - p_ux1) / w_o - 1,
        -2 * p_uy1 / h_o - 1,
        2 * (h_p - p_uy1) / h_o - 1,
        device=img_2x.device,
    ).to(img_2x)
    grid = affine_grid(grid, torch.inverse(G)[:, :2, :].to(img_2x))
    grid = grid * torch.tensor(
        [w_o / w_p, h_o / h_p], device=grid.device
    ) + torch.tensor(
        [(w_o + 2 * p_ux1) / w_p - 1, (h_o + 2 * p_uy1) / h_p - 1], device=grid.device
    )

    img_affine = F.grid_sample(
        img_2x, grid, mode="bilinear", align_corners=False, padding_mode="zeros"
    )

    img_down = upfirdn2d(img_affine, kernel, down=2)

    end_y = -pad_y2 - 1
    if end_y == 0:
        end_y = img_down.shape[2]

    end_x = -pad_x2 - 1
    if end_x == 0:
        end_x = img_down.shape[3]

    img = img_down[:, :, pad_y1:end_y, pad_x1:end_x]

    return img, G


def apply_color(img, mat):
    batch = img.shape[0]
    img = img.permute(0, 2, 3, 1)
    mat_mul = mat[:, :3, :3].transpose(1, 2).view(batch, 1, 3, 3)
    mat_add = mat[:, :3, 3].view(batch, 1, 1, 3)
    img = img @ mat_mul + mat_add
    img = img.permute(0, 3, 1, 2)

    return img


def random_apply_color(img, p, C=None):
    if C is None:
        C = sample_color(p, img.shape[0])

    img = apply_color(img, C.to(img))

    return img, C


def augment(img, p, transform_matrix=(None, None)):
    img, G = random_apply_affine(img, p, transform_matrix[0])
    img, C = random_apply_color(img, p, transform_matrix[1])

    return img, (G, C)
