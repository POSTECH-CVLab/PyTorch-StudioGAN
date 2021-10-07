"""
this code is borrowed from https://github.com/jh-jeong/ContraD with few modifications

MIT License

Copyright (c) 2021 Jongheon Jeong

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

import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.functional import affine_grid, grid_sample
from torch.nn import functional as F
from torch.autograd import Function
from kornia.filters import get_gaussian_kernel2d, filter2d
import numbers


def rgb2hsv(rgb):
    """Convert a 4-d RGB tensor to the HSV counterpart.
    Here, we compute hue using atan2() based on the definition in [1],
    instead of using the common lookup table approach as in [2, 3].
    Those values agree when the angle is a multiple of 30°,
    otherwise they may differ at most ~1.2°.
    >>> %timeit rgb2hsv_lookup(rgb)
    1.07 ms ± 2.96 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> %timeit rgb2hsv(rgb)
    380 µs ± 555 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> (rgb2hsv_lookup(rgb) - rgb2hsv(rgb)).abs().max()
    tensor(0.0031, device='cuda:0')
    References
    [1] https://en.wikipedia.org/wiki/Hue
    [2] https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    [3] https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L212
    """

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]

    hue = torch.atan2(math.sqrt(3) * (g - b), 2 * r - g - b)
    hue = (hue % (2 * math.pi)) / (2 * math.pi)
    saturate = 1 - Cmin / (Cmax + 1e-8)
    value = Cmax
    hsv = torch.stack([hue, saturate, value], dim=1)
    hsv[~torch.isfinite(hsv)] = 0.
    return hsv


def hsv2rgb(hsv):
    """Convert a 4-d HSV tensor to the RGB counterpart.
    >>> %timeit hsv2rgb_lookup(hsv)
    2.37 ms ± 13.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    >>> %timeit hsv2rgb(rgb)
    298 µs ± 542 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> torch.allclose(hsv2rgb(hsv), hsv2rgb_lookup(hsv), atol=1e-6)
    True
    References
    [1] https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    """

    h, s, v = hsv[:, [0]], hsv[:, [1]], hsv[:, [2]]
    c = v * s

    n = hsv.new_tensor([5, 3, 1]).view(3, 1, 1)
    k = (n + h * 6) % 6
    t = torch.min(k, 4. - k)
    t = torch.clamp(t, 0, 1)
    return v - c * t


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, inputs):
        _prob = inputs.new_full((inputs.size(0), ), self.p)
        _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
        return inputs * (1 - _mask) + self.fn(inputs) * _mask


class RandomResizeCropLayer(nn.Module):
    def __init__(self, scale, ratio=(3. / 4., 4. / 3.)):
        '''
            Inception Crop
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        '''
        super(RandomResizeCropLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)
        self.scale = scale
        self.ratio = ratio

    def forward(self, inputs):
        _device = inputs.device
        N, _, width, height = inputs.shape

        _theta = self._eye.repeat(N, 1, 1)

        # N * 10 trial
        area = height * width
        target_area = np.random.uniform(*self.scale, N * 10) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = np.exp(np.random.uniform(*log_ratio, N * 10))

        # If doesn't satisfy ratio condition, then do central crop
        w = np.round(np.sqrt(target_area * aspect_ratio))
        h = np.round(np.sqrt(target_area / aspect_ratio))
        cond = (0 < w) * (w <= width) * (0 < h) * (h <= height)
        w = w[cond]
        h = h[cond]
        if len(w) > N:
            inds = np.random.choice(len(w), N, replace=False)
            w = w[inds]
            h = h[inds]
        transform_len = len(w)

        r_w_bias = np.random.randint(w - width, width - w + 1) / width
        r_h_bias = np.random.randint(h - height, height - h + 1) / height
        w = w / width
        h = h / height

        _theta[:transform_len, 0, 0] = torch.tensor(w, device=_device)
        _theta[:transform_len, 1, 1] = torch.tensor(h, device=_device)
        _theta[:transform_len, 0, 2] = torch.tensor(r_w_bias, device=_device)
        _theta[:transform_len, 1, 2] = torch.tensor(r_h_bias, device=_device)

        grid = affine_grid(_theta, inputs.size(), align_corners=False)
        output = grid_sample(inputs, grid, padding_mode='reflection', align_corners=False)
        return output


class HorizontalFlipLayer(nn.Module):
    def __init__(self):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(HorizontalFlipLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, inputs):
        _device = inputs.device

        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)
        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        _theta[:, 0, 0] = r_sign
        grid = affine_grid(_theta, inputs.size(), align_corners=False)
        output = grid_sample(inputs, grid, padding_mode='reflection', align_corners=False)
        return output


class RandomHSVFunction(Function):
    @staticmethod
    def forward(ctx, x, f_h, f_s, f_v):
        # ctx is a context object that can be used to stash information
        # for backward computation
        x = rgb2hsv(x)
        h = x[:, 0, :, :]
        h += (f_h * 255. / 360.)
        h = (h % 1)
        x[:, 0, :, :] = h
        x[:, 1, :, :] = x[:, 1, :, :] * f_s
        x[:, 2, :, :] = x[:, 2, :, :] * f_v
        x = torch.clamp(x, 0, 1)
        x = hsv2rgb(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


class ColorJitterLayer(nn.Module):
    def __init__(self, brightness, contrast, saturation, hue):
        super(ColorJitterLayer, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        if self.contrast:
            factor = x.new_empty(x.size(0), 1, 1, 1).uniform_(*self.contrast)
            means = torch.mean(x, dim=[2, 3], keepdim=True)
            x = (x - means) * factor + means
        return torch.clamp(x, 0, 1)

    def adjust_hsv(self, x):
        f_h = x.new_zeros(x.size(0), 1, 1)
        f_s = x.new_ones(x.size(0), 1, 1)
        f_v = x.new_ones(x.size(0), 1, 1)

        if self.hue:
            f_h.uniform_(*self.hue)
        if self.saturation:
            f_s = f_s.uniform_(*self.saturation)
        if self.brightness:
            f_v = f_v.uniform_(*self.brightness)

        return RandomHSVFunction.apply(x, f_h, f_s, f_v)

    def transform(self, inputs):
        # Shuffle transform
        if np.random.rand() > 0.5:
            transforms = [self.adjust_contrast, self.adjust_hsv]
        else:
            transforms = [self.adjust_hsv, self.adjust_contrast]

        for t in transforms:
            inputs = t(inputs)
        return inputs

    def forward(self, inputs):
        return self.transform(inputs)


class RandomColorGrayLayer(nn.Module):
    def __init__(self):
        super(RandomColorGrayLayer, self).__init__()
        _weight = torch.tensor([[0.299, 0.587, 0.114]])
        self.register_buffer('_weight', _weight.view(1, 3, 1, 1))

    def forward(self, inputs):
        l = F.conv2d(inputs, self._weight)
        gray = torch.cat([l, l, l], dim=1)
        return gray


class GaussianBlur(nn.Module):
    def __init__(self, sigma_range):
        """Blurs the given image with separable convolution.
        Args:
            sigma_range: Range of sigma for being used in each gaussian kernel.
        """
        super(GaussianBlur, self).__init__()
        self.sigma_range = sigma_range

    def forward(self, inputs):
        _device = inputs.device

        batch_size, num_channels, height, width = inputs.size()

        kernel_size = height // 10
        radius = int(kernel_size / 2)
        kernel_size = radius * 2 + 1

        sigma = np.random.uniform(*self.sigma_range)
        kernel = torch.unsqueeze(get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma)), dim=0)
        blurred = filter2d(inputs, kernel, "reflect")
        return blurred


class CutOut(nn.Module):
    def __init__(self, length):
        super().__init__()
        if length % 2 == 0:
            raise ValueError("Currently CutOut only accepts odd lengths: length % 2 == 1")
        self.length = length

        _weight = torch.ones(1, 1, self.length)
        self.register_buffer('_weight', _weight)
        self._padding = (length - 1) // 2

    def forward(self, inputs):
        _device = inputs.device
        N, _, h, w = inputs.shape

        mask_h = inputs.new_zeros(N, h)
        mask_w = inputs.new_zeros(N, w)

        h_center = torch.randint(h, (N, 1), device=_device)
        w_center = torch.randint(w, (N, 1), device=_device)

        mask_h.scatter_(1, h_center, 1).unsqueeze_(1)
        mask_w.scatter_(1, w_center, 1).unsqueeze_(1)

        mask_h = F.conv1d(mask_h, self._weight, padding=self._padding)
        mask_w = F.conv1d(mask_w, self._weight, padding=self._padding)

        mask = 1. - torch.einsum('bci,bcj->bcij', mask_h, mask_w)
        outputs = inputs * mask
        return outputs


class SimclrAugment(nn.Module):
    def __init__(self, aug_type):
        super().__init__()
        if aug_type == "simclr_basic":
            self.pipeline = nn.Sequential(RandomResizeCropLayer(scale=(0.2, 1.0)), HorizontalFlipLayer(),
                                          RandomApply(ColorJitterLayer(ColorJitterLayer(0.4, 0.4, 0.4, 0.1)), p=0.8),
                                          RandomApply(RandomColorGrayLayer(), p=0.2))
        elif aug_type == "simclr_hq":
            self.pipeline = nn.Sequential(RandomResizeCropLayer(scale=(0.2, 1.0)), HorizontalFlipLayer(),
                                          RandomApply(ColorJitterLayer(0.4, 0.4, 0.4, 0.1), p=0.8),
                                          RandomApply(RandomColorGrayLayer(), p=0.2), RandomApply(GaussianBlur((0.1, 2.0)), p=0.5))
        elif aug_type == "simclr_hq_cutout":
            self.pipeline = nn.Sequential(RandomResizeCropLayer(scale=(0.2, 1.0)), HorizontalFlipLayer(),
                                          RandomApply(ColorJitterLayer(0.4, 0.4, 0.4, 0.1), p=0.8),
                                          RandomApply(RandomColorGrayLayer(), p=0.2), RandomApply(GaussianBlur((0.1, 2.0)), p=0.5),
                                          RandomApply(CutOut(15), p=0.5))
        elif aug_type == "byol":
            self.pipeline = nn.Sequential(RandomResizeCropLayer(scale=(0.2, 1.0)), HorizontalFlipLayer(),
                                          RandomApply(ColorJitterLayer(0.4, 0.4, 0.2, 0.1), p=0.8),
                                          RandomApply(RandomColorGrayLayer(), p=0.2), RandomApply(GaussianBlur((0.1, 2.0)), p=0.5))

    def forward(self, images):
        return self.pipeline(images)
