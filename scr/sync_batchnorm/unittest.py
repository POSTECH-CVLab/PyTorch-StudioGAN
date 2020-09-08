"""
-*- coding: utf-8 -*-
File   : unittest.py
Author : Jiayuan Mao
Email  : maojiayuan@gmail.com
Date   : 27/01/2018

This file is part of Synchronized-BatchNorm-PyTorch.
https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
Distributed under MIT License.

MIT License

Copyright (c) 2018 Jiayuan MAO

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


import unittest
import torch


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, x, y):
        adiff = float((x - y).abs().max())
        if (y == 0).all():
            rdiff = 'NaN'
        else:
            rdiff = float((adiff / y).abs().max())

        message = (
            'Tensor close check failed\n'
            'adiff={}\n'
            'rdiff={}\n'
        ).format(adiff, rdiff)
        self.assertTrue(torch.allclose(x, y), message)

