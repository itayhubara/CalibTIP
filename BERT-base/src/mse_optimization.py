import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import scipy.optimize as opt
import torch.nn.functional as F
import warnings
import numpy as np
from transformers.modeling_quantize import methods,QParams
from tqdm import tqdm
import pandas as pd
import math


def optimize_rounding(layer, cached_inps, cached_outs, test_inp, test_out, iters=7000, batch_size=100):
    print("Optimize rounding")
    ada_quantizer = AdaptiveRoundingQuantizer(layer.weight,
                                              x_min=layer.quantize_weight.running_zero_point,
                                              x_max=layer.quantize_weight.running_zero_point + layer.quantize_weight.running_range,
                                              n_bits=layer.quantize_weight.num_bits,
                                              round_mode='learned_hard_sigmoid')

    # ada_quantizer.init_alpha(layer.weight)
    opt_params = [ada_quantizer.alpha]
    optimizer = torch.optim.Adam(opt_params)

    with torch.no_grad():
        mse_before = F.mse_loss(layer(test_inp), test_out)
        qforward_orig = layer.quantize_weight.forward

        layer.quantize_weight.forward = ada_quantizer.forward

        # out_hard_quant = layer(test_inp)
        ada_quantizer.soft_targets = True
        # out_soft_quant = layer(test_inp)

        # soft_quant_rec = F.mse_loss(out_soft_quant, test_out)
        # print('Reconstruction error before optimization (soft quant):\t', float(soft_quant_rec))
        # hard_quant_rec = F.mse_loss(out_hard_quant, test_out)
        # print('Reconstruction error before optimization (hard quant):\t', float(hard_quant_rec))

    iters = iters
    loss_func = CombinedLoss(ada_quantizer, weight=0.01, max_count=iters, b_range=(20, 2),
                                     decay_type='cosine', warmup=0.2)

    for p in layer.parameters():
        p.requires_grad = False

    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]

        cur_inp = cached_inps[idx]
        cur_out = cached_outs[idx]

        optimizer.zero_grad()
        out_quant = layer(cur_inp)

        loss = loss_func(out_quant, cur_out)

        loss.backward(retain_graph=True)
        optimizer.step()

    with torch.no_grad():
        # out_soft_quant = layer(test_inp)
        ada_quantizer.soft_targets = False
        # out_hard_quant = layer(test_inp)

        # soft_quant_rec = F.mse_loss(out_soft_quant, test_out)
        # print('Reconstruction error after optimization (soft quant):\t', float(soft_quant_rec))
        # hard_quant_rec = F.mse_loss(out_hard_quant, test_out)
        # print('Reconstruction error after optimization (hard quant):\t', float(hard_quant_rec))

        qweight = ada_quantizer.forward(layer.weight)
        layer.weight.data = qweight.view(layer.weight.shape)

        layer.quantize_weight.forward = qforward_orig
        yq = layer(test_inp)
        mse_after = F.mse_loss(yq, test_out)
        print('MSE before adaround:\t', mse_before.item())
        print('MSE after adaround:\t', mse_after.item())


def optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out, batch_size=100):
    print("\nOptimize quantization params")
    w_range_orig = layer.quantize_weight.running_range.data.clone()
    w_zp_orig = layer.quantize_weight.running_zero_point.data.clone()
    inp_range_orig = layer.quantize_input.running_range.data.clone()
    inp_zp_orig = layer.quantize_input.running_zero_point.data.clone()

    def layer_err(p, inp, out):
        layer.quantize_weight.running_range.data = w_range_orig * p[0]
        layer.quantize_weight.running_zero_point.data = w_zp_orig + p[1]
        layer.quantize_input.running_range.data = inp_range_orig * p[2]
        layer.quantize_input.running_zero_point.data = inp_zp_orig + p[3]
        yq = layer(inp)
        return F.mse_loss(yq, out).item()

    init = np.array([1, 0, 1, 0])
    results = []
    for i in tqdm(range(int(cached_inps.size(0) / batch_size))):
        cur_inp = cached_inps[i * batch_size:(i + 1) * batch_size]
        cur_out = cached_outs[i * batch_size:(i + 1) * batch_size]

        # print("init:")
        # print(init)
        res = opt.minimize(lambda p: layer_err(p, cur_inp, cur_out), init, method=methods[0])
        results.append(res.x)

    mean_res = np.array(results).mean(axis=0)
    print(mean_res)
    mse_before = layer_err(init, test_inp, test_out)
    mse_after = layer_err(mean_res, test_inp, test_out)
    return mse_before, mse_after

# def optimize_qparams_embedding(layer, cached_inps, cached_outs, test_inp, test_out, batch_size=100):
#     print("\nOptimize quantization params")
#     w_range_orig = layer.quantize_weight.running_range.data.clone()
#     w_zp_orig = layer.quantize_weight.running_zero_point.data.clone()
#
#     def layer_err(p):
#         layer.quantize_weight.running_range.data = w_range_orig * p[0]
#         layer.quantize_weight.running_zero_point.data = w_zp_orig + p[1]
#         #import pdb; pdb.set_trace()
#         wq = layer.quantize_weight(layer.weight)
#         return F.mse_loss(wq, layer.weight).item()
#
#     init = np.array([1, 0])
#
#     res = opt.minimize(lambda p: layer_err(p), init, method=methods[0])
#     print(res.x)
#     mse_before = layer_err(init)
#     mse_after = layer_err(res.x)
#     return mse_before, mse_after

def optimize_qparams_embedding(layer, cached_inps, cached_outs, test_inp, test_out, batch_size=100):
    print("\nOptimize quantization params")
    w_range_orig = layer.quantize_weight.running_range.data.clone()
    w_zp_orig = layer.quantize_weight.running_zero_point.data.clone()

    def layer_err(p, inp, out):
        layer.quantize_weight.running_range.data = w_range_orig * p[0]
        layer.quantize_weight.running_zero_point.data = w_zp_orig + p[1]
        #import pdb; pdb.set_trace()
        yq = layer(inp)
        return F.mse_loss(yq, out).item()

    init = np.array([1, 0])
    results = []
    for i in tqdm(range(int(cached_inps.size(0) / batch_size))):
        cur_inp = cached_inps[i * batch_size:(i + 1) * batch_size]
        cur_out = cached_outs[i * batch_size:(i + 1) * batch_size]

        # print("init:")
        # print(init)
        res = opt.minimize(lambda p: layer_err(p, cur_inp, cur_out), init, method=methods[1])
        results.append(res.x)

    mean_res = np.array(results).mean(axis=0)
    print(mean_res)
    mse_before = layer_err(init, test_inp, test_out)
    mse_after = layer_err(mean_res, test_inp, test_out)
    return mse_before, mse_after


def adaquant_embedding(layer, cached_inps, cached_outs, test_inp, test_out, lr_in=1e-2, iters=100, progress=True, batch_size=50):
    print("\nRun adaquant")
    mse_before = F.mse_loss(layer(test_inp), test_out)

    opt = torch.optim.Adam([layer.quantize_weight.running_range,
                            layer.quantize_weight.running_zero_point], lr=lr_in)

    losses = []
    for j in (tqdm(range(iters)) if progress else range(iters)):
        idx = torch.randperm(cached_outs.size(0))[:batch_size]

        train_inp = cached_inps[idx]#.cuda()
        train_out = cached_outs[idx]#.cuda()

        qout = layer(train_inp)
        loss = F.mse_loss(qout, train_out)

        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

            # if len(losses) < 10:
            #     total_loss = loss.item()
            # else:
            #     total_loss = np.mean(losses[-10:])
            # print("mse out: {}, pc mean loss: {}, total: {}".format(mse_out.item(), mean_loss.item(), total_loss))

    mse_after = F.mse_loss(layer(test_inp), test_out)
    return mse_before.item(), mse_after.item()

def optimize_qparams_matmul(layer, cached_inps, cached_outs, test_inp, test_out, batch_size=100):
    #import pdb; pdb.set_trace()
    print("\nOptimize quantization params")
    inp1_range_orig = layer.quantize_input1.running_range.data.clone()
    inp1_zp_orig = layer.quantize_input1.running_zero_point.data.clone()
    inp2_range_orig = layer.quantize_input2.running_range.data.clone()
    inp2_zp_orig = layer.quantize_input2.running_zero_point.data.clone()

    def layer_err(p, inp1,inp2, out):
        layer.quantize_input1.running_range.data = inp1_range_orig * p[0]
        layer.quantize_input1.running_zero_point.data = inp1_zp_orig + p[1]
        layer.quantize_input2.running_range.data = inp2_range_orig * p[2]
        layer.quantize_input2.running_zero_point.data = inp2_zp_orig + p[3]
        yq = layer(inp1,inp2)
        return F.mse_loss(yq, out).item()

    init = np.array([1, 0, 1, 0])
    results = []
    for i in tqdm(range(int(cached_outs.size(0) / batch_size))):
        #import pdb; pdb.set_trace()
        cur_inp1 = cached_inps[0][i * batch_size:(i + 1) * batch_size]
        cur_inp2 = cached_inps[1][i * batch_size:(i + 1) * batch_size]
        cur_out = cached_outs[i * batch_size:(i + 1) * batch_size]

        # print("init:")
        # print(init)
        res = opt.minimize(lambda p: layer_err(p, cur_inp1, cur_inp2, cur_out), init, method=methods[2])
        results.append(res.x)

    mean_res = np.array(results).mean(axis=0)
    print(mean_res)
    mse_before = layer_err(init, test_inp[0],test_inp[1], test_out)
    mse_after = layer_err(mean_res, test_inp[0],test_inp[1], test_out)
    return mse_before, mse_after


def adaquant_matmul(layer, cached_inps, cached_outs, test_inp, test_out, lr_in1=1e-1, lr_in2=1e-1, iters=100, progress=True, batch_size=50):
    print("\nRun adaquant")
    mse_before = F.mse_loss(layer(test_inp[0], test_inp[1]), test_out)

    opt1 = torch.optim.Adam([layer.quantize_input1.running_range,
                                       layer.quantize_input1.running_zero_point], lr=lr_in1)
    opt2 = torch.optim.Adam([layer.quantize_input2.running_range,
                                      layer.quantize_input2.running_zero_point], lr=lr_in2)

    losses = []
    for j in (tqdm(range(iters)) if progress else range(iters)):
        idx = torch.randperm(cached_outs.size(0))[:batch_size]

        train_inp1 = cached_inps[0][idx]#.cuda()
        train_inp2 = cached_inps[1][idx]  # .cuda()
        train_out = cached_outs[idx]#.cuda()

        qout = layer(train_inp1, train_inp2)
        loss = F.mse_loss(qout, train_out)

        losses.append(loss.item())
        opt1.zero_grad()
        opt2.zero_grad()
        loss.backward()
        opt1.step()
        opt2.step()

            # if len(losses) < 10:
            #     total_loss = loss.item()
            # else:
            #     total_loss = np.mean(losses[-10:])
            # print("mse out: {}, pc mean loss: {}, total: {}".format(mse_out.item(), mean_loss.item(), total_loss))

    mse_after = F.mse_loss(layer(test_inp[0], test_inp[1]), test_out)
    return mse_before.item(), mse_after.item()


def adaquant(layer, cached_inps, cached_outs, test_inp, test_out, lr_in=1e-1, lr_qpw=1e-3, lr_w=1e-5, lr_b=1e-3,
             iters=100, progress=True, batch_size=50):
    print("\nRun adaquant")
    mse_before = F.mse_loss(layer(test_inp), test_out)

    opt_w = torch.optim.Adam([layer.weight], lr=lr_w)
    if hasattr(layer, 'bias') and layer.bias is not None: opt_bias = torch.optim.Adam([layer.bias], lr=lr_b)
    opt_qparams_in = torch.optim.Adam([layer.quantize_input.running_range,
                                       layer.quantize_input.running_zero_point], lr=lr_in)
    opt_qparams_w = torch.optim.Adam([layer.quantize_weight.running_range,
                                      layer.quantize_weight.running_zero_point], lr=lr_qpw)

    losses = []
    for j in (tqdm(range(iters)) if progress else range(iters)):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]

        train_inp = cached_inps[idx]#.cuda()
        train_out = cached_outs[idx]#.cuda()

        qout = layer(train_inp)
        loss = F.mse_loss(qout, train_out)

        losses.append(loss.item())
        opt_w.zero_grad()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.zero_grad()
        opt_qparams_in.zero_grad()
        opt_qparams_w.zero_grad()
        loss.backward()
        opt_w.step()
        if hasattr(layer, 'bias') and layer.bias is not None: opt_bias.step()
        opt_qparams_in.step()
        opt_qparams_w.step()

            # if len(losses) < 10:
            #     total_loss = loss.item()
            # else:
            #     total_loss = np.mean(losses[-10:])
            # print("mse out: {}, pc mean loss: {}, total: {}".format(mse_out.item(), mean_loss.item(), total_loss))

    mse_after = F.mse_loss(layer(test_inp), test_out)
    return mse_before.item(), mse_after.item()


def optimize_layer(layer, in_out, optimize_weights=False, gradient_based_optimization=True,
                   batch_size=12, is_weight=True, is_embedding=False):

    if is_weight or is_embedding:
        cached_inps = torch.cat([x[0][0] for x in in_out])
    else:
        cached_inps = [torch.cat([x[0][0] for x in in_out]),torch.cat([x[0][1] for x in in_out])]
    cached_outs = torch.cat([x[1] for x in in_out])
    cached_inps = cached_inps.to(layer.weight.device) if is_weight or is_embedding else [cached_inps[0].to(layer.quantize_input1.running_range.device),cached_inps[1].to(layer.quantize_input1.running_range.device)]
    cached_outs = cached_outs.to(layer.weight.device) if is_weight or is_embedding else cached_outs.to(layer.quantize_input1.running_range.device)

    idx = torch.randperm(cached_outs.size(0))[:batch_size]

    test_inp = cached_inps[idx] if is_weight or is_embedding else [cached_inps[0][idx] , cached_inps[1][idx]]
    test_out = cached_outs[idx]

    if gradient_based_optimization:
        if is_weight and not is_embedding:
            lr_w = 1e-5 if optimize_weights else 0.
            lr_b = 1e-3 if optimize_weights else 0.
            mse_before, mse_after = adaquant(layer, cached_inps, cached_outs, test_inp, test_out, iters=50,
                                             lr_in=1e-1, lr_qpw=1e-3, lr_w=lr_w, lr_b=lr_b, batch_size=batch_size)
        elif is_embedding:
            mse_before, mse_after = adaquant_embedding(layer, cached_inps, cached_outs, test_inp, test_out, iters=100,
                                                       lr_in=1e-2, batch_size=batch_size)
        else:
            mse_before, mse_after = adaquant_matmul(layer, cached_inps, cached_outs, test_inp, test_out, iters=50,
                                                    lr_in1=1e-1, lr_in2=1e-1, batch_size=batch_size)
    else:
        if is_weight and not is_embedding:
            mse_before, mse_after = optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out,
                                                     batch_size=batch_size)
        elif is_embedding:
            mse_before, mse_after = optimize_qparams_embedding(layer, cached_inps, cached_outs, test_inp, test_out,
                                                               batch_size=batch_size)
        else:
            mse_before, mse_after = optimize_qparams_matmul(layer, cached_inps, cached_outs, test_inp, test_out,
                                                            batch_size=batch_size)

    with torch.no_grad():
        N = test_out.numel()
        snr_before = (1/math.sqrt(N)) * math.sqrt(N * mse_before) / torch.norm(test_out).item()
        snr_after = (1/math.sqrt(N)) * math.sqrt(N * mse_after) / torch.norm(test_out).item()

    #import pdb; pdb.set_trace()
    if is_embedding:
        kurt_in = None
    else:
        kurt_in = kurtosis(test_inp).item() if is_weight else 0.5*(kurtosis(test_inp[0]).item()+kurtosis(test_inp[1]).item())
    kurt_w = kurtosis(layer.weight).item() if is_weight or is_embedding  else None
    return mse_before, mse_after, snr_before, snr_after, kurt_in, kurt_w

def kurtosis(x):
    var = torch.mean((x - x.mean())**2)
    return torch.mean((x - x.mean())**4 / var**2)


class TempDecay:
    def __init__(self, t_max, rel_start_decay=0.2, start_b=10, end_b=2, type='linear'):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b
        self.type = type

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            if self.type == 'linear':
                return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
            elif self.type == 'cosine':
                return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))
            elif self.type == 'power4':
                return self.end_b + (self.start_b - self.end_b) * (1 - rel_t ** 4)
            elif self.type == 'power6':
                return self.end_b + (self.start_b - self.end_b) * (1 - rel_t ** 6)
            elif self.type == 'power8':
                return self.end_b + (self.start_b - self.end_b) * (1 - rel_t ** 8)
            else:
                raise ValueError('Unknown temperature decay type {}'.format(self.type))


class CombinedLoss:
    def __init__(self, quantizer, round_loss='relaxation', weight=1, rec_loss='mse', max_count=2000,
                 b_range=(10, 2), decay_type='linear', decay_start=0.0, warmup=0.0):
        self.quantizer = quantizer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup

        self.temp_decay = TempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                    start_b=b_range[0], end_b=b_range[1], type=decay_type)
        self.count = 0

    def __call__(self, pred, tgt):
        self.count += 1
        reg_loss = F.mse_loss(pred, tgt, reduction='none').sum(1).mean()

        if self.count < self.loss_start:
            b = round_loss = 0
        elif self.round_loss == 'temp_decay':
            b = self.temp_decay(self.count)
            self.quantizer.temperature = b
            round_loss = 0
        elif self.round_loss == 'relaxation':
            # 1 - ((.5-x)*2)**b
            b = self.temp_decay(self.count)
            round_vals = self.quantizer.get_rest().view(-1)
            round_loss = self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        elif self.round_loss is None:
            b = round_loss = 0
        else:
            raise NotImplementedError()

        total_loss = reg_loss + round_loss
        if self.count % 100 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(reg_loss), float(round_loss), b, self.count))
        return total_loss


class AdaptiveRoundingQuantizer(nn.Module):
    def __init__(self, x, x_min=None, x_max=None, n_bits=8, symmetric=False, round_mode='nearest',
                 binary_opt=None, init='minmax', temperature=1.0):
        super(AdaptiveRoundingQuantizer, self).__init__()

        self.n_bits = n_bits
        self.symmetric = symmetric
        self.round_mode = round_mode
        self.binary_opt = binary_opt
        self.init = init

        self.delta = None
        self.zero_point = None
        self.soft_targets = False
        self.temperature = temperature

        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2./3

        if x_min is not None and x_max is not None:
            self.set_quant_params(x_min, x_max)

        x_floor = torch.floor((x - self.zero_point) / self.delta)
        rest = ((x - self.zero_point) / self.delta) - x_floor  # rest of rounding [0, 1)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
        self.alpha = nn.Parameter(alpha)

    # def init_alpha(self, x):
    #     x_floor = torch.floor(x / self.delta)
    #     rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
    #     alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
    #     self.alpha = nn.Parameter(alpha)

    def set_quant_params(self, x_min, x_max):
        # Ensure we always use zero
        # x_min = min(x_min, 0)
        # x_max = max(x_max, 0)

        self.delta = (x_max - x_min) / (2 ** self.n_bits - 1)
        self.delta = torch.max(self.delta, self.delta.new_tensor([1e-8]))

        # self.zero_point = torch.round(-x_min / self.delta)
        self.zero_point = x_min

    def get_rest(self):
        if self.round_mode == 'sigmoid_temp_decay' and self.alpha is not None:
            return torch.sigmoid(self.alpha / self.temperature)
        if self.round_mode == 'learned_sigmoid' and self.alpha is not None:
            return torch.sigmoid(self.alpha)
        if self.round_mode == 'learned_hard_sigmoid' and self.alpha is not None:
            return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
        if self.round_mode == 'external':
            return self.binary_opt if self.binary_opt is not None else torch.zeros(1)
        else:
            Warning('Get rest is not implemented for rounding mode {}'.format(self.round_mode))
            return torch.zeros(1)

    def forward(self, x, qparams=None):
        if self.round_mode == 'nearest':
            x_int = torch.round((x - self.zero_point) / self.delta)
        else:
            x_floor = torch.floor((x - self.zero_point) / self.delta)

            if self.soft_targets:
                x_int = x_floor + self.get_rest()
            else:
                x_int = x_floor + (self.alpha >= 0).float()

        x_quant = torch.clamp(x_int, 0, 2 ** self.n_bits - 1)
        x_float_q = x_quant * self.delta + self.zero_point

        return x_float_q

    # def forward(self, x, qparams=None):
    #     x_floor = torch.floor(x / self.delta)
    #
    #     if self.soft_targets:
    #         x_int = x_floor + self.get_rest()
    #     else:
    #         x_int = x_floor + (self.alpha >= 0).float()
    #
    #     x_quant = torch.clamp(x_int + self.zero_point, 0, 2 ** self.n_bits - 1)
    #     x_float_q = (x_quant - self.zero_point) * self.delta
    #
    #     return x_float_q
