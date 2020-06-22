from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import scipy.optimize as opt
import numpy as np

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


methods = ['Nelder-Mead','Powell','COBYLA']


def lp_norm(x, xq, p):
    err = torch.mean(torch.abs(xq - x) ** p)
    return err


def mse(x, xq):
    err = torch.mean((xq - x) ** 2)
    return err


def tensor_range(x, pcq=False):
    if pcq:
        return x.view(x.shape[0], -1).max(dim=-1)[0] - x.view(x.shape[0], -1).min(dim=-1)[0]
    else:
        return x.max() - x.min()


def zero_point(x, pcq=False):
    if pcq:
        return x.view(x.shape[0], -1).min(dim=-1)[0]
    else:
        return x.min()


def quant_err(p, t, num_bits=4, metric='mse'):
    qp = QParams(range=t.new_tensor(p[0]), zero_point=t.new_tensor(p[1]), num_bits=num_bits)
    tq = quantize_with_grad(t, num_bits=qp.num_bits, qparams=qp)
    # TODO: Add other metrics
    return mse(t, tq).item()

def quant_round_constrain(t1, t2, trange, tzp):
    qp = QParams(range=t1.new_tensor(trange), zero_point=t1.new_tensor(tzp), num_bits=4)
    t1q = quantize_with_grad(t1, num_bits=qp.num_bits, qparams=qp, dequantize=False)
    t2q = quantize_with_grad(t2, num_bits=qp.num_bits, qparams=qp, dequantize=False)
    out=torch.max(torch.min(t2q,t1q+1),t1q-1)
    # TODO: Add other metrics
    return dequantize(out,num_bits=qp.num_bits, qparams=qp)

def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,  reduce_type='max', keepdim=False, true_zero=False,per_ch_input=False,quant_mode = 'maxmin'):
    alpha_gaus = {1:1.24,2:1.71,3:2.215,4:2.55,5:2.93,6:3.28,7:3.61,8:3.92}
    alpha_gaus_positive = {1:1.71,2:2.215,3:2.55,4:2.93,5:3.28,6:3.61,7:3.92,8:4.2}

    alpha_laplas = {1:1.05,2:1.86,3:2.83,4:5.03,5:6.2,6:7.41,7:8.64,8:9.89}
    alpha_laplas_positive = {1:1.86,2:2.83,3:5.03,4:6.2,5:7.41,6:8.64,7:9.89,8:11.16}
    if per_ch_input:
        x = x.transpose(0,1)
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if quant_mode =='mean_std' and num_bits<8: #If you want to apply only on the activation add "and reduce_dim is not None"
            mu   = x_flat.mean() if x_flat.dim() == 1 else x_flat.mean(-1)
            std  = x_flat.std() if x_flat.dim() == 1 else x_flat.std(-1)
            b = torch.abs(x_flat-mu).mean() if x_flat.dim() == 1 else torch.mean(torch.abs(x_flat-mu.unsqueeze(1)),-1)
            minv = x_flat.min() if x_flat.dim() == 1 else x_flat.min(-1)[0]
            maxv = x_flat.max() if x_flat.dim() == 1 else x_flat.max(-1)[0]
            #print((b-std).abs().max(),x.shape)
            ## Asic
            #import pdb; pdb.set_trace()
            #const = alpha_laplas_positive[num_bits] if reduce_dim is not None else alpha_laplas[num_bits] 
            #min_values = _deflatten_as(torch.max(mu - const*b,minv), x)  
            #max_values = _deflatten_as(torch.min(mu + const*b,maxv), x)
            ## Welling
            min_values = _deflatten_as(torch.max(mu - 6*std,minv), x)  
            max_values = _deflatten_as(torch.min(mu + 6*std,maxv), x)
        else:
            if x_flat.dim() == 1:
                min_values = _deflatten_as(x_flat.min(), x)
                max_values = _deflatten_as(x_flat.max(), x)
            else:
                min_values = _deflatten_as(x_flat.min(-1)[0], x)
                max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                if isinstance(reduce_dim, list) and len(reduce_dim)>1:
                    C=min_values.shape[-1]
                    min_values = min_values.view(-1).min(reduce_dim[0], keepdim=keepdim)[0]
                    max_values = max_values.view(-1).max(reduce_dim[0], keepdim=keepdim)[0]
                else:    
                    min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                    max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        # TODO: re-add true zero computation
        min_values[min_values > 0] = 0
        max_values[max_values < 0] = 0
        range_values = max_values - min_values
        range_values[range_values==0]=1
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=True, stochastic=False, inplace=False):

        ctx.inplace = inplace
        #if (num_bits is None and qparams.num_bits>4) or (num_bits is not None and num_bits>4 and input.dim()>2):
        #    import pdb; pdb.set_trace()                                                                           
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**num_bits - 1.
        running_range=qparams.range.clamp(min=1e-6,max=1e5)
        scale = running_range / (qmax - qmin)
        running_zero_point_round = Round().apply(qmin-zero_point/scale,False)
        zero_point = (qmin-running_zero_point_round.clamp(qmin,qmax))*scale    
        output.add_(qmin * scale - zero_point).div_(scale)
        if stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        # quantize
        output.clamp_(qmin, qmax).round_()
        if dequantize:
            output.mul_(scale).add_(
                zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None

class Round(InplaceFunction):

    @staticmethod
    def forward(ctx, input,inplace):

        ctx.inplace = inplace                                                                          
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        output.round_()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input,None



class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad, flatten_dims=(1, -1))
    return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach()
                    if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def quantize_with_grad(input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=True, stochastic=False, inplace=False):

    if inplace:
        output = input
    else:
        output = input.clone()
    if qparams is None:
        assert num_bits is not None, "either provide qparams of num_bits to quantize"
        qparams = calculate_qparams(
            input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)
    zero_point = qparams.zero_point
    num_bits = qparams.num_bits
    qmin = -(2.**(num_bits - 1)) if signed else 0.
    qmax = qmin + 2.**num_bits - 1.
    # ZP quantization for HW compliance
    running_range=qparams.range.clamp(min=1e-6,max=1e5)
    scale = running_range / (qmax - qmin)
    running_zero_point_round = Round().apply(qmin-zero_point/scale,False)
    zero_point = (qmin-running_zero_point_round.clamp(qmin,qmax))*scale
    output.add_(qmin * scale - zero_point).div_(scale)
    if stochastic:
        noise = output.new(output.shape).uniform_(-0.5, 0.5)
        output.add_(noise)
    # quantize
    output = Round().apply(output.clamp_(qmin, qmax),inplace)
    if dequantize:
        output.mul_(scale).add_(
            zero_point - qmin * scale)  # dequantize
    return output

def dequantize(input, num_bits=None, qparams=None,signed=False, inplace=False):
                                                                        
    if inplace:
        output = input
    else:
        output = input.clone()
    zero_point = qparams.zero_point
    num_bits = qparams.num_bits
    qmin = -(2.**(num_bits - 1)) if signed else 0.
    qmax = qmin + 2.**num_bits - 1.
    scale = qparams.range / (qmax - qmin)        
    output.mul_(scale).add_(
        zero_point - qmin * scale)  # dequantize
    return output

def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
    return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.1, measure=False,per_ch_input=False,reduce_dim=0, cal_qparams=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace
        self.num_bits = num_bits
        self.per_ch_input = per_ch_input
        self.reduce_dim = reduce_dim
        self.cal_qparams = cal_qparams

    def forward(self, input, qparams=None):

        if self.training or self.measure:
            if qparams is None:
                if self.cal_qparams:
                    init = np.array([tensor_range(input, pcq=False).item(), zero_point(input, pcq=False).item()])
                    res = opt.minimize(lambda p: quant_err(p, input, num_bits=self.num_bits, metric='mse'), init, method=methods[0])
                    qparams = QParams(range=input.new_tensor(res.x[0]), zero_point=input.new_tensor(res.x[1]), num_bits=self.num_bits)
                    print("Measure and optimize: bits - {}, error before - {:.6f}, error after {:.6f}".format(self.num_bits, quant_err(init, input), res.fun))
                else:
                    reduce_dim = None if self.per_ch_input else self.reduce_dim
                    if input.dim()==3 and reduce_dim == 0: reduce_dim = [self.reduce_dim,1]
                    qparams = calculate_qparams(input, num_bits=self.num_bits, flatten_dims=self.flatten_dims, reduce_dim=reduce_dim,per_ch_input=self.per_ch_input)

            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(
                    qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(
                    qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range,
                              zero_point=self.running_zero_point, num_bits=self.num_bits)
        if self.measure:
            return input
        else:
            if self.per_ch_input: input=input.transpose(0,1)
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               stochastic=self.stochastic, inplace=self.inplace)
            if self.per_ch_input: q_input=q_input.transpose(0,1)
            return q_input


class QuantThUpdate(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.1, measure=False,per_ch_input=False,reduce_dim=0):
        super(QuantThUpdate, self).__init__()
        self.running_zero_point = nn.Parameter(torch.ones(*shape_measure))
        self.running_range = nn.Parameter(torch.ones(*shape_measure))
        self.measure = measure
        self.flatten_dims = flatten_dims
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace
        self.num_bits = num_bits
        self.per_ch_input = per_ch_input
        self.reduce_dim = reduce_dim
        self.register_buffer('num_measured', torch.zeros(1))
        
    def forward(self, input, qparams=None):
        qparams = QParams(range=self.running_range,
                          zero_point=self.running_zero_point, num_bits=self.num_bits)
        
        if self.per_ch_input: input=input.transpose(0,1)
        q_input = quantize_with_grad(input, qparams=qparams, dequantize=self.dequantize,
                           stochastic=self.stochastic, inplace=self.inplace)
        if self.per_ch_input: q_input=q_input.transpose(0,1)
        return q_input



class QConv2dSamePadding(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=None, biprecision=False,measure=False):
        super(QConv2dSamePadding, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        #import pdb; pdb.set_trace()   
        if in_channels==groups:
            num_bits=8
            num_bits_weight=8
            per_ch_input = False
        else:                  
            per_ch_input = False             
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.measure = measure
        num_measure = in_channels if per_ch_input else 1
        self.quantize_input = QuantMeasure(
            self.num_bits, shape_measure=(num_measure, 1, 1, 1), flatten_dims=(1, -1), measure=measure,per_ch_input=per_ch_input)
        self.biprecision = biprecision
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, input):
        ih, iw = input.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            input = F.pad(input, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
   
        qweight = quantize(self.weight, qparams=weight_qparams) if not self.measure else self.weight

        if self.bias is not None:
            qbias = self.bias if self.measure else quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits,flatten_dims=(0, -1))
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad, flatten_dims=(1, -1))
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        return output

class QConv2d_o(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=None, biprecision=False,measure=False):
        super(QConv2d_o, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.measure = measure
        self.quantize_input = QuantMeasure(
            self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), measure=measure)
        self.biprecision = biprecision

    def forward(self, input):
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        
        qweight = quantize(self.weight, qparams=weight_qparams) if not self.measure else self.weight

        if self.bias is not None:
            qbias = self.bias if self.measure else quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits,flatten_dims=(0, -1))
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad, flatten_dims=(1, -1))
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        return output


class QConv2d_lapq(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=None, biprecision=False,measure=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.measure = measure
        self.quantize_input = QuantMeasure(
            self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), measure=measure)
        self.quantize_weight = QuantMeasure(
            self.num_bits, shape_measure=(out_channels, 1, 1, 1), flatten_dims=(1, -1), measure=measure, reduce_dim=None)
        self.biprecision = biprecision

    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = self.quantize_weight(self.weight)
        if self.bias is not None:
            qbias = self.bias if self.measure else quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits,flatten_dims=(0, -1))
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad, flatten_dims=(1, -1))
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        return output

class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=None, perC=True, biprecision=False, measure=False, cal_qparams=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.measure = measure
        self.equ_scale = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        if measure:
            self.quantize_input = QuantMeasure(
                self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), measure=measure, cal_qparams=cal_qparams)
            self.quantize_weight = QuantMeasure(
                self.num_bits, shape_measure=(out_channels if perC else 1, 1, 1, 1), flatten_dims=(1,-1) if perC else (0,-1), measure=measure, reduce_dim=None if perC else 0)
        else:
            self.quantize_input = QuantThUpdate(
                self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), measure=measure)
            self.quantize_weight = QuantThUpdate(
                self.num_bits, shape_measure=(out_channels if perC else 1, 1, 1, 1), flatten_dims=(1,-1) if perC else (0,-1), measure=measure, reduce_dim=None if perC else 0)
        self.biprecision = biprecision
        self.cal_params = cal_qparams
        self.quantize = True

    def forward(self, input):
        qinput = self.quantize_input(input) if self.quantize else input
        qweight = self.quantize_weight(self.weight * self.equ_scale) if self.quantize and not self.cal_params else self.weight

        if self.bias is not None:
            qbias = self.bias if (self.measure or not self.quantize) else quantize(self.bias, num_bits=self.num_bits_weight + self.num_bits,flatten_dims=(0, -1))
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad, flatten_dims=(1, -1))
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        return output


class QSigmoid(nn.Sigmoid):
    """docstring for QSigmoid."""

    def __init__(self, num_bits=8, measure=False):
        super(QSigmoid, self).__init__()
        self.num_bits = num_bits
        self.measure = measure
        self.quantize_input = QuantMeasure(
            self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), measure=measure)

    def forward(self, input):
        qinput = self.quantize_input(input)
        output = torch.sigmoid(qinput)
        return output

class QSwish(nn.Module):
    def __init__(self,num_bits=8, measure=False):
        super(QSwish, self).__init__()
        self.num_bits=num_bits
        self.measure=measure
        self.qsigmoid=QSigmoid(num_bits,measure)
        self.quantize_input = QuantMeasure(
            self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), measure=measure)  
    def forward(self, input1,input2=None):
        if input2 is None:
            input2=input1
        output = self.quantize_input(input1) * self.qsigmoid(input2)
        return output

class QLinear_o(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=None, biprecision=False,measure=False):
        super(QLinear_o, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits,measure=measure)
        self.measure = measure

    def forward(self, input):
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams) if not self.measure else self.weight
        if self.bias is not None:
            qbias = self.bias if self.measure else quantize(
                self.bias, num_bits=self.num_bits_weight + self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output

class QLinear_lapq(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=None, biprecision=False,measure=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits,measure=measure)
        self.quantize_weight = QuantMeasure(self.num_bits,shape_measure=(out_features, 1), measure=measure,reduce_dim=None)
        self.measure = measure
        
    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = self.quantize_weight(self.weight)

        if self.bias is not None:
            qbias = self.bias if self.measure else quantize(
                self.bias, num_bits=self.num_bits_weight + self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output

class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=None, perC=True, biprecision=False,measure=False, cal_qparams=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.equ_scale = nn.Parameter(torch.ones(out_features, 1))
        if measure:
            self.quantize_input = QuantMeasure(self.num_bits,measure=measure, cal_qparams=cal_qparams)
            self.quantize_weight = QuantMeasure(self.num_bits,shape_measure=(out_features if perC else 1, 1), flatten_dims=(1,-1) if perC else (0,-1), measure=measure,reduce_dim=None if perC else 0)
        else:
            self.quantize_input = QuantThUpdate(self.num_bits,measure=measure)
            self.quantize_weight = QuantThUpdate(self.num_bits,shape_measure=(out_features if perC else 1, 1), flatten_dims=(1,-1) if perC else (0,-1), measure=measure,reduce_dim=None if perC else 0)
        self.measure = measure
        self.cal_params = cal_qparams
        self.quantize = True
        #import pdb; pdb.set_trace()
        
    def forward(self, input):
        #import pdb; pdb.set_trace()
        qinput = self.quantize_input(input) if self.quantize else input
        qweight = self.quantize_weight(self.weight * self.equ_scale) if self.quantize and not self.cal_params else self.weight
        
        if self.bias is not None:
            qbias = self.bias if (self.measure or not self.quantize) else quantize(
                self.bias, num_bits=self.num_bits_weight + self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output

class QMatmul(nn.Module):
    """docstring for QConv2d."""

    def __init__(self, num_bits=8, num_bits_weight=8, num_bits_grad=None, perC=True, biprecision=False,measure=False, cal_qparams=False):
        super(QMatmul, self).__init__()
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        if measure:
            self.quantize_input1 = QuantMeasure(self.num_bits,measure=measure, cal_qparams=cal_qparams,reduce_dim=[0,1,2])
            self.quantize_input2 = QuantMeasure(self.num_bits,measure=measure, cal_qparams=cal_qparams,reduce_dim=[0,1,2])
        else:
            self.quantize_input1 = QuantThUpdate(self.num_bits,measure=measure)
            self.quantize_input2 = QuantThUpdate(self.num_bits,measure=measure)
        self.measure = measure
        self.cal_params = cal_qparams
        self.quantize = True

    def forward(self, input1, input2):
        qinput1 = self.quantize_input1(input1) if self.quantize else input1
        qinput2 = self.quantize_input2(input2) if self.quantize else input2
        
        output = torch.matmul(qinput1, qinput2)

        return output

class QEmbedding(nn.Embedding):
    """docstring for QConv2d."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                num_bits=8, num_bits_weight=8, num_bits_grad=None, perC=True, biprecision=False,measure=False, cal_qparams=False):
        super(QEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        if measure:
            self.quantize_weight = QuantMeasure(self.num_bits_weight,shape_measure=(1, 1), measure=measure,reduce_dim=0)
        else:
            self.quantize_weight = QuantThUpdate(self.num_bits_weight,shape_measure=(1, 1), measure=measure,reduce_dim=0)
        self.measure = measure
        self.cal_params = cal_qparams
        self.quantize = True

    def forward(self, input):
        #import pdb; pdb.set_trace()
        qweight = self.quantize_weight(self.weight) if self.quantize else self.weight
        output = F.embedding(input, qweight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return output

class RangeBN(nn.Module):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(
            self.num_bits, inplace=True, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        x = self.quantize_input(x)
        if x.dim() == 2:  # 1d
            x = x.unsqueeze(-1,).unsqueeze(-1)

        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()  # C x B x H x W
            y = y.view(C, self.num_chunks, (B * H * W) // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)  # C
            mean_min = y.min(-1)[0].mean(-1)  # C
            mean = y.view(C, -1).mean(-1)  # C
            scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
                                        0.5) / ((2 * math.log(y.size(-1))) ** 0.5)

            scale = (mean_max - mean_min) * scale_fix
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(
                    mean * (1 - self.momentum))

                self.running_var.mul_(self.momentum).add_(
                    scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        # scale = quantize(scale, num_bits=self.num_bits, min_value=float(
        #     scale.min()), max_value=float(scale.max()))
        out = (x - mean.view(1, -1, 1, 1)) / \
            (scale.view(1, -1, 1, 1) + self.eps)

        if self.weight is not None:
            qweight = self.weight
            # qweight = quantize(self.weight, num_bits=self.num_bits,
            #                    min_value=float(self.weight.min()),
            #                    max_value=float(self.weight.max()))
            out = out * qweight.view(1, -1, 1, 1)

        if self.bias is not None:
            qbias = self.bias
            # qbias = quantize(self.bias, num_bits=self.num_bits)
            out = out + qbias.view(1, -1, 1, 1)
        if self.num_bits_grad is not None:
            out = quantize_grad(
                out, num_bits=self.num_bits_grad, flatten_dims=(1, -1))

        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out


class RangeBN1d(RangeBN):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeBN1d, self).__init__(num_features, dim, momentum,
                                        affine, num_chunks, eps, num_bits, num_bits_grad)
        self.quantize_input = QuantMeasure(
            self.num_bits, inplace=True, shape_measure=(1, 1), flatten_dims=(1, -1))

if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
