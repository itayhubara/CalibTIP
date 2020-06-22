import torch
import torch.nn as nn
import logging
# from efficientnet_pytorch.utils import Conv2dSamePadding

def remove_bn_params(bn_module):
    bn_module.register_buffer('running_mean', None)
    bn_module.register_buffer('running_var', None)
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)


def init_bn_params(bn_module):
    bn_module.running_mean.fill_(0)
    bn_module.running_var.fill_(1)
    if bn_module.affine:
        bn_module.weight.fill_(1)
        bn_module.bias.fill_(0)


def absorb_bn(module, bn_module, remove_bn=True, verbose=False):
    with torch.no_grad():
        w = module.weight
        if module.bias is None:
            zeros = torch.zeros(module.out_channels,
                                dtype=w.dtype, device=w.device)
            bias = nn.Parameter(zeros)
            module.register_parameter('bias', bias)
        b = module.bias

        if hasattr(bn_module, 'running_mean'):
            b.add_(-bn_module.running_mean)
        if hasattr(bn_module, 'running_var'):
            invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
            w.mul_(invstd.view(w.size(0), 1, 1, 1))
            b.mul_(invstd)
            if hasattr(module, 'quantize_weight'):
                module.quantize_weight.running_range.mul_(invstd.view(w.size(0), 1, 1, 1))
                module.quantize_weight.running_zero_point.mul_(invstd.view(w.size(0), 1, 1, 1))

        if hasattr(bn_module, 'weight'):
            w.mul_(bn_module.weight.view(w.size(0), 1, 1, 1))
            b.mul_(bn_module.weight)
            module.register_parameter('gamma', nn.Parameter(bn_module.weight.data.clone()))
            if hasattr(module, 'quantize_weight'):
                module.quantize_weight.running_range.mul_(bn_module.weight.view(w.size(0), 1, 1, 1))
                module.quantize_weight.running_zero_point.mul_(bn_module.weight.view(w.size(0), 1, 1, 1))
        if hasattr(bn_module, 'bias'):
            b.add_(bn_module.bias)
            module.register_parameter('beta', nn.Parameter(bn_module.bias.data.clone()))

        if remove_bn:
            remove_bn_params(bn_module)
        else:
            init_bn_params(bn_module)

        if verbose:
            logging.info('BN module %s was asborbed into layer %s' %
                         (bn_module, module))


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, Conv2dSamePadding)


def search_absorbe_bn(model, prev=None, remove_bn=True, verbose=False):
    with torch.no_grad():
        for m in model.children():
            if is_bn(m) and is_absorbing(prev):
                # print(prev,m)
                absorb_bn(prev, m, remove_bn=remove_bn, verbose=verbose)
            search_absorbe_bn(m, remove_bn=remove_bn, verbose=verbose)
            prev = m


def absorb_fake_bn(module, bn_module, verbose=False):
    with torch.no_grad():
        w = module.weight
        if module.bias is None:
            zeros = torch.zeros(module.out_channels,
                                dtype=w.dtype, device=w.device)
            bias = nn.Parameter(zeros)
            module.register_parameter('bias', bias)

        if verbose:
            logging.info('BN module %s was asborbed into layer %s' %
                         (bn_module, module))


def is_fake_bn(m):
    from models.resnet import Lambda
    return isinstance(m, Lambda)


def search_absorbe_fake_bn(model, prev=None, remove_bn=True, verbose=False):
    with torch.no_grad():
        for m in model.children():
            if is_fake_bn(m) and is_absorbing(prev):
                # print(prev,m)
                absorb_fake_bn(prev, m, verbose=verbose)
            search_absorbe_fake_bn(m, remove_bn=remove_bn, verbose=verbose)
            prev = m


def add_bn(module, bn_module, verbose=False):
    bn = nn.BatchNorm2d(module.out_channels)

    def bn_forward(bn, x):
        res = bn(x)
        return res

    bn_module.forward_orig = bn_module.forward
    bn_module.forward = lambda x: bn_forward(bn, x)
    bn.to(module.weight.device)

    bn.register_buffer('running_var', module.gamma**2)
    bn.register_buffer('running_mean', module.beta.clone())
    bn.register_parameter('weight', nn.Parameter(torch.sqrt(bn.running_var + bn.eps)))
    bn.register_parameter('bias', nn.Parameter(bn.running_mean.clone()))

    bn_module.bn = bn


def need_tuning(module):
    return hasattr(module, 'num_bits') #and module.groups == 1


def search_add_bn(model, prev=None, remove_bn=True, verbose=False):
    with torch.no_grad():
        for m in model.children():
            if is_fake_bn(m) and is_absorbing(prev) and need_tuning(prev):
                # print(prev,m)
                add_bn(prev, m, verbose=verbose)
            search_add_bn(m, remove_bn=remove_bn, verbose=verbose)
            prev = m


def search_absorbe_tuning_bn(model, prev=None, remove_bn=True, verbose=False):
    with torch.no_grad():
        for m in model.children():
            if is_fake_bn(m) and is_absorbing(prev) and need_tuning(prev):
                # print(prev,m)
                absorb_bn(prev, m.bn, remove_bn=remove_bn, verbose=verbose)
                m.forward = m.forward_orig
                m.bn = None
            search_absorbe_tuning_bn(m, remove_bn=remove_bn, verbose=verbose)
            prev = m


def copy_bn_params(module, bn_module, remove_bn=True, verbose=False):
    with torch.no_grad():
        if hasattr(bn_module, 'weight'):
            module.register_parameter('gamma', nn.Parameter(bn_module.weight.data.clone()))

        if hasattr(bn_module, 'bias'):
            module.register_parameter('beta', nn.Parameter(bn_module.bias.data.clone()))


def search_copy_bn_params(model, prev=None, remove_bn=True, verbose=False):
    with torch.no_grad():
        for m in model.children():
            if is_bn(m) and is_absorbing(prev):
                # print(prev,m)
                copy_bn_params(prev, m, remove_bn=remove_bn, verbose=verbose)
            search_copy_bn_params(m, remove_bn=remove_bn, verbose=verbose)
            prev = m


# def recalibrate_bn(module, bn_module, verbose=False):
#     bn = bn_module.bn
#     bn.register_parameter('weight', nn.Parameter(torch.sqrt(bn.running_var + bn.eps)))
#     bn.register_parameter('bias', nn.Parameter(bn.running_mean.clone()))
#
#
# def search_bn_recalibrate(model, prev=None, remove_bn=True, verbose=False):
#     with torch.no_grad():
#         for m in model.children():
#             if is_fake_bn(m) and is_absorbing(prev) and need_tuning(prev):
#                 recalibrate_bn(prev, m, verbose=verbose)
#             search_bn_recalibrate(m, remove_bn=remove_bn, verbose=verbose)
#             prev = m
