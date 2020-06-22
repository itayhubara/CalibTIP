import torch
from torch.autograd.function import Function

class ScaleGrad(Function):

    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.scale * grad_output
        return grad_input, None


def scale_grad(x, scale):
    return ScaleGrad().apply(x, scale)

def negate_grad(x):
    return scale_grad(x, -1)
