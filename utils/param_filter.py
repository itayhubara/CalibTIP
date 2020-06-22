import torch
import torch.nn as nn


def is_not_bias(name):
    return not name.endswith('bias')


def is_bn(module):
    return isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d)


def is_not_bn(module):
    return not is_bn(module)


def filtered_parameter_info(model, module_fn=None, module_name_fn=None, parameter_name_fn=None, memo=None):
    if memo is None:
        memo = set()

    for module_name, module in model.named_modules():
        if module_fn is not None and not module_fn(module):
            continue
        if module_name_fn is not None and not module_name_fn(module_name):
            continue
        for parameter_name, param in module.named_parameters(prefix=module_name, recurse=False):
            if parameter_name_fn is not None and not parameter_name_fn(parameter_name):
                continue
            if param not in memo:
                memo.add(param)
                yield {'named_module': (module_name, module), 'named_parameter': (parameter_name, param)}


class FilterParameters(object):
    def __init__(self, source, module=None, module_name=None, parameter_name=None):
        if isinstance(source, FilterParameters):
            self._filtered_parameter_info = list(source.filter(
                                                 module=module,
                                                 module_name=module_name,
                                                 parameter_name=parameter_name))
        elif isinstance(source, torch.nn.Module):  # source is a model
            self._filtered_parameter_info = list(filtered_parameter_info(source,
                                                                         module_fn=module,
                                                                         module_name_fn=module_name,
                                                                         parameter_name_fn=parameter_name))

    def named_parameters(self):
        for p in self._filtered_parameter_info:
            yield p['named_parameter']

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def filter(self, module=None, module_name=None, parameter_name=None):
        for p_info in self._filtered_parameter_info:
            if (module is None or module(p_info['named_module'][1])
                and (module_name is None or module_name(p_info['named_module'][0]))
                    and (parameter_name is None or parameter_name(p_info['named_parameter'][0]))):
                yield p_info

    def named_modules(self):
        for m in self._filtered_parameter_info:
            yield m['named_module']

    def modules(self):
        for _, m in self.named_modules():
            yield m

    def to(self, *kargs, **kwargs):
        for m in self.modules():
            m.to(*kargs, **kwargs)


class FilterModules(FilterParameters):
    pass

if __name__ == '__main__':
    from torchvision.models import resnet50
    model = resnet50()
    filterd_params = FilterParameters(model,
                                      module=lambda m: isinstance(
                                          m, torch.nn.Linear),
                                      parameter_name=lambda n: 'bias' in n)
