import torch
import logging.config
from copy import deepcopy
from six import string_types
from .regime import Regime
from .param_filter import FilterParameters
from . import regularization
import torch.nn as nn

_OPTIMIZERS = {name: func for name, func in torch.optim.__dict__.items()}

try:
    from adabound import AdaBound
    _OPTIMIZERS['AdaBound'] = AdaBound
except ImportError:
    pass


def copy_params(param_target, param_src):
    with torch.no_grad():
        for p_src, p_target in zip(param_src, param_target):
            p_target.copy_(p_src)


def copy_params_grad(param_target, param_src):
    for p_src, p_target in zip(param_src, param_target):
        if p_target.grad is None:
            p_target.backward(p_src.grad.to(dtype=p_target.dtype))
        else:
            p_target.grad.detach().copy_(p_src.grad)


class ModuleFloatShadow(nn.Module):
    def __init__(self, module):
        super(ModuleFloatShadow, self).__init__()
        self.original_module = module
        self.float_module = deepcopy(module)
        self.float_module.to(dtype=torch.float)

    def parameters(self, *kargs, **kwargs):
        return self.float_module.parameters(*kargs, **kwargs)

    def named_parameters(self, *kargs, **kwargs):
        return self.float_module.named_parameters(*kargs, **kwargs)

    def modules(self, *kargs, **kwargs):
        return self.float_module.modules(*kargs, **kwargs)

    def named_modules(self, *kargs, **kwargs):
        return self.float_module.named_modules(*kargs, **kwargs)

    def original_parameters(self, *kargs, **kwargs):
        return self.original_module.parameters(*kargs, **kwargs)

    def original_named_parameters(self, *kargs, **kwargs):
        return self.original_module.named_parameters(*kargs, **kwargs)

    def original_modules(self, *kargs, **kwargs):
        return self.original_module.modules(*kargs, **kwargs)

    def original_named_modules(self, *kargs, **kwargs):
        return self.original_module.named_modules(*kargs, **kwargs)


class OptimRegime(Regime):
    """
    Reconfigures the optimizer according to setting list.
    Exposes optimizer methods - state, step, zero_grad, add_param_group

    Examples for regime:

    1)  "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
          {'epoch': 2, 'optimizer': 'Adam', 'lr': 5e-4},
          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
          {'epoch': 8, 'optimizer': 'Adam', 'lr': 5e-5}
         ]"
    2)
        "[{'step_lambda':
            "lambda t: {
            'optimizer': 'Adam',
            'lr': 0.1 * min(t ** -0.5, t * 4000 ** -1.5),
            'betas': (0.9, 0.98), 'eps':1e-9}
         }]"
    """

    def __init__(self, model, regime, defaults={}, filter=None, use_float_copy=False):
        super(OptimRegime, self).__init__(regime, defaults)
        if filter is not None:
            model = FilterParameters(model, **filter)
        if use_float_copy:
            model = ModuleFloatShadow(model)
            self._original_parameters = list(model.original_parameters())

        self.parameters = list(model.parameters())
        self.optimizer = torch.optim.SGD(self.parameters, lr=0)
        self.regularizer = regularization.Regularizer(model)
        self.use_float_copy = use_float_copy

    def update(self, epoch=None, train_steps=None):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        if super(OptimRegime, self).update(epoch, train_steps):
            self.adjust(self.setting)
            return True
        else:
            return False

    def adjust(self, setting):
        """adjusts optimizer according to a setting dict.
        e.g: setting={optimizer': 'Adam', 'lr': 5e-4}
        """
        if 'optimizer' in setting:
            optim_method = _OPTIMIZERS[setting['optimizer']]
            if not isinstance(self.optimizer, optim_method):
                self.optimizer = optim_method(self.optimizer.param_groups)
                logging.debug('OPTIMIZER - setting method = %s' %
                              setting['optimizer'])
        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    new_val = setting[key]
                    if new_val != param_group[key]:
                        logging.debug('OPTIMIZER - setting %s = %s' %
                                      (key, setting[key]))
                        param_group[key] = setting[key]
                        # fix for AdaBound
                        if key == 'lr' and hasattr(self.optimizer, 'base_lrs'):
                            self.optimizer.base_lrs = list(
                                map(lambda group: group['lr'], self.optimizer.param_groups))

        if 'regularizer' in setting:
            reg_list = deepcopy(setting['regularizer'])
            if not (isinstance(reg_list, list) or isinstance(reg_list, tuple)):
                reg_list = (reg_list,)
            regularizers = []
            for reg in reg_list:
                if isinstance(reg, dict):
                    logging.debug('OPTIMIZER - Regularization - %s' % reg)
                    name = reg.pop('name')
                    regularizers.append((regularization.__dict__[name], reg))
                elif isinstance(reg, regularization.Regularizer):
                    regularizers.append(reg)
                else:  # callable on model
                    regularizers.append(reg(self.regularizer._model))
            self.regularizer = regularization.RegularizerList(self.regularizer._model,
                                                              regularizers)

    def __getstate__(self):
        return {
            'optimizer_state': self.optimizer.__getstate__(),
            'regime': self.regime,
        }

    def __setstate__(self, state):
        self.regime = state.get('regime')
        self.optimizer.__setstate__(state.get('optimizer_state'))

    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.
        """
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'regime': self.regime,
        }

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        optimizer_state_dict = state_dict['optimizer_state']

        self.__setstate__({'optimizer_state': optimizer_state_dict,
                           'regime': state_dict['regime']})

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()
        if self.use_float_copy:
            for p in self._original_parameters:
                if p.grad is not None:
                    p.grad.detach().zero_()

    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        if self.use_float_copy:
            copy_params_grad(self.parameters, self._original_parameters)
        self.regularizer.pre_step()
        self.optimizer.step(closure)
        self.regularizer.post_step()
        if self.use_float_copy:
            copy_params(self._original_parameters, self.parameters)

    def pre_forward(self):
        """ allows modification pre-forward pass - e.g for regularization
        """
        self.regularizer.pre_forward()

    def pre_backward(self):
        """ allows modification post-forward pass and pre-backward - e.g for regularization
        """
        self.regularizer.pre_backward()


class MultiOptimRegime(OptimRegime):

    def __init__(self, *optim_regime_list):
        self.optim_regime_list = []
        for optim_regime in optim_regime_list:
            assert isinstance(optim_regime, OptimRegime)
            self.optim_regime_list.append(optim_regime)

    def update(self, epoch=None, train_steps=None):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        updated = False
        for i, optim in enumerate(self.optim_regime_list):
            current_updated = optim.update(epoch, train_steps)
            if current_updated:
                logging.debug('OPTIMIZER #%s was updated' % i)
            updated = updated or current_updated
        return updated

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        for optim in self.optim_regime_list:
            optim.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        for optim in self.optim_regime_list:
            optim.step(closure)
