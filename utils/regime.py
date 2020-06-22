import torch
from copy import deepcopy
from six import string_types


def eval_func(f, x):
    if isinstance(f, string_types):
        f = eval(f)
    return f(x)


class Regime(object):
    """
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

    def __init__(self, regime, defaults={}):
        self.regime = regime
        self.current_regime_phase = None
        self.setting = defaults

    def update(self, epoch=None, train_steps=None):
        """adjusts according to current epoch or steps and regime.
        """
        if self.regime is None:
            return False
        epoch = -1 if epoch is None else epoch
        train_steps = -1 if train_steps is None else train_steps
        setting = deepcopy(self.setting)
        if self.current_regime_phase is None:
            # Find the first entry where the epoch is smallest than current
            for regime_phase, regime_setting in enumerate(self.regime):
                start_epoch = regime_setting.get('epoch', 0)
                start_step = regime_setting.get('step', 0)
                if epoch >= start_epoch or train_steps >= start_step:
                    self.current_regime_phase = regime_phase
                    break
                # each entry is updated from previous
                setting.update(regime_setting)
        if len(self.regime) > self.current_regime_phase + 1:
            next_phase = self.current_regime_phase + 1
            # Any more regime steps?
            start_epoch = self.regime[next_phase].get('epoch', float('inf'))
            start_step = self.regime[next_phase].get('step', float('inf'))
            if epoch >= start_epoch or train_steps >= start_step:
                self.current_regime_phase = next_phase
        setting.update(self.regime[self.current_regime_phase])

        if 'lr_decay_rate' in setting and 'lr' in setting:
            decay_steps = setting.pop('lr_decay_steps', 100)
            if train_steps % decay_steps == 0:
                decay_rate = setting.pop('lr_decay_rate')
                setting['lr'] *= decay_rate ** (train_steps / decay_steps)
        elif 'step_lambda' in setting:
            setting.update(eval_func(setting.pop('step_lambda'), train_steps))
        elif 'epoch_lambda' in setting:
            setting.update(eval_func(setting.pop('epoch_lambda'), epoch))

        if 'execute' in setting:
            setting.pop('execute')()

        if 'execute_once' in setting:
            setting.pop('execute_once')()
            # remove from regime, so won't happen again
            self.regime[self.current_regime_phase].pop('execute_once', None)

        if setting == self.setting:
            return False
        else:
            self.setting = setting
            return True
