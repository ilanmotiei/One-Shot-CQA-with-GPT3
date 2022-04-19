import torch.optim
from typing import Optional


class CostumeScheduler(torch.optim.Optimizer):
    def __init__(self, warmup_steps, lr_upper_bound, lr_decay_steps, lr_decay_gamma, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0
        self.lr_additive_factor = lr_upper_bound / warmup_steps
        self.warmup_steps = warmup_steps
        self.lr_upper_bound = lr_upper_bound
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_gamma = lr_decay_gamma

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0

    def step(self):
        for p in self.optimizer.param_groups:
            p['lr'] = self.get_updated_rate(p['lr'])

        self.optimizer.step()
        self._step += 1

    def get_updated_rate(self, rate):
        if self._step < self.warmup_steps:
            return rate + self.lr_additive_factor

        if self._step == self.warmup_steps:
            return self.lr_upper_bound

        # else:
        if (self._step + 1) % self.lr_decay_steps == 0:
            return rate * self.lr_decay_gamma

        # else:
        return rate

    def zero_grad(self, set_to_none: Optional[bool] = ...) -> None:
        self.optimizer.zero_grad()

    def state_dict(self) -> dict:
        dct = dict()
        dct['optimizer'] = self.optimizer.state_dict()
        dct['step'] = self._step
        dct['rate'] = self._rate
        dct['lr_additive_factor'] = self.lr_additive_factor
        dct['warmup_steps'] = self.warmup_steps
        dct['lr_upper_bound'] = self.lr_upper_bound
        dct['lr_decay_steps'] = self.lr_decay_steps
        dct['lr_decay_gamma'] = self.lr_decay_gamma

        return dct

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._step = state_dict['step']
        self._rate = state_dict['rate']
        self.lr_additive_factor = state_dict['lr_additive_factor']
        self.warmup_steps = state_dict['warmup_steps']
        self.lr_upper_bound = state_dict['upper_bound']
        self.lr_decay_steps = state_dict['lr_decay_steps']
        self.lr_decay_gamma = state_dict['lr_decay_gamma']

    def add_param_group(self, param_group: dict) -> None:
        self.optimizer.add_param_group(param_group)
