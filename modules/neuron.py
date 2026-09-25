from typing import Callable
import torch
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj

class BPTTNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = None,
            detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)


    def neuronal_charge(self, x: torch.Tensor):
        # 默认有这个decay，如果tau过大的话，虽然衰减速度变慢，但是输入会直接除以decay，使得输入特别小，神经元基本都不放电
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x

        self.v_pre_for_record = self.v.detach()















