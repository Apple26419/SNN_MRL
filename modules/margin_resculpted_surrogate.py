from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn as nn


def _infer_device_from_model(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


class _MRLCollector:
    def __init__(self):
        self.regs: List[torch.Tensor] = []
        self.device: torch.device = torch.device("cpu")

    def reset(self, device: torch.device) -> None:
        self.device = device
        self.regs.clear()

    def add(self, reg: torch.Tensor) -> None:
        self.regs.append(reg)

    def pop_aux_and_clear(self) -> torch.Tensor:
        if len(self.regs) == 0:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        aux = torch.stack(self.regs).mean()
        self.regs.clear()
        return aux

    def force_clear(self) -> None:
        self.regs.clear()


_MRL = _MRLCollector()


def _get_alpha_from_surrogate(base_surrogate: nn.Module, default: float) -> float:
    if hasattr(base_surrogate, "alpha"):
        try:
            a = float(getattr(base_surrogate, "alpha"))
            if a > 0:
                return a
        except Exception:
            pass
    return float(default)


def _kernel_sigmoid(v_hat: torch.Tensor, alpha: float) -> torch.Tensor:
    s = torch.sigmoid(v_hat * alpha)
    return 4.0 * s * (1.0 - s)


def _kernel_triangle(v_hat: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.relu(1.0 - v_hat.abs() / alpha)


def _kernel_rectangle(v_hat: torch.Tensor, alpha: float) -> torch.Tensor:
    tau = max(alpha * 0.25, 1e-3)
    return torch.sigmoid((alpha - v_hat.abs()) / tau)


class MarginResculptedSurrogate(nn.Module):
    def __init__(
        self,
        base_surrogate: nn.Module,
        gamma: float = 0.1,
        eps: float = 1e-8,
        detach_stats: bool = True,
        kernel: str = "auto",
        alpha_sig: float = 4.0,
        alpha_rec: float = 1.0,
        alpha_tri: float = 1.0,
        c: float = 1.0,
        max_elems: int = 4096,
        gate_tau: float = 0.25,
    ):
        super().__init__()
        assert gamma >= 0.0
        assert eps > 0
        assert c > 0
        assert max_elems >= 256

        self.base_surrogate = base_surrogate
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.detach_stats = bool(detach_stats)

        self.kernel = str(kernel).lower()
        self.alpha_sig = float(alpha_sig)
        self.alpha_rec = float(alpha_rec)
        self.alpha_tri = float(alpha_tri)

        self.c = float(c)
        self.max_elems = int(max_elems)
        self.gate_tau = float(gate_tau)

    def _subsample_flat(self, v: torch.Tensor) -> torch.Tensor:
        flat = v.reshape(-1)
        n = flat.numel()
        if n <= self.max_elems:
            return flat
        stride = max(1, n // self.max_elems)
        return flat[::stride]

    def _std(self, v_s: torch.Tensor) -> torch.Tensor:
        if self.detach_stats:
            std = v_s.detach().float().std(unbiased=False)
        else:
            std = v_s.float().std(unbiased=False)
        return std.clamp_min(self.eps)

    def _infer_kernel_type(self) -> str:
        if self.kernel != "auto":
            return self.kernel

        name = self.base_surrogate.__class__.__name__.lower()
        if "sigmoid" in name:
            return "sig"
        if "rectangle" in name:
            return "rec"
        if "piecewisequadratic" in name or "quadratic" in name or "triangle" in name:
            return "tri"
        return "sig"

    def _kernel_weight(self, v_hat: torch.Tensor) -> torch.Tensor:
        k = self._infer_kernel_type()
        if k == "sig":
            a = _get_alpha_from_surrogate(self.base_surrogate, self.alpha_sig)
            return _kernel_sigmoid(v_hat, a)
        if k == "rec":
            a = _get_alpha_from_surrogate(self.base_surrogate, self.alpha_rec)
            return _kernel_rectangle(v_hat, a)
        if k == "tri":
            a = _get_alpha_from_surrogate(self.base_surrogate, self.alpha_tri)
            return _kernel_triangle(v_hat, a)

        a = _get_alpha_from_surrogate(self.base_surrogate, self.alpha_sig)
        return _kernel_sigmoid(v_hat, a)

    def _gate_pos(self, v_hat: torch.Tensor) -> torch.Tensor:

        return torch.sigmoid(v_hat / self.gate_tau)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        y = self.base_surrogate(v)

        if self.training and self.gamma > 0.0:
            v_s = self._subsample_flat(v)
            std = self._std(v_s)
            v_hat = v_s / std.to(v_s.dtype)
            w = self._kernel_weight(v_hat)
            occ = w.mean()
            rho = y.detach().float().mean()
            b = (rho * (1.0 - rho)) ** 2
            r_occ = torch.relu(occ - b)
            reg = r_occ
            _MRL.add(reg)
        return y

def mrl_reset(model: nn.Module) -> None:
    device = _infer_device_from_model(model)
    _MRL.reset(device=device)

def mrl_loss(model: nn.Module) -> torch.Tensor:
    device = _infer_device_from_model(model)
    _MRL.device = device
    return _MRL.pop_aux_and_clear()

def mrl_force_clear() -> None:
    _MRL.force_clear()
