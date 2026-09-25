# membrane_recorder.py
import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def _safe_name(name: str) -> str:
    # 文件名安全化
    return re.sub(r"[^0-9a-zA-Z_\-\.]", "_", name)


class MembraneStatsRecorder:
    """
    多层膜电位统计（pre-spike u_pre）：
    - 从 module.v_pre_for_record 读取（你已在 neuronal_charge 写入）
    - module_name_filter 为 list[str] 时：按 list 指定的层分别统计
    - 每层一个样本 buffer，上限 max_samples_total_per_layer，防止内存爆炸
    - 推理结束后每层分别保存图
    """

    def __init__(self,
                 module_name_filter=None,          # None or list[str]
                 neuron_type=None,                 # e.g., BPTTNeuron
                 max_batches=None,
                 sample_per_batch=20000,
                 max_samples_total_per_layer=1_000_000):
        self.module_name_filter = module_name_filter
        self.neuron_type = neuron_type
        self.max_batches = max_batches
        self.sample_per_batch = sample_per_batch
        self.max_samples_total_per_layer = max_samples_total_per_layer

        self.handles = []
        self._current_batch_idx = 0
        self._stop_collect = False

        # 维持顺序
        if isinstance(module_name_filter, (list, tuple)):
            self.layer_names = list(module_name_filter)
        elif module_name_filter is None:
            self.layer_names = None
        else:
            # 兼容传单个字符串
            self.layer_names = [module_name_filter]

        # 每层一个 buffer（CPU 1D tensor）
        self.samples_by_layer = {}  # name -> torch.Tensor

    # ------------------- 核心：为每个 layer 绑定一个 hook -------------------

    def _make_hook(self, layer_name: str):
        def _hook_fn(module, inputs, output):
            if self._stop_collect:
                return
            if not hasattr(module, "v_pre_for_record"):
                return
            u = module.v_pre_for_record
            if not isinstance(u, torch.Tensor):
                return

            u_flat = u.reshape(-1).float()

            # 每次调用抽样，控制开销
            if self.sample_per_batch is not None and u_flat.numel() > self.sample_per_batch:
                idx = torch.randint(0, u_flat.numel(), (self.sample_per_batch,),
                                    device=u_flat.device)
                u_flat = u_flat[idx]

            u_cpu = u_flat.cpu()

            # 合并到该层 buffer，并做全局上限随机下采样
            if layer_name not in self.samples_by_layer or self.samples_by_layer[layer_name] is None:
                if u_cpu.numel() > self.max_samples_total_per_layer:
                    idx = torch.randperm(u_cpu.numel())[:self.max_samples_total_per_layer]
                    self.samples_by_layer[layer_name] = u_cpu[idx]
                else:
                    self.samples_by_layer[layer_name] = u_cpu
            else:
                combined = torch.cat([self.samples_by_layer[layer_name], u_cpu], dim=0)
                if combined.numel() > self.max_samples_total_per_layer:
                    idx = torch.randperm(combined.numel())[:self.max_samples_total_per_layer]
                    combined = combined[idx]
                self.samples_by_layer[layer_name] = combined

        return _hook_fn

    def register(self, model):
        """
        注册 hooks：
        - 若 layer_names 给定，则只对这些精确名字的模块注册 hook（按顺序）
        - 若 layer_names 为 None，则对所有满足 neuron_type 的模块注册（不建议用于你现在的多图需求）
        """
        if self.layer_names is not None:
            name_to_module = dict(model.named_modules())
            for name in self.layer_names:
                if name not in name_to_module:
                    print(f"[MemRecorder] WARNING: layer name not found: {name}")
                    continue
                m = name_to_module[name]
                if self.neuron_type is not None and not isinstance(m, self.neuron_type):
                    print(f"[MemRecorder] WARNING: layer is not target type: {name} ({type(m)})")
                    continue
                h = m.register_forward_hook(self._make_hook(name))
                self.handles.append(h)
        else:
            # 全模型统计（可选）
            for name, m in model.named_modules():
                if self.neuron_type is not None and not isinstance(m, self.neuron_type):
                    continue
                h = m.register_forward_hook(self._make_hook(name))
                self.handles.append(h)

    def step_batch(self):
        if self.max_batches is not None:
            self._current_batch_idx += 1
            if self._current_batch_idx >= self.max_batches:
                self._stop_collect = True

    # ------------------- 绘图/保存 -------------------

    def plot_and_save_all(self,
                          save_path_template: str,
                          bins: int = 300,
                          density: bool = True,
                          vth: float = None,
                          remove_zero_eps: float = None):
        """
        save_path_template: 字符串，必须包含 {layer}
            例如:
            "./logs/exp1/fgsm_eps8_membrane_{dataset}_{model}_T8_{layer}.png"
        """

        if "{layer}" not in save_path_template:
            raise ValueError("save_path_template must contain '{layer}' placeholder")

        ordered = self.layer_names if self.layer_names is not None else list(self.samples_by_layer.keys())

        for idx, layer_name in enumerate(ordered):
            if layer_name not in self.samples_by_layer:
                print(f"[MemRecorder] No samples for layer: {layer_name}")
                continue

            vals = self.samples_by_layer[layer_name].numpy()

            # 可选：过滤接近 0 的值（只影响可视化）
            if remove_zero_eps is not None:
                vals = vals[abs(vals) > remove_zero_eps]
                if vals.size == 0:
                    print(f"[MemRecorder] All values filtered for layer: {layer_name}")
                    continue

            vmin, vmax = float(vals.min()), float(vals.max())

            plt.figure(figsize=(6, 4))
            plt.hist(vals, bins=bins, range=(vmin, vmax), density=density, alpha=0.7)

            plt.xlim(-2.5, 2.5)

            if vth is not None:
                plt.axvline(x=vth, color='r', linestyle='--', label='Vth', lw=0.5)
                # plt.axvline(x=0.0, color='g', linestyle='--', label='0')
                plt.legend()

            plt.xlabel("Membrane potential u (pre-spike)")
            plt.ylabel("Density" if density else "Count")
            plt.title(layer_name)

            #  用 layer_name 展开路径
            safe_layer = layer_name.replace('.', '_')
            save_path = save_path_template.format(layer=safe_layer)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[MemRecorder] Saved: {save_path}")

    def export_hist_to_excel_all(self,
                                excel_path: str,
                                bins: int = 300,
                                density: bool = True,
                                vth: float = None,               # 保留接口一致性（导出用不到也没关系）
                                remove_zero_eps: float = None,
                                xlim: tuple = (-2.5, 2.5),
                                use_xlim_as_hist_range: bool = False):
        """
        将每层的直方图数据导出到一个 Excel 文件（每层一个 sheet）。

        - 默认严格复刻 plot_and_save_all 的逻辑：hist range = (vmin, vmax)
        - 你代码里虽然 plt.xlim(-2.5, 2.5)，但 hist 仍按 (vmin, vmax) 统计；
          如果你希望 histogram 也只统计 [-2.5, 2.5] 这段区间，把 use_xlim_as_hist_range=True
        """

        ordered = self.layer_names if self.layer_names is not None else list(self.samples_by_layer.keys())

        os.makedirs(os.path.dirname(excel_path), exist_ok=True)

        def _safe_sheet(name: str) -> str:
            # Excel sheet 名最长 31，且不能含某些字符
            name = re.sub(r"[\[\]\*:/\\\?]", "_", name)
            return name[:31] if len(name) > 31 else name

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for layer_name in ordered:
                if layer_name not in self.samples_by_layer:
                    print(f"[MemRecorder] No samples for layer: {layer_name}")
                    continue

                vals = self.samples_by_layer[layer_name].numpy()

                if remove_zero_eps is not None:
                    vals = vals[np.abs(vals) > remove_zero_eps]
                    if vals.size == 0:
                        print(f"[MemRecorder] All values filtered for layer: {layer_name}")
                        continue

                if use_xlim_as_hist_range and xlim is not None:
                    vmin, vmax = float(xlim[0]), float(xlim[1])
                else:
                    vmin, vmax = float(vals.min()), float(vals.max())

                # 复刻 plt.hist 的统计结果
                hist_y, edges = np.histogram(vals, bins=bins, range=(vmin, vmax), density=density)
                centers = 0.5 * (edges[:-1] + edges[1:])

                df = pd.DataFrame({
                    "x": centers,
                    "y": hist_y
                })

                sheet_name = _safe_sheet(layer_name.replace('.', '_'))
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"[MemRecorder] Exported histogram data to: {excel_path}")


    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear(self):
        self._current_batch_idx = 0
        self._stop_collect = False
        self.samples_by_layer.clear()
