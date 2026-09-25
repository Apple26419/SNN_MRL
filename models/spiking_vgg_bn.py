# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from spikingjelly.clock_driven import layer
#
# __all__ = [
#     'SpikingVGGBN', 'vgg5', 'vgg11'
# ]
#
# import torch
# cfg = {
#     'VGG5': [
#         [64, 'M'],
#         [128, 128, 'M'],
#         [],
#         [],
#         []
#     ],
#
#     'VGG11': [
#         [64, 'M'],
#         [128, 'M'],
#         [256, 256, 'M'],
#         [512, 512, 'M'],
#         [512, 512, 'M']
#     ],
#     'VGG13': [
#         [64, 64, 'M'],
#         [128, 128, 'M'],
#         [256, 256, 'M'],
#         [512, 512, 'M'],
#         [512, 512, 'M']
#     ],
#     'VGG16': [
#         [64, 64, 'M'],
#         [128, 128, 'M'],
#         [256, 256, 256, 'M'],
#         [512, 512, 512, 'M'],
#         [512, 512, 512, 'M']
#     ],
#     'VGG19': [
#         [64, 64, 'M'],
#         [128, 128, 'M'],
#         [256, 256, 256, 256, 'M'],
#         [512, 512, 512, 512, 'M'],
#         [512, 512, 512, 512, 'M']
#     ]
# }
#
#
# class SpikingVGGBN(nn.Module):
#     def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
#
#         super(SpikingVGGBN, self).__init__()
#         self.whether_bias = True
#         self.init_channels = kwargs.get('c_in', 2)
#         max_channels = max(x for sublist in cfg[vgg_name] for x in sublist if isinstance(x, int))
#
#
#         self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
#         self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
#         self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
#         self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
#         self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(max_channels*7*7, num_classes),
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layers(self, cfg_layer, dropout, neuron, **kwargs):
#         layers = []
#         for x in cfg_layer:
#             if x == 'M':
#                 layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
#             else:
#                 layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
#                 layers.append(nn.BatchNorm2d(x))
#                 # 判断是否需要统计脉冲：如果 record_spikes 为 True，则用 SpikeRecorder 包装 neuron 层
#                 layers.append(neuron(**kwargs))
#                 layers.append(layer.Dropout(dropout))
#                 self.init_channels = x
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # 如果需要记录脉冲，清空之前的记录
#
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.avgpool(out)
#         out = self.classifier(out)
#         # 如果需要记录脉冲，可以选择同时返回 spike_records
#         return out
#
#
# def vgg11(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
#     return SpikingVGGBN('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)
#
# def vgg5(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
#     return SpikingVGGBN('VGG5', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)
#


'''
这是脉冲稀疏性正则化
'''

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from spikingjelly.clock_driven import layer
#
# __all__ = [
#     'SpikingVGGBN', 'vgg5', 'vgg11'
# ]
#
# cfg = {
#     'VGG5': [
#         [64, 'M'],
#         [128, 128, 'M'],
#         [],
#         [],
#         []
#     ],
#
#     'VGG11': [
#         [64, 'M'],
#         [128, 'M'],
#         [256, 256, 'M'],
#         [512, 512, 'M'],
#         [512, 512, 'M']
#     ],
#     'VGG13': [
#         [64, 64, 'M'],
#         [128, 128, 'M'],
#         [256, 256, 'M'],
#         [512, 512, 'M'],
#         [512, 512, 'M']
#     ],
#     'VGG16': [
#         [64, 64, 'M'],
#         [128, 128, 'M'],
#         [256, 256, 256, 'M'],
#         [512, 512, 512, 'M'],
#         [512, 512, 512, 'M']
#     ],
#     'VGG19': [
#         [64, 64, 'M'],
#         [128, 128, 'M'],
#         [256, 256, 256, 256, 'M'],
#         [512, 512, 512, 512, 'M'],
#         [512, 512, 512, 512, 'M']
#     ]
# }
#
#
# class SpikingVGGBN(nn.Module):
#     def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, record=False, **kwargs):
#         super(SpikingVGGBN, self).__init__()
#         self.whether_bias = True
#         self.init_channels = kwargs.get('c_in', 2)
#         max_channels = max(x for sublist in cfg[vgg_name] for x in sublist if isinstance(x, int))
#
#         self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
#         self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
#         self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
#         self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
#         self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(max_channels*7*7, num_classes),
#         )
#
#         self.record = record  # 控制是否记录每个neuron层的输出
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layers(self, cfg_layer, dropout, neuron, **kwargs):
#         layers = []
#         for x in cfg_layer:
#             if x == 'M':
#                 layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
#             else:
#                 layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
#                 layers.append(nn.BatchNorm2d(x))
#                 # 创建 neuron 层，并标记以便在forward时记录输出
#                 neuron_layer = neuron(**kwargs)
#                 neuron_layer.is_neuron = True  # 添加标记属性
#                 layers.append(neuron_layer)
#                 layers.append(layer.Dropout(dropout))
#                 self.init_channels = x
#         return nn.Sequential(*layers)
#
#
#     # # 以下是直接将neuron层的输出保存在列表中的实现
#     # def forward(self, x):
#     #     if self.training and self.record:
#     #         neuron_outputs = []
#     #     # 对每个layer block中的每个子层进行逐层前向传播
#     #     for block in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
#     #         for l in block:
#     #             x = l(x)
#     #             # 若当前层为neuron层，且满足记录条件，则保存该层输出
#     #             if self.training and self.record and hasattr(l, 'is_neuron') and l.is_neuron:
#     #                 neuron_outputs.append(x)
#     #     x = self.avgpool(x)
#     #     x = self.classifier(x)
#     #     # 只有在训练阶段且record=True时返回额外的neuron输出
#     #     if self.training and self.record:
#     #         return x, neuron_outputs
#     #     else:
#     #         return x
#
#     # spike_rate 实现
#     def forward(self, x):
#         # 如果在训练阶段且需要记录，则初始化累加器和计数器
#         if self.training and self.record:
#             spike_reg_sum = 0.0  # 累加所有 neuron 层的 spike_rate
#             spike_count = 0  # 计数 neuron 层出现的次数
#
#         # 逐层遍历每个 block 中的各个层
#         for block in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
#             for l in block:
#                 x = l(x)
#                 # 若当前层为 neuron 层且需要记录，则计算脉冲发射率并累加
#                 if self.training and self.record and hasattr(l, 'is_neuron') and l.is_neuron:
#                     # 计算当前层输出的脉冲发射率
#                     rate = x.float().sum() / x.numel()
#                     spike_reg_sum = spike_reg_sum + rate  # 累加
#                     spike_count += 1
#
#         x = self.avgpool(x)
#         x = self.classifier(x)
#
#         # 若需要记录，则返回输出和 neuron 层的平均脉冲发射率
#         if self.training and self.record and spike_count > 0:
#             spike_reg_mean = spike_reg_sum / spike_count
#             return x, spike_reg_mean
#         else:
#             return x
#
#
# def vgg11(neuron: callable = None, num_classes=10, neuron_dropout=0.0, record=False, **kwargs):
#     return SpikingVGGBN('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, record=record, **kwargs)
#
#
# def vgg5(neuron: callable = None, num_classes=10, neuron_dropout=0.0, record=False, **kwargs):
#     return SpikingVGGBN('VGG5', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, record=record, **kwargs)



import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer

__all__ = [
    'SpikingVGGBN', 'vgg5', 'vgg11'
]

import torch
cfg = {
    'VGG5': [
        [64, 'M'],
        [128, 128, 'M'],
        [],
        [],
        []
    ],

    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class SpikingVGGBN(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):

        super(SpikingVGGBN, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)
        self.name = vgg_name
        max_channels = max(x for sublist in cfg[vgg_name] for x in sublist if isinstance(x, int))


        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_channels*7*7, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg_layer, dropout, neuron, **kwargs):
        layers = []
        for x in cfg_layer:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                layers.append(nn.BatchNorm2d(x))
                layers.append(neuron(**kwargs))
                layers.append(layer.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out




def vgg16(neuron: callable = None, num_classes=10, neuron_dropout=0.0, record=False, **kwargs):
    return SpikingVGGBN('VGG16', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, record=record, **kwargs)

def vgg13(neuron: callable = None, num_classes=10, neuron_dropout=0.0, record=False, **kwargs):
    return SpikingVGGBN('VGG13', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, record=record, **kwargs)

def vgg11(neuron: callable = None, num_classes=10, neuron_dropout=0.0, record=False, **kwargs):
    return SpikingVGGBN('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, record=record, **kwargs)

def vgg5(neuron: callable = None, num_classes=10, neuron_dropout=0.0, record=False, **kwargs):
    return SpikingVGGBN('VGG5', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, record=record, **kwargs)

