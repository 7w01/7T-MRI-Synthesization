# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgPool3d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.max_r3 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-3]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-2]
            self.kernel_size[2] = x.shape[4] * self.base_size[2] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-3])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-2])
            self.max_r3 = max(1, self.rs[0] * x.shape[4] // train_size[-1])

        if self.kernel_size[0] >= x.size(-3) and self.kernel_size[1] >= x.size(-2) and self.kernel_size[2] >= x.size(-1):
            return F.adaptive_avg_pool3d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            d, h, w = x.shape[2:]
            if self.kernel_size[0] >= d and self.kernel_size[1] >= h and self.kernel_size[2] >= w:
                out = F.adaptive_avg_pool3d(x, 1)
            else:
                r1 = [r for r in self.rs if d % r == 0][0]
                r2 = [r for r in self.rs if h % r == 0][0]
                r3 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                r3 = min(self.max_r3, r3)
                s = x[:, :, ::r1, ::r2, ::r3].cumsum(dim=-1).cumsum(dim=-2).cumsum(dim=-3)
                n, c, d, h, w = s.shape
                k1, k2, k3 = min(d - 1, self.kernel_size[0] // r1), min(h - 1, self.kernel_size[1] // r2), min(w - 1, self.kernel_size[2] // r3)
                out = (s[:, :, :-k1, :-k2, :-k3] - s[:, :, :-k1, :-k2, k3:] - s[:, :, :-k1, k2:, :-k3] - s[:, :, k1:, :-k2, :-k3] 
                       + s[:, :, :-k1, k2:, k3:] + s[:, :, k1:, :-k2, k3:] + s[:, :, k1:, k2:, :-k3] - s[:, :, k1:, k2:, k3:]) / (k1 * k2 * k3)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2, r3))
        else:
            n, c, d, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2).cumsum_(dim=-3)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0, 1, 0))  # pad 0 for convenience
            k1, k2, k3 = min(d, self.kernel_size[0]), min(h, self.kernel_size[1]), min(w, self.kernel_size[2])
            s1 = s[:, :, :-k1, :-k2, :-k3]
            s2 = s[:, :, :-k1, :-k2, k3:]
            s3 = s[:, :, :-k1, k2:, :-k3]
            s4 = s[:, :, :-k1, k2:, k3:]
            s5 = s[:, :, k1:, :-k2, :-k3]
            s6 = s[:, :, k1:, :-k2, k3:]
            s7 = s[:, :, k1:, k2:, :-k3]
            s8 = s[:, :, k1:, k2:, k3:]
            out = s8 + s1 + s2 + s3 - s4 - s5 - s6 - s7
            out = out / (k1 * k2 * k3)

        if self.auto_pad:
            n, c, d, h, w = x.shape
            _d, _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad3d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2, (d - _d) // 2, (d - _d + 1) // 2)
            out = torch.nn.functional.pad(out, pad3d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool3d):
            pool = AvgPool3d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


'''
ref. 
@article{chu2021tlsc,
  title={Revisiting Global Statistics Aggregation for Improving Image Restoration},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''
class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)