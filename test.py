import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import os

# 10,8839,1168   1,088,391,168
# [144, 144, 144]
# 1. (b, 1, 24, 24, 24, 6, 6, 6) -> (b, 512, 24, 24, 24) -> (b, 512, 3, 3, 3, 8, 8, 8)
# 2. (b, 1, 144, 144, 144) -> (b, 24, 144, 144, 144) -> (b, 24, 8, 8, 8, 18, 18, 18)

# 288, 360, 312

# 1. (b, 1, 48, 60, 52, 6, 6, 6) -> (b, 512, 48, 60, 52) -> (b, 512, 6, 8, 7, 8, 8, 8)
# 1. (b, 1, 24, 30, 26, 12, 12, 12) -> (b, 2048, 24, 30, 26) -> (b, 2048, 3, 4, 4, 8, 8, 8)
# 2. (b, 1, 288, 360, 312) -> (b, 24, 288, 360, 312)

# [36, 45, 39] [8, 8, 8]


# for file in os.listdir('./visualize'):
#     if file.endswith('.png'):
#         os.system(f'rm ./visualize/{file}')


import torch

a = torch.tensor([[1,2,3,4,5], [1,2,3,4,5]])
print(a[2:])
