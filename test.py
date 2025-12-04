import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import os
from utils import patchify, unpatchify

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

def cdf_quantile_match_pytorch(s, t):
    """
    s: source distribution
    t: target distribution
    """

    N = len(s)

    sorted_indices_s = torch.argsort(s) 
    
    s_ranks = torch.empty_like(sorted_indices_s, dtype=torch.float32).to(s.device)
    s_ranks[sorted_indices_s] = torch.arange(N, dtype=torch.float32).to(s.device) + 1
    
    t_sorted, _ = torch.sort(t)
    t_indices = (s_ranks - 1).long()

    s_new = t_sorted[t_indices]

    return s_new

def _single_match(T1_i: torch.Tensor, T2_i: torch.Tensor) -> torch.Tensor:
    """
    辅助函数：对单个 (L,) 样本进行 CDF 分位数匹配。
    """
    L = T1_i.numel()
    
    # 1. 排序
    T1_sorted, T1_indices = torch.sort(T1_i)
    T2_sorted, _ = torch.sort(T2_i)
    
    # 2. 创建 T1 的反向排序索引
    # 这一步是关键，它建立了原 tensor 值和其排名的映射
    T1_inverse_indices = torch.empty_like(T1_indices)
    T1_inverse_indices[T1_indices] = torch.arange(L, device=T1_i.device)
    
    # 3. 匹配和重建
    T_matched_i = torch.empty_like(T1_i)
    T_matched_i[T1_inverse_indices] = T2_sorted
    
    return T_matched_i

def cdf_quantile_match_vmap(tensor_to_match: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    使用 torch.vmap 进行 Batch-wise CDF 分位数匹配。
    """
    # 映射到第一个维度 (Batch 维度 B)
    # in_dims=(0, 0) 表示 T1 和 T2 的第 0 维都是 Batch 维度
    # out_dim=0 表示输出的第 0 维是 Batch 维度
    return torch.vmap(_single_match, in_dims=(0, 0))(tensor_to_match, target_tensor)



# --- 您的原始数据 ---
a_target = torch.tensor([[0, 0.1, 0.1, 0.2, 0.2, 2, 3, 3, 4]], dtype=torch.float32)
b_original = torch.tensor([[0, 1, 1, 1, 1, 2, 2, 2, 3]], dtype=torch.float32)

# # --- 运行 PyTorch 匹配 ---
# b_transformed_cdf = cdf_quantile_match_vmap(b_original, a_target)

b_transformed_cdf = (b_original - b_original.mean(dim=1, keepdim=True)) / b_original.std(dim=1, keepdim=True) * \
                    a_target.std(dim=1, keepdim=True) + a_target.mean(dim=1, keepdim=True)

# --- 结果展示 ---
import pandas as pd

# 创建一个 Pandas DataFrame 进行对比
df_results = pd.DataFrame({
    'b_original': b_original[0],
    'a_target': a_target[0],
    'b_transformed_cdf': b_transformed_cdf[0].tolist()
})

print("\n--- 最终变换结果 (PyTorch CDF 匹配) ---")
print(df_results)

print("\n结果验证:")
print(f"目标 a 排序: {torch.sort(torch.tensor(a_target))[0][0].tolist()}")
print(f"b_transformed 排序: {torch.sort(b_transformed_cdf)[0][0].tolist()}")