import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect, C2f

class RoLAAdapter(nn.Module):
    def __init__(self, in_channels, rank=8):
        super().__init__()
        # 低秩矩阵参数化
        self.down_proj = nn.Linear(in_channels, rank, bias=False)
        self.up_proj = nn.Linear(rank, in_channels, bias=False)
    
    def forward(self, x):
        identity = x
        x = self.down_proj(x)
        x = self.up_proj(x)
        return identity + x  # 残差连接

# 修改C2f模块（插入RoLA）
class C2f_RoLA(C2f):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rola = RoLAAdapter(self.cv2.conv.in_channels)  # 在C2f的第二层卷积后插入

# 修改检测头（插入RoLA）
class Detect_RoLA(Detect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rola = RoLAAdapter(self.cv3.conv.in_channels)  # 在检测头的分类分支插入



