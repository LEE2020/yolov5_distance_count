from ultralytics import YOLO
import yaml
import torch 
from rola import C2f_RoLA, Detect_RoLA
# 加载预训练模型并注入RoLA结构
model = YOLO('yolo11n-seg')  # 加载官方预训练权重
for name,module in model.model.named_modules():
    print(name,"-=>",module.__class__.__name__)
#model.add_custom_modules({'C2f_RoLA': C2f_RoLA, 'Detect_RoLA': Detect_RoLA})  # 注册自定义模块‌:ml-citation{ref="8" data="citationList"}
# 执行微调

freeze_layers =[]
for name,module in model.model.named_modules():
    if 'backbone' in name:
        freeze_layers.append(name)
    
print(len(freeze_layers))

