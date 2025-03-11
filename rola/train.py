from ultralytics import YOLO
import yaml
import torch 
from rola import C2f_RoLA, Detect_RoLA
# 加载预训练模型并注入RoLA结构
model = YOLO('yolo11n-seg')  # 加载官方预训练权重
print(model.model)  # 显示模型结构

model.add_custom_modules({'C2f_RoLA': C2f_RoLA, 'Detect_RoLA': Detect_RoLA})  # 注册自定义模块‌:ml-citation{ref="8" data="citationList"}


# 执行微调
results = model.train(
    data='dataset.yaml',
    cfg='yolov11-rola.yaml',
    epochs = 10,
    imgsz=640,
    device='0'  # 使用GPU加速
    
)
