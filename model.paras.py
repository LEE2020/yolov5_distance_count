import torch
model = torch.load('yolov5s.pt')['model']  # 加载 .pt 文件中的模型对象
#print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")  # 输出总参数量 ‌:ml-citation{ref="1,2" data="citationList"}
import torch

names = model.names  # YOLOv5 直接通过 model.names 获取类别列表 ‌:ml-citation{ref="5" data="citationList"}  
# 或  
names = model.module.names if hasattr(model, 'module') else model.names  # 兼容多 GPU 训练场景 ‌:ml-citation{ref="5" data="citationList"}  
print(len(names))  # 输出所有可识别的目标类别名称
print()
model2 = torch.load('./runs/detect/train4/weights/best.pt')['model']
names2 = model2.names
print(names2)

