# -*- coding: utf-8 -*-
# @Time    : 2025/3/3 15:00
# @Author  : brigg
# @Email   :
# @File    : model_used.py
# @Software: VS Code

import torch
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import random
from ultralytics import YOLO
from models.yolo import Model
# 加载预训练 YOLOv5 模型（通过 torch.hub）
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 使用 'yolov5s' 小型模型
#model = YOLO('./runs/detect/train4/weights/best.pt')
model = Model('./runs/train/exp21/weights/best.pt')

print(type(model))
print(model)
# 读取图像（YOLOv5 支持直接输入文件路径或 OpenCV 格式）
image = cv2.imread("cloud.jpg")  # 使用 OpenCV 读取图像
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式（YOLOv5 要求）

# 执行目标检测
results = model(image_rgb)  # 输入 RGB 格式图像
print(type(results))
# 解析检测结果
predictions = results.pandas().xyxy[0]  # 提取检测结果的 DataFrame 格式
# 绘制边界框和标签
# 定义列名  
columns = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]  
# 转换为 DataFrame  
data_2d = np.squeeze(predictions, axis=0)
predictions = pd.DataFrame(data_2d, columns=columns)  

for _, row in predictions.iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    confidence = row['confidence']
    class_name = row['name']
    
    # 绘制边界框
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # 绘制标签和置信度
    label = f"{class_name} {confidence:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 保存或显示结果
cv2.imwrite("cloud_.jpg", image)
cv2.imshow("YOLOv5 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
