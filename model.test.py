import cv2
from ultralytics import YOLO
# 在静态的图片上检测yoloV5微调后的模型效果
# 步骤1：加载模型
model = YOLO('./runs/detect/train4/weights/best.pt')  # YOLOv5 微调模型文件‌:ml-citation{ref="1,2" data="citationList"}

# 步骤2：读取图像（无需预处理）
image = cv2.imread('win1.jpg')  # 直接输入 BGR 格式‌:ml-citation{ref="2,3" data="citationList"}

# 步骤3：执行推理
results = model(image)  # 返回 Results 对象列表‌:ml-citation{ref="1,2" data="citationList"}

# 步骤4：提取检测结果（兼容列表结构）
detections = results.xyxy.cpu().numpy()  # 提取第一张图的检测框并转为 NumPy 数组‌:ml-citation{ref="1,2" data="citationList"}




# 遍历检测框并绘制
for detection in detections:
    x1, y1, x2, y2 = map(int, detection[:4])  # 坐标转整数‌:ml-citation{ref="2,3" data="citationList"}
    confidence = detection‌:ml-citation{ref="3" data="citationList"}  # 置信度‌:ml-citation{ref="2" data="citationList"}
    class_id = int(detection‌:ml-citation{ref="6" data="citationList"})  # 类别 ID‌:ml-citation{ref="2" data="citationList"}
    class_name = model.names[class_id]  # 类别名称映射‌:ml-citation{ref="1,2" data="citationList"}

    # 绘制边界框和标签
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f'{class_name} {confidence:.2f}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 显示结果
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()