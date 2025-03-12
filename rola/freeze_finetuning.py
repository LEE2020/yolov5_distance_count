from ultralytics import YOLO
import yaml
import torch

# 加载模型和数据集配置
model = YOLO("yolo11n-seg.pt")
with open("rola/dataset.yaml", "r") as f:
    data_config = yaml.safe_load(f)

modules_layer = list(model.model.named_modules())
#第一个阶段，冻结backbond部分层数
freeze_layers = int(len(modules_layer)*0.1)
for  idx, (name, module) in enumerate(modules_layer):
        if idx < freeze_layers:
            for param in module.parameters():
                param.requires_grad = False

results_phase1 = model.train(data="rola/dataset.yaml", epochs=50,batch=32, imgsz=640,\
                         hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,device='cpu')

mAP_history = results_phase1["validation"]["mAP"]
max_phase1_epochs = 100
current_epoch = 0

while current_epoch < max_phase1_epochs:
    result = model.train(data="rola/dataset.yaml", epochs=1)
    # 假设每个epoch有一个mAP值（取最新的验证mAP）
    current_mAP = result["validation"]["mAP"][-1]
    mAP_history.append(current_mAP)
    current_epoch += 1

    fluctuation = max(mAP_history) - min(mAP_history)
    print(f"Epoch {current_epoch}: mAP = {current_mAP:.4f}, fluctuation = {fluctuation:.4f}")
    
    if fluctuation <= 0.03 and len(mAP_history) > 10:
        print("mAP fluctuation is within 3%, stopping phase 1 early.")
        break

for  idx, (name, module) in enumerate(modules_layer):
        if idx < freeze_layers:
            for param in module.parameters():
                param.requires_grad = True

print("Starting phase 2 training ...")
results_phase2 = model.train(data="rola/dataset.yaml", epochs=30, resume=True)
#start /B python "g:\dev\yolov5_distance_count\rola\freeze_finetuning.py" >> finetuning_freeze.log 2>&1
