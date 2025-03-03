from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
model = YOLO('yolov11n-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='self-seg.yaml', epochs=5, imgsz=640, workers=0,patience=10,batch=32)
# esults = model.train(data='self-seg.yaml', epochs=100)
