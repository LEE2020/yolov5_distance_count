import cv2

from ultralytics import solutions


def calculate_object_distance(box, known_width=50, focal_length=700):
    """
    通过检测框计算物体距离
    :param box: 检测框坐标 [x1, y1, x2, y2]
    :param known_width: 物体在现实世界中的宽度（单位：厘米），需要根据具体物体调整
    :param focal_length: 相机焦距（单位：像素），需要通过标定获得
    :return: 距离（单位：厘米）
    """
    x1, y1, x2, y2 = box
    pixel_width = x2 - x1
    if pixel_width <= 0:
        return None
    distance = (known_width * focal_length) / pixel_width
    return distance

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init distance-calculation obj
distance = solutions.DistanceCalculation(model="yolo11n.pt", show=True)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = distance.calculate(im0)
    # 增加的部分
    # 如果检测结果存在，则对每个检测框计算距离并标注
    if hasattr(distance, "results") and len(distance.results) > 0:
        for result in distance.results:
            # 这里假设 result.boxes.xyxy 为检测框列表，每个框格式 [x1, y1, x2, y2]
            for box in result.boxes.xyxy.tolist():
                dist = calculate_object_distance(box)
                if dist is not None:
                    # 在检测框上方标注距离（单位：cm），字体大小和颜色可自行调整
                    cv2.putText(im0, f"{dist:.1f}cm", (int(box[0]), int(box[1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 绘制检测框（可选）
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    
    
    
    
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()