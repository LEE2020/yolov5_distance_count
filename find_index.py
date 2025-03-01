import cv2
cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
#cap2 = cv2.VideoCapture(1, cv2.CAP_V4L2)

# 检查 VideoCapture 是否成功打开
if not cap1.isOpened():
    print("错误：无法打开摄像头cap1")
    exit()
else:
    print("摄像头成功打开。")


cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap1.read()
# 尝试设置摄像头分辨率为2560×720

if not ret or frame is None or frame.size == 0:
    print("错误：无法从摄像头读取到有效帧。")
else:
    height, width = frame.shape[:2]
    print("摄像头图像分辨率：{} x {}".format(width, height))
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
