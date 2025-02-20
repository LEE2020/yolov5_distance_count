import cv2
import numpy as np
import glob

def capture_images(num_images=20, delay=1):
    """
    从左右摄像头拍摄照片并保存。
    
    参数:
        num_images (int): 需要拍摄的照片数量。
        delay (int): 每次拍摄的间隔时间（秒）。
    """
    # 打开左右摄像头
    cap_left = cv2.VideoCapture(3)  # 左摄像头索引
    cap_right = cv2.VideoCapture(0)  # 右摄像头索引

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("无法打开摄像头！")
        return

    for i in range(num_images):
        print(f"拍摄第 {i+1} 张照片...")
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if ret_left and ret_right:
            # 保存左右摄像头的照片
            cv2.imwrite(f'left_{i+1}.jpg', frame_left)
            cv2.imwrite(f'right_{i+1}.jpg', frame_right)
            print(f"照片 left_{i+1}.jpg 和 right_{i+1}.jpg 已保存。")
        else:
            print("拍摄失败！")

        # 等待一段时间
        cv2.waitKey(delay * 1000)

    # 释放摄像头
    cap_left.release()
    cap_right.release()
    print("照片拍摄完成。")

def stereo_calibrate():
    """
    立体相机标定。
    """
    # 标定板参数
    CHECKERBOARD = (10,10) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备对象点
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # 存储所有图像的对象点和图像点
    objpoints = []  # 3D点
    imgpoints_left = []  # 左相机2D点
    imgpoints_right = []  # 右相机2D点

    # 读取图像
    images_left = glob.glob('left_*.jpg')
    images_right = glob.glob('right_*.jpg')

    for left_img, right_img in zip(images_left, images_right):
        img_left = cv2.imread(left_img)
        img_right = cv2.imread(right_img)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 查找角点
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

        if ret_left and ret_right:
            objpoints.append(objp)

            # 亚像素级角点检测
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners_left)

            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            imgpoints_right.append(corners_right)

    # 立体相机标定
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        None, None, None, None,
        gray_left.shape[::-1],
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    return mtx_left, dist_left, mtx_right, dist_right, R, T

# 主程序
if __name__ == "__main__":
    # 拍摄照片
    capture_images(num_images=20, delay=2)  # 拍摄10张照片，间隔2秒

    # 进行立体相机标定
    mtx_left, dist_left, mtx_right, dist_right, R, T = stereo_calibrate()

    # 输出标定结果
    print('左边的标定矩阵：', mtx_left)
    print('左边的畸变系数：', dist_left)
    print('右边的标定矩阵：', mtx_right)
    print('右边的畸变系数：', dist_right)
    print('旋转矩阵:', R)
    print('平移向量:', T)