import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ================== 1D 高斯函数 ==================
def gaussian(y, A, y0, sigma, C):
    """1D 高斯函数，用于局部峰值拟合"""
    return A * np.exp(-(y - y0)**2 / (2*sigma**2)) + C

# ================== 离子识别核心算法 ==================
def detect_ions_1d(image, blur_sigma=3.0, y_blur_sigma=1.5,
                   threshold_ratio=0.985, min_peak_prominence=50,
                   min_peak_distance=30, y_half_window=8):
    """
    核心算法：
    1. x 方向高斯模糊
    2. 阈值分割
    3. x 方向投影 + 峰值检测
    4. y 方向局部投影 + 峰值检测
    5. 局部高斯拟合精确提取离子中心
    输出：
        centers: [(x, y_fit), ...] 离子坐标列表
    """
    y_half_window = int(round(y_half_window))
    min_peak_distance = int(round(min_peak_distance))

    # x 方向模糊
    blur = cv2.GaussianBlur(image, (0,0), blur_sigma)

    # 阈值掩码
    mask = blur > np.percentile(blur, threshold_ratio*100)

    # x 方向投影
    projection_x = np.sum(blur * mask, axis=0).astype(np.float32)
    peaks_x, _ = find_peaks(projection_x, prominence=min_peak_prominence,
                            distance=min_peak_distance)

    centers = []
    h, w = image.shape
    x_half_window = 3

    for x0 in peaks_x:
        x0 = int(np.clip(x0, 0, w-1))
        x_min, x_max = max(0, x0 - x_half_window), min(w, x0 + x_half_window + 1)

        # y 方向投影
        y_projection = np.sum(blur[:, x_min:x_max], axis=1).astype(np.float32)

        if y_blur_sigma > 0:
            y_projection = cv2.GaussianBlur(y_projection.reshape(-1,1),
                                            ksize=(1,0), sigmaX=0, sigmaY=y_blur_sigma).ravel()

        peaks_y, _ = find_peaks(y_projection, prominence=min_peak_prominence/2,
                                distance=y_half_window)

        for y0 in peaks_y:
            y0 = int(round(y0))
            y_min, y_max = max(0, y0 - y_half_window), min(h, y0 + y_half_window)

            y_range = np.arange(y_min, y_max)
            signal = y_projection[y_min:y_max]

            if signal.sum() <= 0:
                continue

            # 局部高斯拟合
            p0 = [signal.max(), y0, y_half_window/2, np.median(signal)]
            try:
                _, y_fit, _, _ = curve_fit(gaussian, y_range, signal, p0=p0)[0][1],
                centers.append((x0, float(y_fit)))
            except RuntimeError:
                continue

    return centers