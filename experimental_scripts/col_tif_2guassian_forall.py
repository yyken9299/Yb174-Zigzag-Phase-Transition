import os
import numpy as np
import pandas as pd
import cv2
from tifffile import imread
from natsort import natsorted
from tqdm import tqdm
from tkinter import Tk, filedialog
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
from datetime import datetime

warnings.simplefilter("ignore", OptimizeWarning)

# ================== 参数初始化（必须与 GUI 一致） ==================
params = {
    "blur_sigma": 3.0,
    "y_blur_sigma": 1.5,
    "threshold_ratio": 0.985,
    "min_peak_prominence": 50,
    "min_peak_distance": 30,
    "y_half_window": 8
}

# ================== 最大离子数阈值 ==================
max_ion_count = 25  # 超过此数量认为异常，跳过

# ================== 选择总文件夹（包含多个子文件夹） ==================
root = Tk()
root.withdraw()
root_dir = filedialog.askdirectory(title="选择包含多个 TIF 子文件夹的总目录")
if not root_dir:
    raise SystemExit("未选择总文件夹")

# 获取第一层子文件夹
subfolders = [
    os.path.join(root_dir, d)
    for d in natsorted(os.listdir(root_dir))
    if os.path.isdir(os.path.join(root_dir, d))
]

if not subfolders:
    raise RuntimeError("总目录下未找到任何子文件夹")

print(f"共检测到 {len(subfolders)} 个子文件夹")

# ================== 高斯函数 ==================
def gaussian(y, A, y0, sigma, C):
    return A * np.exp(-(y - y0) ** 2 / (2 * sigma ** 2)) + C

# ================== 高斯拟合 + 一维投影算法（完全复刻 GUI，仅加 y blur） ==================
def detect_ions_1d(image,
                   blur_sigma,
                   y_blur_sigma,
                   threshold_ratio,
                   min_peak_prominence,
                   min_peak_distance,
                   y_half_window):
    
    y_half_window = int(round(y_half_window))
    min_peak_distance = int(round(min_peak_distance))

    blur = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    thresh_val = np.percentile(blur, threshold_ratio * 100)
    mask = blur > thresh_val
    mask_uint8 = np.uint8(mask) * 255

    projection_x = np.sum(blur * mask, axis=0).astype(np.float32)
    peaks_x, _ = find_peaks(
        projection_x,
        prominence=min_peak_prominence,
        distance=min_peak_distance
    )

    centers = []
    h, w = image.shape
    x_half_window = 3

    for x0 in peaks_x:
        x0 = int(np.clip(x0, 0, w - 1))
        x_min = max(0, x0 - x_half_window)
        x_max = min(w, x0 + x_half_window + 1)

        y_projection = np.sum(blur[:, x_min:x_max], axis=1).astype(np.float32)

        if y_blur_sigma > 0:
            y_projection = cv2.GaussianBlur(
                y_projection.reshape(-1, 1),
                ksize=(1, 0),
                sigmaX=0,
                sigmaY=y_blur_sigma
            ).ravel()

        peaks_y, _ = find_peaks(
            y_projection,
            prominence=min_peak_prominence / 2,
            distance=y_half_window
        )

        for y0 in peaks_y:
            y0 = int(round(y0))
            y_min = max(0, y0 - y_half_window)
            y_max = min(h, y0 + y_half_window)

            y = np.arange(y_min, y_max)
            signal = y_projection[y_min:y_max]

            if signal.sum() <= 0:
                continue

            p0 = [
                signal.max(),
                y0,
                y_half_window / 2,
                np.median(signal)
            ]

            try:
                popt, _ = curve_fit(gaussian, y, signal, p0=p0)
                _, y_fit, _, _ = popt
                centers.append((x0, float(y_fit)))
            except RuntimeError:
                continue

    return centers, mask_uint8

# ================== 遍历每个子文件夹进行处理 ==================
for folder_path in subfolders:
    print(f"\n📂 正在处理文件夹: {folder_path}")

    tif_files = natsorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".tif")]
    )

    if not tif_files:
        print("⚠️  未找到 TIF 文件，跳过")
        continue

    print(f"  共检测到 {len(tif_files)} 张 TIF 图像")

    all_results = []
    valid_image_count = 0

    for fname in tqdm(tif_files, desc=os.path.basename(folder_path)):
        img = imread(os.path.join(folder_path, fname))

        img = (img.astype(np.float32) * (65535.0 / img.max())).clip(0, 65535).astype(np.uint16)
        img_gray = cv2.convertScaleAbs(img, alpha=255.0 / img.max())

        centers, _ = detect_ions_1d(
            img_gray,
            params["blur_sigma"],
            params["y_blur_sigma"],
            params["threshold_ratio"],
            params["min_peak_prominence"],
            params["min_peak_distance"],
            params["y_half_window"]
        )

        if len(centers) > max_ion_count:
            print(f"⚠️  {fname}: 离子数 {len(centers)} > {max_ion_count}，已跳过")
            continue

        valid_image_count += 1
        centers_sorted = sorted(centers, key=lambda c: c[0])

        for ion_id, (x, y) in enumerate(centers_sorted, start=1):
            all_results.append([fname, ion_id, x, y])

    # ================== 保存 CSV ==================
    df = pd.DataFrame(all_results, columns=["Filename", "IonIndex", "X", "Y"])
    out_csv = os.path.join(folder_path, "ion_positions_batch.csv")
    df.to_csv(out_csv, index=False)

    print(f"  ✅ 有效图像数量: {valid_image_count} / {len(tif_files)}")
    print(f"  📁 CSV 文件已保存至:\n  {out_csv}")

    # ================== 保存参数到单独文件（带时间戳） ==================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_rows = [[k, v] for k, v in params.items()]
    param_rows.append(["max_ion_count", max_ion_count])
    param_df = pd.DataFrame(param_rows, columns=["Parameter", "Value"])

    param_csv = os.path.join(folder_path, f"ion_detection_params_{timestamp}.csv")
    param_df.to_csv(param_csv, index=False)
    print(f"  🧾 参数文件已保存至:\n  {param_csv}")

print("\n🎉 所有子文件夹处理完成")
