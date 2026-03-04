import os
import cv2
import numpy as np
from tifffile import imread
from tkinter import Tk, Scale, Label, Button, HORIZONTAL, filedialog, Text, Scrollbar, RIGHT, Y, BOTH, Frame
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ========== 参数初始化 ==========
params = {
    "blur_sigma": 2.2,
    "y_blur_sigma": 1.5,          # ⭐ 新增：y 方向 blur
    "threshold_ratio": 0.985,
    "min_peak_prominence": 50,
    "min_peak_distance": 30,
    "y_half_window": 8
}

screen_res = 1280, 720  # 显示窗口最大尺寸

# ========== 选择文件夹 ==========
root = Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="选择 TIF 文件夹")
if not folder_path:
    raise SystemExit("未选择文件夹")
root.destroy()

tif_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
if not tif_files:
    raise FileNotFoundError("未找到 TIF 文件")

idx = 0

# ========== 高斯函数 ==========
def gaussian(y, A, y0, sigma, C):
    return A * np.exp(-(y - y0) ** 2 / (2 * sigma ** 2)) + C

# ========== 高斯拟合 + 一维投影算法 ==========
def detect_ions_1d(image,
                   blur_sigma,
                   y_blur_sigma,          # ⭐ 新增
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

    # x 方向投影
    projection_x = np.sum(blur * mask, axis=0).astype(np.float32)

    peaks_x, _ = find_peaks(
        projection_x,
        prominence=min_peak_prominence,
        distance=min_peak_distance
    )

    centers = []
    h, w = image.shape
    x_half_window = 3  # x 方向固定小窗口

    for x0 in peaks_x:
        x0 = int(np.clip(x0, 0, w - 1))
        x_min = max(0, x0 - x_half_window)
        x_max = min(w, x0 + x_half_window + 1)

        # x 小窗口内，对 y 投影
        y_projection = np.sum(blur[:, x_min:x_max], axis=1).astype(np.float32)

        # ⭐ 新增：y 方向 1D 高斯平滑（抑制尖锐噪点）
        if y_blur_sigma > 0:
            y_projection = cv2.GaussianBlur(
                y_projection.reshape(-1, 1),
                ksize=(1, 0),
                sigmaX=0,
                sigmaY=y_blur_sigma
            ).ravel()

        # y 方向找峰
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

    return mask_uint8, centers

# ========== GUI ==========
root = Tk()
root.title("TIF 参数调节器（高斯拟合）")

frame_left = Frame(root)
frame_left.pack(side="left", padx=10, pady=10)

frame_right = Frame(root)
frame_right.pack(side="right", fill=BOTH, expand=True, padx=10, pady=10)
scrollbar = Scrollbar(frame_right)
scrollbar.pack(side=RIGHT, fill=Y)
text_box = Text(frame_right, width=40, height=40, yscrollcommand=scrollbar.set)
text_box.pack(fill=BOTH, expand=True)
scrollbar.config(command=text_box.yview)

# ========== 显示图像 ==========
def show_frame():
    global idx
    img = imread(os.path.join(folder_path, tif_files[idx]))
    img = (img.astype(np.float32) * (65535.0 / img.max())).clip(0, 65535).astype(np.uint16)
    img_gray = cv2.convertScaleAbs(img, alpha=255.0 / img.max())
    img_gray_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    img_mask, centers = detect_ions_1d(
        img_gray,
        params["blur_sigma"],
        params["y_blur_sigma"],      # ⭐
        params["threshold_ratio"],
        params["min_peak_prominence"],
        params["min_peak_distance"],
        params["y_half_window"]
    )

    centers_int = [(int(x), int(y)) for (x, y) in centers]

    disp = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    for (cX, cY) in centers_int:
        cv2.circle(disp, (cX, cY), 4, (0, 0, 255), -1)
    cv2.putText(disp, f"Ions: {len(centers_int)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    h, w = img_gray_color.shape[:2]
    canvas_disp = np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8)
    scale = min(screen_res[0]/w, (screen_res[1]/2)/h, 1.0)
    new_w, new_h = int(w*scale), int(h*scale)

    img_gray_resized = cv2.resize(img_gray_color, (new_w, new_h))
    disp_resized = cv2.resize(disp, (new_w, new_h))

    x_off = (screen_res[0] - new_w)//2
    y_off_top = (screen_res[1]//2 - new_h)//2
    y_off_bottom = screen_res[1]//2 + y_off_top

    canvas_disp[y_off_top:y_off_top+new_h, x_off:x_off+new_w] = img_gray_resized
    canvas_disp[y_off_bottom:y_off_bottom+new_h, x_off:x_off+new_w] = disp_resized

    cv2.putText(canvas_disp, "Original", (x_off+10, y_off_top+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas_disp, "Processed", (x_off+10, y_off_bottom+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("TIF Parameter Tuning", canvas_disp)

    text_box.delete("1.0", "end")
    text_box.insert("end", f"当前文件: {tif_files[idx]}\n")
    text_box.insert("end", f"总离子数: {len(centers_int)}\n\n")
    for i, (cX, cY) in enumerate(centers_int, start=1):
        text_box.insert("end", f"Ion {i}: X={cX}, Y={cY}\n")

# ========== 滑条 ==========
def add_slider(label, key, from_, to_, resolution):
    def on_change(val):
        params[key] = float(val)
        show_frame()
    Label(frame_left, text=label).pack()
    scale = Scale(frame_left, from_=from_, to=to_, resolution=resolution,
                  orient=HORIZONTAL, length=300, command=on_change)
    scale.set(params[key])
    scale.pack()

add_slider("Blur Sigma", "blur_sigma", 0.1, 5.0, 0.1)
add_slider("Y Blur Sigma", "y_blur_sigma", 0.0, 5.0, 0.2)   # ⭐ 新增
add_slider("Threshold Ratio", "threshold_ratio", 0.9, 0.999, 0.001)
add_slider("Peak Prominence", "min_peak_prominence", 10, 300, 5)
add_slider("Peak Distance", "min_peak_distance", 3, 30, 1)
add_slider("Y Half Window", "y_half_window", 3, 30, 1)

# ========== 按钮 ==========
def prev_frame():
    global idx
    idx = (idx - 1) % len(tif_files)
    show_frame()

def next_frame():
    global idx
    idx = (idx + 1) % len(tif_files)
    show_frame()

Button(frame_left, text="上一帧", command=prev_frame).pack(pady=2)
Button(frame_left, text="下一帧", command=next_frame).pack(pady=2)
Button(frame_left, text="退出", command=lambda: (cv2.destroyAllWindows(), root.destroy())).pack(pady=2)

show_frame()
root.mainloop()
