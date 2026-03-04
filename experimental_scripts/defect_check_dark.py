import os
import cv2
import numpy as np
import pandas as pd
from tifffile import imread
from tkinter import Tk, filedialog, Text, Scrollbar, END
from natsort import natsorted

# ================== 配置 ==================
screen_res = 1280, 720
circle_radius = 6
square_size = 4
text_offset = (5, -5)

# ================== 控制显示模式 ==================
show_only_with_defects = True   # True: 只显示含缺陷的图像
show_only_with_dark = False     # True: 只显示含暗离子的图像
# 如果两个都为 True，则显示同时满足两个条件的图像

# ================== 选择文件夹 ==================
root = Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="选择 TIF 文件夹")
if not folder_path:
    raise SystemExit("未选择文件夹")

# ================== 加载离子坐标 CSV ==================
csv_path = os.path.join(folder_path, "ion_positions_batch.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("未找到 ion_positions_batch.csv 文件")
df = pd.read_csv(csv_path)
df["Filename"] = df["Filename"].astype(str).str.strip()

# ================== 加载缺陷结果 CSV ==================
result_path = os.path.join(folder_path, "zigzag_defect_analysis_full.csv")
if not os.path.exists(result_path):
    raise FileNotFoundError("未找到 zigzag_defect_analysis_full.csv 文件")
df_res = pd.read_csv(result_path)
df_res["Filename"] = df_res["Filename"].astype(str).str.strip()

# ================== 获取 TIF 文件 ==================
tif_files = natsorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
if len(tif_files) == 0:
    raise FileNotFoundError("未在指定文件夹中找到 TIF 文件")
print(f"共检测到 {len(tif_files)} 张图像")

# ================== 筛选显示图像 ==================
filtered_indices = []
for idx, img_name in enumerate(tif_files):
    frame_data = df_res[df_res["Filename"] == img_name]

    # 判断是否存在缺陷（但排除 dark_missing）
    has_defect = frame_data['DefectType'].apply(
        lambda x: isinstance(x, str) and x.strip() not in ['', 'dark_missing']
    ).any()

    # 判断是否存在暗离子（Predicted=True）
    has_dark = frame_data['Predicted'].any()

    # 按条件筛选
    if show_only_with_defects and not has_defect:
        continue
    if show_only_with_dark and not has_dark:
        continue

    filtered_indices.append(idx)

if not filtered_indices:
    raise SystemExit("没有符合条件的图像！")

# ================== 创建 OpenCV 窗口 ==================
win_name = "Ion Defect Check (←/→切换, ESC退出)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# ================== 创建 Tkinter 文本窗口 ==================
text_win = Tk()
text_win.title("Ion Defect Info")
text_win.geometry("400x600")
text_box = Text(text_win, wrap="none", font=("Consolas", 12))
scrollbar = Scrollbar(text_win, command=text_box.yview)
text_box.configure(yscrollcommand=scrollbar.set)
text_box.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ================== 显示图像函数 ==================
def show_frame(index):
    img_name = tif_files[index]
    img_path = os.path.join(folder_path, img_name)
    img = imread(img_path)
    img_disp = cv2.convertScaleAbs(img, alpha=255.0 / img.max())
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

    frame_data = df_res[df_res["Filename"] == img_name]
    if not frame_data.empty:
        centers_sorted = frame_data.sort_values("X").reset_index()

        # ===== 绘制离子类型 (圆圈) =====
        for _, row in centers_sorted.iterrows():
            x, y = row["X"], row["Y"]
            if np.isnan(x) or np.isnan(y):
                continue

            predicted = row["Predicted"]
            kink = row["Kink"]

            if kink:
                color = (0, 255, 255)
            elif predicted:
                color = (200, 200, 200)
            else:
                color = (0, 255, 0)

            cv2.circle(img_disp, (int(x), int(y)), circle_radius, color, 2)

        # ===== 绘制缺陷类型 (方形) =====
        for _, row in centers_sorted.iterrows():
            x, y = row["X"], row["Y"]
            if np.isnan(x) or np.isnan(y):
                continue

            defect = row["DefectType"]
            if defect == '':
                continue

            if defect == 'normal_defect':
                color = (0, 0, 255)
            elif defect == 'kink_defect':
                color = (0, 165, 255)
            elif defect == 'dark_normal_defect':
                color = (255, 0, 0)
            elif defect == 'dark_kink_defect':
                color = (255, 0, 255)
            elif defect == 'quench_boundary_defect':        # ⭐ 新增缺陷
                color = (0, 255, 255)                # 青色
            else:
                color = (0, 0, 0)

            top_left = (int(x - square_size/2), int(y - square_size/2))
            bottom_right = (int(x + square_size/2), int(y + square_size/2))
            cv2.rectangle(img_disp, top_left, bottom_right, color, 2)

        # ===== 绘制离子序号 =====
        for _, row in centers_sorted.iterrows():
            x, y = row["X"], row["Y"]
            if np.isnan(x) or np.isnan(y):
                continue
            ion_idx = row["IonIndex"]
            cv2.putText(
                img_disp,
                str(int(ion_idx)),
                (int(x) + text_offset[0], int(y) + text_offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # ===== 更新文本窗口 =====
        text_box.delete("1.0", END)
        text_box.insert(END, f"当前图像: {img_name}\n")
        text_box.insert(END, f"总离子数: {len(frame_data)}\n\n")
        text_box.insert(END, "=== 图例说明 ===\n\n")

        text_box.insert(END, "【离子类型 - 圆圈】\n")
        text_box.insert(END, "  🟢 绿色: 正常离子\n")
        text_box.insert(END, "  ⚪ 灰色: 预测插补离子\n")
        text_box.insert(END, "  🟡 黄色: Kink 离子\n\n")

        text_box.insert(END, "【缺陷类型 - 方形】\n")
        text_box.insert(END, "  🔴 红色: Normal Defect\n")
        text_box.insert(END, "  🟠 橙色: Kink Defect\n")
        text_box.insert(END, "  🔵 蓝色: Dark Normal Defect\n")
        text_box.insert(END, "  🟣 紫色: Dark Kink Defect\n")
        text_box.insert(END, "  🟡 青色: Quench Boundary Defect\n")  # ⭐ 新增
        text_box.update()

    # ===== 自动缩放显示 =====
    h, w = img_disp.shape[:2]
    scale = min(screen_res[0] / w, screen_res[1] / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_disp, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8)
    y_off = (screen_res[1] - new_h) // 2
    x_off = (screen_res[0] - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    cv2.imshow(win_name, canvas)

# ================== 主循环 ==================
current = 0
idx = filtered_indices[current]
show_frame(idx)

while True:
    key = cv2.waitKey(0)
    if key == 27:
        break
    elif key == 81 or key == ord('a'):
        current = (current - 1) % len(filtered_indices)
        idx = filtered_indices[current]
        show_frame(idx)
        text_win.update()
    elif key == 83 or key == ord('d'):
        current = (current + 1) % len(filtered_indices)
        idx = filtered_indices[current]
        show_frame(idx)
        text_win.update()

cv2.destroyAllWindows()
text_win.destroy()
