import os
import cv2
import numpy as np
import pandas as pd
from tifffile import imread
from natsort import natsorted
from tqdm import tqdm
from tkinter import Tk, filedialog

# ================== 配置 ==================
root = Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="选择 TIF 文件夹")
if not folder_path:
    raise SystemExit("未选择文件夹")

screen_res = 1280, 720      # 显示窗口最大尺寸
circle_radius = 5           # 标注离子半径
text_offset = (5, -5)       # 标注文字偏移

# ================== 读取 CSV ==================
csv_path = os.path.join(folder_path, "ion_positions_batch.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("未找到 CSV 文件，请先运行批量处理")
df = pd.read_csv(csv_path)
df["Filename"] = df["Filename"].astype(str).str.strip()

# ================== 获取 TIF 文件 ==================
tif_files = natsorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
if len(tif_files) == 0:
    raise FileNotFoundError("未找到 TIF 文件")
print(f"共检测到 {len(tif_files)} 张 TIF 图像")

# ================== 可视化函数 ==================
show_marks = True  # 新增：标记点显示开关

def show_frame(index):
    img_name = tif_files[index]
    img_path = os.path.join(folder_path, img_name)
    img = imread(img_path)
    img_disp = cv2.convertScaleAbs(img, alpha=255.0 / img.max())
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

    # 找出当前图像对应的离子数据
    frame_data = df[df["Filename"] == img_name].drop_duplicates(subset=["X","Y"])
    centers = [(row["X"], row["Y"]) for _, row in frame_data.iterrows()]

    if show_marks:  # ✅ 仅在开启时绘制标记点
        centers_sorted = sorted(enumerate(centers), key=lambda x: x[1][0])
        for new_idx, (orig_idx, (x, y)) in enumerate(centers_sorted, start=1):
            cv2.circle(img_disp, (int(x), int(y)), circle_radius, (0, 0, 255), 2)
            cv2.putText(img_disp, str(new_idx), (int(x)+text_offset[0], int(y)+text_offset[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(img_disp, f"Image: {img_name}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img_disp, f"CSV: {len(centers)} ions", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 缩放适应窗口
    h, w = img_disp.shape[:2]
    scale = min(screen_res[0]/w, screen_res[1]/h, 1.0)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img_disp, (new_w,new_h), interpolation=cv2.INTER_AREA)

    # 黑底居中显示
    canvas = np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8)
    y_off = (screen_res[1]-new_h)//2
    x_off = (screen_res[0]-new_w)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    cv2.imshow("CSV Verification", canvas)

# ================== 主循环 ==================
idx = 0
show_frame(idx)

print("使用 ←/→ 翻页, M键切换标记显示, ESC退出")
while True:
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    elif key == 81 or key == ord('a'):  # ←
        idx = (idx-1) % len(tif_files)
        show_frame(idx)
    elif key == 83 or key == ord('d'):  # →
        idx = (idx+1) % len(tif_files)
        show_frame(idx)
    elif key in (ord('m'), ord('M')):  # ✅ M 键开关标记点
        show_marks = not show_marks
        print(f"标记显示：{'开' if show_marks else '关'}")
        show_frame(idx)

cv2.destroyAllWindows()
