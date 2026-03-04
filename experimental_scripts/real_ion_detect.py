import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from mss import mss
from scipy.signal import find_peaks

class IonMonitorApp:
    def __init__(self):
        # ================= 参数 =================
        self.blur_sigma = 1.0           # 仅用于降噪
        self.threshold_ratio = 0.98     # 粗阈值（保守）
        self.min_peak_prominence = 50   # 一维峰显著性
        self.min_peak_distance = 8      # 离子最小间距（像素）
        self.y_half_window = 8          # y 拟合窗口半宽

        # ================= ROI =================
        print("🟢 请用鼠标框选离子区域...")
        self.roi = self.select_roi()
        print(f"✅ ROI: {self.roi}")

        # ================= GUI =================
        self.root = tk.Tk()
        self.root.title("Ion Chain Monitor (1D Projection Method)")
        self.root.geometry("900x780")

        self.label_count = tk.Label(self.root, text="离子数量: --", font=("Arial", 20))
        self.label_count.pack(pady=8)

        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        self.create_param_control("Blur Sigma", 0.1, 5.0, 0.1,
                                  self.blur_sigma, "blur_sigma")
        self.create_param_control("Threshold Ratio", 0.9, 0.999, 0.001,
                                  self.threshold_ratio, "threshold_ratio")
        self.create_param_control("Peak Prominence", 10, 300, 5,
                                  self.min_peak_prominence, "min_peak_prominence")
        self.create_param_control("Peak Distance", 3, 30, 1,
                                  self.min_peak_distance, "min_peak_distance")
        self.create_param_control("Y Half Window", 3, 30, 1,
                                  self.y_half_window, "y_half_window")

        self.running = True
        self.imgtk = None
        self.sct = mss()

        self.root.after(100, self.update_frame)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    # ================= ROI =================
    def select_roi(self):
        screenshot = np.array(mss().grab(mss().monitors[1]))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        r = cv2.selectROI("请选择离子监测区域", screenshot,
                           fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("请选择离子监测区域")
        x, y, w, h = r
        return {"left": int(x), "top": int(y),
                "width": int(w), "height": int(h)}

    # ================= GUI 控件 =================
    def create_param_control(self, label_text, min_val, max_val,
                             step, init_val, attr_name):
        frame = tk.Frame(self.root)
        frame.pack(pady=3)

        tk.Label(frame, text=label_text, width=22,
                 anchor="w").pack(side=tk.LEFT)

        scale = tk.Scale(frame, from_=min_val, to=max_val,
                         resolution=step, orient=tk.HORIZONTAL,
                         length=200)
        scale.set(init_val)
        scale.pack(side=tk.LEFT, padx=5)

        entry = tk.Entry(frame, width=8)
        entry.insert(0, str(init_val))
        entry.pack(side=tk.LEFT)

        def on_scale(val):
            setattr(self, attr_name, float(val))
            entry.delete(0, tk.END)
            entry.insert(0, val)

        def on_entry(event):
            try:
                val = float(entry.get())
                setattr(self, attr_name, val)
                scale.set(val)
            except ValueError:
                pass

        scale.config(command=on_scale)
        entry.bind("<Return>", on_entry)

    # ================= 捕获 =================
    def capture_screen(self):
        img = np.array(self.sct.grab(self.roi))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # ================= 核心算法 =================
    def detect_ions_1d(self, image):
        # --- 降噪 ---
        blur = cv2.GaussianBlur(image, (0, 0), self.blur_sigma)

        # --- 粗阈值（只用于选区域） ---
        thresh_val = np.percentile(blur, self.threshold_ratio * 100)
        mask = blur > thresh_val

        # --- y 方向投影 ---
        projection = np.sum(blur * mask, axis=0).astype(np.float32)

        # --- 一维峰检测 ---
        peaks, props = find_peaks(
            projection,
            prominence=self.min_peak_prominence,
            distance=self.min_peak_distance
        )

        centers = []
        h, w = image.shape

        for x0 in peaks:
            x0 = int(x0)
            x0 = np.clip(x0, 0, w - 1)

            # 在 x0 列附近取 y 分布
            col = blur[:, x0]
            y = np.arange(h)

            y_min = max(0, int(np.argmax(col) - self.y_half_window))
            y_max = min(h, y_min + 2 * self.y_half_window)

            weights = col[y_min:y_max].astype(np.float64)
            if weights.sum() <= 0:
                continue

            y_center = np.sum(y[y_min:y_max] * weights) / np.sum(weights)
            centers.append((x0, y_center))

        return centers, mask.astype(np.uint8) * 255

    # ================= 更新 =================
    def update_frame(self):
        img = self.capture_screen()
        centers, mask = self.detect_ions_1d(img)

        disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for i, (x, y) in enumerate(centers, start=1):
            cv2.circle(disp, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(disp, str(i), (int(x)+6, int(y)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

        cv2.putText(disp, f"Ions: {len(centers)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        self.label_count.config(text=f"离子数量: {len(centers)}")

        im = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        self.imgtk = ImageTk.PhotoImage(image=im)
        self.canvas.configure(image=self.imgtk)

        if self.running:
            self.root.after(200, self.update_frame)

    def on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    IonMonitorApp()
