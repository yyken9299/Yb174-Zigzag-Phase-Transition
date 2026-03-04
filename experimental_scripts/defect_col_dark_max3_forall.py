import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tkinter import Tk, filedialog
from tqdm import tqdm
from natsort import natsorted

# ================== 选择总文件夹（包含多个子文件夹） ==================
root = Tk()
root.withdraw()
root_dir = filedialog.askdirectory(title="选择包含多个子文件夹的总目录")
if not root_dir:
    raise SystemExit("未选择总文件夹")

# 只遍历第一层子文件夹
subfolders = [
    os.path.join(root_dir, d)
    for d in natsorted(os.listdir(root_dir))
    if os.path.isdir(os.path.join(root_dir, d))
]

if not subfolders:
    raise RuntimeError("总目录下未找到任何子文件夹")

# ================== 参数设置 ==================
axis_threshold = 2
total_ions = 25
center_fraction = 0.4
gap_threshold = 1.2

# ============================================================
# 逐个子文件夹处理（其余逻辑完全不变）
# ============================================================
for folder_path in subfolders:

    print(f"\n📂 正在处理文件夹: {folder_path}")

    csv_path = os.path.join(folder_path, "ion_positions_batch.csv")
    if not os.path.exists(csv_path):
        print("⚠️  未找到 ion_positions_batch.csv，跳过")
        continue

    # ================== 读取离子位置 ==================
    df = pd.read_csv(csv_path)
    df["Filename"] = df["Filename"].astype(str).str.strip()

    # ================== 按图像分组处理 ==================
    all_results = []

    for img_name, group in tqdm(df.groupby("Filename"),
                                desc=os.path.basename(folder_path),
                                ncols=100):
        coords = group[["X", "Y"]].values
        n_real = len(coords)
        if n_real < 3:
            continue

        pca = PCA(n_components=2)
        pca.fit(coords)
        coords_rot = pca.transform(coords)
        x_rot, y_rot = coords_rot[:, 0], coords_rot[:, 1]

        sort_idx = np.argsort(x_rot)
        x_sorted = x_rot[sort_idx]
        y_sorted = y_rot[sort_idx]

        # ================== 暗离子预测（中心区域） ==================
        predicted_coords = []
        n_missing = total_ions - n_real

        if n_missing > 0:
            n = len(x_sorted)
            margin = (1 - center_fraction) / 2
            center_start = int(n * margin)
            center_end = int(n * (1 - margin))
            center_indices = np.arange(center_start, center_end)

            avg_dist_center = (
                np.mean(np.diff(x_sorted[center_indices]))
                if len(center_indices) > 1
                else np.mean(np.diff(x_sorted))
            )

            for i in center_indices[:-1]:
                curr_dist = x_sorted[i+1] - x_sorted[i]
                if curr_dist > gap_threshold * avg_dist_center:
                    num_gaps = max(int(round(curr_dist / avg_dist_center)) - 1, 1)
                    for j in range(num_gaps):
                        x_pred = x_sorted[i] + (j+1)*(x_sorted[i+1]-x_sorted[i])/(num_gaps+1)
                        predicted_coords.append([x_pred, np.nan])

            predicted_coords = predicted_coords[:n_missing]

        # ================== 合并真实和预测离子 ==================
        real_coords = np.column_stack((x_sorted, y_sorted))
        if len(predicted_coords) > 0:
            combined_coords = np.vstack([real_coords, np.array(predicted_coords)])
            predicted_mask = np.array([False]*n_real + [True]*len(predicted_coords))
        else:
            combined_coords = real_coords
            predicted_mask = np.array([False]*n_real)

        sort_all = np.argsort(combined_coords[:, 0])
        combined_coords = combined_coords[sort_all]
        predicted_mask = predicted_mask[sort_all]
        x_all, y_all = combined_coords[:, 0], combined_coords[:, 1]
        n_total = len(x_all)

        # ================== kink 离子标记 ==================
        is_kink = np.array([
            False if pred else abs(y) < axis_threshold
            for y, pred in zip(y_all, predicted_mask)
        ])

        # ================== zigzag 构型识别（仅对真实离子） ==================
        zigzag_labels = np.full(n_total, '', dtype=object)
        real_non_kink_idx = np.where(~is_kink & ~predicted_mask)[0]

        for idx in real_non_kink_idx:
            if idx % 2 == 0:
                zigzag_labels[idx] = 'A' if y_all[idx] >= 0 else 'B'
            else:
                zigzag_labels[idx] = 'A' if y_all[idx] < 0 else 'B'

        # ================== 缺陷判定 ==================
        defect_type = [''] * n_total
        center_start_idx = int(n_total*(1-center_fraction)/2)
        center_end_idx = int(n_total*(1+center_fraction)/2)
        real_indices = np.where(~predicted_mask)[0]

        for i in real_indices:
            if is_kink[i] and (center_start_idx <= i < center_end_idx):
                defect_type[i] = 'kink_defect'

        visited_pairs = set()
        for j in range(len(real_indices)-1):
            idx_prev_real = real_indices[j]
            idx_next_real = real_indices[j+1]

            if not (center_start_idx <= idx_prev_real < center_end_idx and
                    center_start_idx <= idx_next_real < center_end_idx):
                continue

            if zigzag_labels[idx_prev_real] == zigzag_labels[idx_next_real]:
                continue

            if (idx_prev_real, idx_next_real) in visited_pairs:
                continue
            visited_pairs.add((idx_prev_real, idx_next_real))

            if (defect_type[idx_prev_real] == 'kink_defect' or
                defect_type[idx_next_real] == 'kink_defect'):
                continue

            defect_type[idx_next_real] = 'normal_defect'

        # ================== 二次 quench 判定 ==================
        valid_defect_mask = np.array([
            (dt != '') and (dt != 'dark_missing')
            for dt in defect_type
        ])

        defect_indices = np.where(valid_defect_mask)[0]

        if len(defect_indices) > 0:
            blocks = []
            block = [defect_indices[0]]

            for idx in defect_indices[1:]:
                if idx == block[-1] + 1:
                    block.append(idx)
                else:
                    blocks.append(block)
                    block = [idx]
            blocks.append(block)

            chain_center = (n_total - 1) / 2

            for block in blocks:
                if len(block) >= 4:
                    left, right = block[0], block[-1]
                    keep = left if abs(left - chain_center) < abs(right - chain_center) else right

                    for i in block:
                        defect_type[i] = 'quench_boundary_defect' if i == keep else ''

        # ================== 坐标逆旋转 ==================
        coords_all_rot = np.column_stack((x_all, y_all))
        coords_all = pca.inverse_transform(coords_all_rot)

        for idx in range(n_total):
            all_results.append({
                "Filename": img_name,
                "IonIndex": idx + 1,
                "X": coords_all[idx, 0],
                "Y": coords_all[idx, 1],
                "X_rot": x_all[idx],
                "Y_rot": y_all[idx],
                "Kink": is_kink[idx],
                "Predicted": predicted_mask[idx],
                "Zigzag": zigzag_labels[idx],
                "DefectType": defect_type[idx]
            })

    # ================== 保存结果 ==================
    result_df = pd.DataFrame(all_results)

    out_path = os.path.join(folder_path, "zigzag_defect_analysis_full.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ================== 缺陷统计 ==================
    filtered_df = result_df[result_df["DefectType"] != "dark_missing"]
    summary = filtered_df["DefectType"].value_counts().to_dict()

    summary_path = os.path.join(folder_path, "zigzag_defect_summary_full.csv")
    pd.DataFrame({
        'DefectType': list(summary.keys()),
        'Count': list(summary.values())
    }).to_csv(summary_path, index=False, encoding="utf-8-sig")

    # ================== 总缺陷密度 ==================
    num_images = len(df["Filename"].unique())
    valid_defects = result_df[
        (result_df["DefectType"] != "") &
        (result_df["DefectType"] != "dark_missing")
    ]
    total_defects = valid_defects.shape[0]
    defect_density = total_defects / num_images if num_images > 0 else np.nan

    print("✅ 分析完成（已加入二次 quench 缺陷）")
    print(f"📁 详细结果文件：{out_path}")
    print(f"📄 统计汇总文件：{summary_path}")
    print(f"📊 总缺陷密度 = {defect_density:.4f}")

print("\n🎉 所有子文件夹处理完成")
