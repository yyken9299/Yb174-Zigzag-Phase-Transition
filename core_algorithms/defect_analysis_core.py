import numpy as np
from sklearn.decomposition import PCA

# ================== 核心缺陷识别算法 ==================
def analyze_zigzag_defects(coords, total_ions=25,
                           center_fraction=0.4,
                           gap_threshold=1.2,
                           axis_threshold=2):
    """
    输入:
        coords: Nx2 numpy array，离子二维坐标 [[x1,y1],[x2,y2],...]
        total_ions: 离子总数
        center_fraction: 用于中心区域暗离子预测
        gap_threshold: 检测中心区域缺陷的阈值
        axis_threshold: 判定 kink 的 Y 轴阈值
    输出:
        results: list of dict，每个离子包含:
            X_rot, Y_rot: PCA 旋转坐标
            Kink: 是否 kink 离子
            Predicted: 是否缺失预测离子
            Zigzag: A/B zigzag 构型
            DefectType: 缺陷类型
    """
    n_real = len(coords)
    if n_real < 3:
        return []

    # PCA 坐标旋转
    pca = PCA(n_components=2)
    coords_rot = pca.fit_transform(coords)
    x_rot, y_rot = coords_rot[:,0], coords_rot[:,1]

    # x 排序
    sort_idx = np.argsort(x_rot)
    x_sorted, y_sorted = x_rot[sort_idx], y_rot[sort_idx]

    # 暗离子预测（只保留算法核心，输出空预测即可）
    predicted_mask = np.array([False]*n_real)

    # kink 判定
    is_kink = np.array([abs(y)<axis_threshold for y in y_sorted])

    # zigzag 构型
    zigzag_labels = np.full(n_real, '', dtype=object)
    for idx in range(n_real):
        if idx % 2 == 0:
            zigzag_labels[idx] = 'A' if y_sorted[idx]>=0 else 'B'
        else:
            zigzag_labels[idx] = 'A' if y_sorted[idx]<0 else 'B'

    # 缺陷判定
    defect_type = np.array(['']*n_real, dtype=object)
    center_start = int(n_real*(1-center_fraction)/2)
    center_end   = int(n_real*(1+center_fraction)/2)
    real_indices = np.arange(n_real)

    # kink_defect
    for i in real_indices:
        if is_kink[i] and (center_start <= i < center_end):
            defect_type[i] = 'kink_defect'

    # normal_defect: 连续 zigzag 不匹配
    visited_pairs = set()
    for j in range(n_real-1):
        idx_prev, idx_next = real_indices[j], real_indices[j+1]
        if not (center_start <= idx_prev < center_end and center_start <= idx_next < center_end):
            continue
        if zigzag_labels[idx_prev]==zigzag_labels[idx_next]:
            continue
        if (idx_prev, idx_next) in visited_pairs:
            continue
        visited_pairs.add((idx_prev, idx_next))
        if defect_type[idx_prev]=='kink_defect' or defect_type[idx_next]=='kink_defect':
            continue
        defect_type[idx_next] = 'normal_defect'

    # 二次 quench 缺陷判定：连续缺陷块
    defect_indices = np.where(defect_type!='')[0]
    if len(defect_indices) > 0:
        blocks, block = [], [defect_indices[0]]
        for idx in defect_indices[1:]:
            if idx == block[-1]+1:
                block.append(idx)
            else:
                blocks.append(block)
                block = [idx]
        blocks.append(block)

        chain_center = (n_real-1)/2
        for block in blocks:
            if len(block)>=4:
                left, right = block[0], block[-1]
                keep = left if abs(left-chain_center)<abs(right-chain_center) else right
                for i in block:
                    defect_type[i] = 'quench_boundary_defect' if i==keep else ''

    # 逆 PCA 旋转回原始坐标
    coords_all = pca.inverse_transform(np.column_stack((x_sorted, y_sorted)))

    results = []
    for i in range(n_real):
        results.append({
            "X_rot": x_sorted[i],
            "Y_rot": y_sorted[i],
            "Kink": is_kink[i],
            "Predicted": predicted_mask[i],
            "Zigzag": zigzag_labels[i],
            "DefectType": defect_type[i],
            "X": coords_all[i,0],
            "Y": coords_all[i,1]
        })

    return results