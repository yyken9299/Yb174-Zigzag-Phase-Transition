import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import linregress
from multiprocessing import Pool
from tqdm import tqdm
import time
import os
import csv

# ================== 1. 物理引擎类 ==================
class ZigzagSimulation:
    def __init__(self, opts):
        # 物理常数 (SI)
        self.E_CHARGE = 1.602176634e-19
        self.EPS0 = 8.8541878128e-12
        self.AMU = 1.66053906660e-27
        self.M_YB = 171 * self.AMU
        self.KB = 1.380649e-23
        self.Q2_FACTOR = self.E_CHARGE**2 / (4 * np.pi * self.EPS0)

        # 核心参数：从 opts 获取
        self.N = opts.get('N', 30)
        self.omegaX_Hz = opts.get('omegaX_Hz', 109.6e3)
        self.tauQ_s = opts.get('tauQ_s', 1e-3)
        self.gamma_hat = opts.get('gamma_hat', 0.15)
        self.T_K = opts.get('T_K', 2e-7)
        
        # --- 自由设置频率逻辑 ---
        self.f_start = opts.get('f_start', 1.385e6) # 起始频率 Hz
        self.f_end = opts.get('f_end', 0.788e6)     # 终止频率 Hz
        self.hold_time_s = opts.get('hold_time_s', 100e-6) # 实验等待时间
        self.fixed_quench_steps = opts.get('fixed_steps', 60000)
        
        self.centerMaskFrac = opts.get('centerMaskFrac', 0.6)
        
        # 尺度计算
        self.omegaX = 2 * np.pi * self.omegaX_Hz
        self.l_scale = (self.Q2_FACTOR / (self.M_YB * self.omegaX**2))**(1/3)
        
        # 初始化位置 (求无量纲平衡位置)
        self.a0, self.pos_dimless = self.init_ion_positions()
        self.omega0 = np.sqrt(self.Q2_FACTOR / (self.M_YB * self.a0**3))
        
        # 将频率转换为无量纲 alpha
        self.alphaT0 = (2 * np.pi * self.f_start / self.omega0)**2
        self.alphaT1 = (2 * np.pi * self.f_end / self.omega0)**2
        self.alpha_x = (self.omegaX / self.omega0)**2
        
        # 时间步长控制 (由淬火时间决定)
        self.tauQ_hat = self.omega0 * self.tauQ_s
        self.dt = (2 * self.tauQ_hat) / self.fixed_quench_steps
        
        # 总步数 = 淬火步数 + 等待步数
        self.relax_steps = int(np.ceil((self.hold_time_s * self.omega0) / self.dt))
        self.nSteps = self.fixed_quench_steps + self.relax_steps
        
        # 热环境参数
        self.Ttilde = (self.KB * self.T_K) / (self.M_YB * (self.a0**2) * (self.omega0**2))

    def init_ion_positions(self):
        def equations(u):
            N = len(u); F = np.zeros(N)
            for j in range(N):
                mask = np.arange(N) != j
                diff = u[j] - u[mask]
                F[j] = u[j] - np.sum(1.0 / diff**2 * np.sign(diff))
            return F
        u0 = np.linspace(-(self.N-1)/2, (self.N-1)/2, self.N) * 0.5
        u_opt = fsolve(equations, u0)
        mid = self.N // 2
        a0 = np.abs(u_opt[mid] - u_opt[mid+1]) * self.l_scale
        pos = np.zeros((self.N, 2))
        pos[:, 0] = u_opt * (self.l_scale / a0)
        return a0, pos

    def get_forces(self, pos, alpha_t):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2) + 1e-18
        np.fill_diagonal(dist_sq, np.inf)
        dist_inv3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
        F_c = np.sum(diff * dist_inv3[:, :, np.newaxis], axis=1)
        F_trap = np.array([-self.alpha_x * pos[:, 0], -alpha_t * pos[:, 1]]).T
        return F_c + F_trap

    def run_single(self, seed=None):
        if seed is not None: np.random.seed(seed)
        pos, vel = self.pos_dimless.copy(), np.zeros((self.N, 2))
        c_ou = np.exp(-self.gamma_hat * self.dt)
        s_ou = np.sqrt(max(0, (1 - c_ou**2)) * self.Ttilde)
        
        f_steps = self.fixed_quench_steps
        dt = self.dt
        
        for step in range(self.nSteps):
            if step < f_steps:
                # 淬火阶段：频率线性下降
                tHat = -self.tauQ_hat + step * dt
                alpha_t = self.alphaT0 + (tHat + self.tauQ_hat)/(2*self.tauQ_hat) * (self.alphaT1 - self.alphaT0)
            else:
                # Holding 阶段：频率保持不变
                alpha_t = self.alphaT1 
                
            # BAOAB 积分器核心循环
            vel += 0.5 * dt * self.get_forces(pos, alpha_t)
            pos += 0.5 * dt * vel
            if self.gamma_hat > 0: 
                vel = c_ou * vel + s_ou * np.random.randn(self.N, 2)
            pos += 0.5 * dt * vel
            vel += 0.5 * dt * self.get_forces(pos, alpha_t)
            
        return self.count_kinks(pos)

    def count_kinks(self, pos):
        idx = np.argsort(pos[:, 0])
        ys, xs = pos[idx, 1], pos[idx, 0]
        # 截取中心区域排除边界效应
        mask = np.abs(xs) <= (self.centerMaskFrac * np.max(np.abs(xs)))
        ys_c = ys[mask]
        kinks = 0
        for i in range(len(ys_c)-1):
            # Zigzag 缺陷判定：相邻离子处于同侧且有显著偏移
            if ys_c[i] * ys_c[i+1] > 0 and np.abs(ys_c[i]) > 0.1:
                kinks += 1
        return kinks

# ================== 2. 并行包装 ==================
def worker(args):
    sim_obj, seed = args
    return sim_obj.run_single(seed=seed)

# ================== 3. 主程序入口 ==================
if __name__ == '__main__':
    # 全效率运行：使用所有物理核心
    use_threads = os.cpu_count()
    
    # 实验配置参数
    N_ions = 25
    num_realizations = 500 # 每组采样 200 次
    tauQ_values = np.logspace(-7.0, -4.5, 30) # 从 1us 到 1ms 扫描 40 个点
    
    # 自由设置物理频率范围 (对标实验)
    F_X = 109.6e3
    F_START = 1.385e6 
    F_END = 0.750e6
    HOLD_TIME = 100e-6
    FIX_STEP = 10000
    GAMMA = 0.15
    T = 2e-7
    
    avg_defects_list = []
    start_time = time.time()

    print(f"--- IKZM Scaling Law 自动扫描 (全效率模式) ---")
    print(f"并行核心: {use_threads} | 离子数: {N_ions}")
    print(f"扫描范围: {F_START/1e6:.3f} MHz -> {F_END/1e6:.3f} MHz")
    print(f"等待时间: {HOLD_TIME*1e6:.0f} us")

    # 

    # 循环扫描各个 tauQ
    for tq in tauQ_values:
        opts = {
            'N': N_ions,
            'tauQ_s': tq,
            'omegaX_Hz': F_X,
            'f_start': F_START,
            'f_end': F_END,
            'hold_time_s': HOLD_TIME, 
            'fixed_steps': FIX_STEP, 
            'gamma_hat': GAMMA, 
            'T_K': T
        }
        sim = ZigzagSimulation(opts)
        
        # 准备并行任务
        tasks = [(sim, i + int(time.time())) for i in range(num_realizations)]
        
        with Pool(processes=use_threads) as pool:
            # 使用 tqdm 监控当前采样点的进度
            results = list(tqdm(pool.imap(worker, tasks), total=num_realizations, 
                                desc=f"tauQ={tq*1e6:8.3g}us", leave=True))
        
        avg_defects_list.append(np.mean(results))

    # --- 4. 数据保存与结果分析 ---
    x_data = np.log10(tauQ_values*F_X)
    y_data = np.log10(avg_defects_list)
    
    # 线性拟合
    slope, intercept, r_val, _, _ = linregress(x_data, y_data)

    # 自动创建数据目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subdir = os.path.join(script_dir, "IKZM_Production_Data")
    if not os.path.exists(subdir): os.makedirs(subdir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(subdir, f"Scaling_N{N_ions}_{timestamp}.csv")
    
    # 写入 CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["# Simulation Metadata"])
        writer.writerow(["# N_ions", N_ions])
        writer.writerow(["# Photo_num", num_realizations])
        writer.writerow(["# F_X", F_X])
        writer.writerow(["# F_start_Hz", F_START])
        writer.writerow(["# F_end_Hz", F_END])
        writer.writerow(["# Hold_Time_s", HOLD_TIME])
        writer.writerow(["# fixed_steps", FIX_STEP])
        writer.writerow(["# gamma_hat", GAMMA])
        writer.writerow(["# T_K", T])
        writer.writerow(["# Fit_Slope_b", -slope])
        writer.writerow([])
        writer.writerow(["tauQ_s", "avg_defects", "log10_tauQ", "log10_density"])
        for i in range(len(tauQ_values)):
            writer.writerow([tauQ_values[i], avg_defects_list[i], x_data[i], y_data[i]])

    print(f"\n--- 任务完成 ---")
    print(f"总耗时: {time.time() - start_time:.2f} s")
    print(f"拟合缩放指数 b: {-slope:.4f}") # 通常汇报正值指数
    print(f"R^2 决定系数: {r_val**2:.4f}")
    print(f"原始数据已存至: {csv_path}")

    # 

    # 快速预览图
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, color='blue', label='Sim Data')
    plt.plot(x_data, slope * x_data + intercept, 'r--', label=f'Fit (b={-slope:.3f})')
    plt.xlabel(r'$\log_{10}(\tau_Q / s)$')
    plt.ylabel(r'$\log_{10}(d)$')
    plt.title(f'KZM Scaling (N={N_ions}, b={-slope:.3f})')
    plt.legend()
    plt.grid(True, ls=':', alpha=0.6)
    plt.savefig(os.path.join(subdir, f"Plot_N{N_ions}_{timestamp}.png"), dpi=300)
    plt.show()