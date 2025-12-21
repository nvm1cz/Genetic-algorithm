import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. THIẾT LẬP HỆ THỐNG (M=3, U=2, Q=1)
# =====================================================
np.random.seed(1)
M, U, Q = 3, 2, 1        # Số AP, Số người dùng, Số luồng cảm biến
S = U + Q                # Tổng số luồng tín hiệu
P_AP = 1.0               # Công suất tối đa mỗi AP
noise_u = 0.1            # Nhiễu tại người dùng
gamma = 0.35446747231755743  # Ngưỡng SINR tối thiểu (Gamma*)

# Tham số GA3 (Adaptive Mutation)
POP_SIZE = 40            # Kích thước quần thể
N_GEN = 120              # Số thế hệ
PC = 0.9                 # Xác suất lai ghép
PM_BASE = 0.12           # Xác suất đột biến cơ sở ban đầu
ELITE = 2                # Số lượng cá thể ưu tú
N_RUNS = 10              # Chạy 10 lần lấy trung bình

# Khởi tạo kênh truyền và Beamforming cố định (MRT)
def init_system():
    rng = np.random.default_rng(1)
    h = rng.standard_normal((U, M)) + 1j * rng.standard_normal((U, M))
    fbar_comm = rng.standard_normal((M, U)) + 1j * rng.standard_normal((M, U))
    fbar_comm /= (np.linalg.norm(fbar_comm, axis=1, keepdims=True) + 1e-9)
    rng_s = np.random.default_rng(2)
    fbar_sens = rng_s.standard_normal((M, Q)) + 1j * rng_s.standard_normal((M, Q))
    fbar_sens /= (np.linalg.norm(fbar_sens, axis=1, keepdims=True) + 1e-9)
    f_bar = np.concatenate([fbar_comm, fbar_sens], axis=1)
    return h, f_bar

H_CHANNELS, F_BAR = init_system()

# =====================================================
# 2. HÀM TÍNH TOÁN HIỆU NĂNG & FITNESS
# =====================================================
def compute_sinr(p):
    sinr = np.zeros(U)
    for u in range(U):
        sig, interf = 0.0, 0.0
        for m in range(M):
            sig += p[m, u] * abs(np.conj(H_CHANNELS[u, m]) * F_BAR[m, u])**2
            for s in range(S):
                if s != u:
                    interf += p[m, s] * abs(np.conj(H_CHANNELS[u, m]) * F_BAR[m, s])**2
        sinr[u] = sig / (interf + noise_u)
    return sinr

def get_sensing_snr(p):
    return float(np.sum(p[:, U:]))

def fitness_ga3(p):
    # Lexicographic: Ưu tiên tính khả thi tuyệt đối
    sinr = compute_sinr(p)
    if np.any(sinr < gamma):
        return -1e6 - np.sum(np.maximum(0.0, gamma - sinr)) * 1e3
    return get_sensing_snr(p)

# =====================================================
# 3. TOÁN TỬ GA & REPAIR
# =====================================================
def repair_power(p):
    p = np.maximum(p, 0.0)
    for m in range(M):
        s = np.sum(p[m])
        if s > P_AP:
            p[m] *= P_AP / (s + 1e-9)
    return p

def repair_feasible(p):
    p = repair_power(p)
    if np.any(compute_sinr(p) < gamma):
        p[:, U:] *= 0.5 # Giảm Sensing để cứu Communication
    return repair_power(p)

def crossover(a, b):
    if np.random.rand() < PC:
        alpha = np.random.rand()
        return alpha * a + (1 - alpha) * b
    return a.copy()

def mutation(p, pm_rate): # Nhận xác suất đột biến thích nghi
    if np.random.rand() < pm_rate:
        m, s = np.random.randint(M), np.random.randint(S)
        p[m, s] += 0.08 * np.random.randn()
    return p

def tournament_selection(pop, fits, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax([fits[i] for i in idx])]]

# =====================================================
# 4. HÀM IN BÁO CÁO
# =====================================================
def print_report(best_p, best_fit, title):
    sinr = compute_sinr(best_p)
    snr_s = get_sensing_snr(best_p)
    is_feasible = "Có" if np.all(sinr >= gamma - 1e-6) else "Không"
    
    print("\n" + "="*55)
    print(f"BÁO CÁO KẾT QUẢ TỐI ƯU: {title}")
    print("="*55)
    print(f"{'Chỉ tiêu':<30} | {'Giá trị':<15}")
    print("-" * 55)
    print(f"{'Best fitness':<30} | {best_fit:.4f}")
    print(f"{'Best sensing SNR':<30} | {snr_s:.4f}")
    print(f"{'SINR user 1':<30} | {sinr[0]:.4f}")
    print(f"{'SINR user 2':<30} | {sinr[1]:.4f}")
    print(f"{'SINR nhỏ nhất':<30} | {np.min(sinr):.4f}")
    print(f"{'Ngưỡng SINR (gamma)':<30} | {gamma:.4f}")
    print(f"{'Tính khả thi (Feasible)':<30} | {is_feasible}")
    print("-" * 55)
    print("\nMa trận phân bổ công suất tương ứng:")
    print(f"{'AP':<5} | {'User 1':<10} | {'User 2':<10} | {'Sensing':<10} | {'Tổng':<10}")
    print("-" * 55)
    for m in range(M):
        total = np.sum(best_p[m])
        print(f"AP{m+1:<3} | {best_p[m,0]:<10.4f} | {best_p[m,1]:<10.4f} | {best_p[m,2]:<10.4f} | {total:<10.4f}")
    print("="*55 + "\n")

# =====================================================
# 5. VÒNG LẶP TIẾN HÓA GA3 (ADAPTIVE)
# =====================================================
def run_ga3():
    pop = [repair_power(np.random.rand(M, S)) for _ in range(POP_SIZE)]
    trace = np.zeros(N_GEN)
    best_ind_run = None
    best_fit_run = -np.inf
    
    pm = PM_BASE
    no_improve = 0

    for gen in range(N_GEN):
        fits = [fitness_ga3(ind) for ind in pop]
        best_idx = np.argmax(fits)
        
        # Logic thích nghi PM
        if fits[best_idx] > best_fit_run + 1e-6:
            best_fit_run = fits[best_idx]
            best_ind_run = pop[best_idx].copy()
            no_improve = 0
            pm = max(0.05, pm - 0.01) # Giảm PM khi đang tiến triển tốt
        else:
            no_improve += 1
            if no_improve > 5:
                pm = min(0.6, pm + 0.05) # Tăng PM để thoát cực trị địa phương
        
        trace[gen] = get_sensing_snr(pop[best_idx]) if fits[best_idx] > -1.0 else 0

        # Elitism
        sorted_indices = np.argsort(fits)[::-1]
        new_pop = [pop[i].copy() for i in sorted_indices[:ELITE]]

        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            child = mutation(crossover(p1, p2), pm) # Sử dụng PM thích nghi
            new_pop.append(repair_feasible(child))
            
        pop = new_pop
    return trace, best_ind_run, best_fit_run

if __name__ == "__main__":
    print(f"Đang chạy GA3 (Adaptive Mutation) trung bình {N_RUNS} lần...")
    all_traces = np.zeros((N_RUNS, N_GEN))
    global_best_p = None
    global_best_fit = -np.inf

    for r in range(N_RUNS):
        trace, best_p, best_fit = run_ga3()
        all_traces[r, :] = trace
        if best_fit > global_best_fit:
            global_best_fit = best_fit
            global_best_p = best_p
    
    print_report(global_best_p, global_best_fit, "GA3 (ADAPTIVE)")

    avg_trace = np.mean(all_traces, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(avg_trace, color='red', linewidth=2, label='GA3 (Adaptive)')
    plt.xlabel("Generation")
    plt.ylabel("Average Best Sensing SNR")
    plt.title(f"Hội tụ của GA3 - Adaptive Mutation GA\n(Gamma* = {gamma:.4f})")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()
