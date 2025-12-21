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

# Tham số GA1 (Elitism)
POP_SIZE = 40            # Kích thước quần thể
N_GEN = 120              # Số thế hệ
PC = 0.9                 # Xác suất lai ghép
PM = 0.12                # Xác suất đột biến
ELITE = 2                # Số lượng cá thể ưu tú giữ lại mỗi thế hệ
LAMBDA_SINR = 80.0       # Hệ số phạt vi phạm SINR
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
# 2. HÀM TÍNH TOÁN HIỆU NĂNG
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

def fitness_ga1(p):
    sinr = compute_sinr(p)
    snr_s = get_sensing_snr(p)
    penalty = LAMBDA_SINR * np.sum(np.maximum(0.0, gamma - sinr) ** 2)
    return snr_s - penalty

# =====================================================
# 3. TOÁN TỬ GA
# =====================================================
def repair_power(p):
    p = np.maximum(p, 0.0)
    for m in range(M):
        s = np.sum(p[m])
        if s > P_AP:
            p[m] *= P_AP / (s + 1e-9)
    return p

def crossover(a, b):
    if np.random.rand() < PC:
        alpha = np.random.rand()
        return alpha * a + (1 - alpha) * b
    return a.copy()

def mutation(p): # Tên hàm thống nhất là mutation
    if np.random.rand() < PM:
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
# 5. VÒNG LẶP TIẾN HÓA GA1
# =====================================================
def run_ga1():
    pop = [repair_power(np.random.rand(M, S)) for _ in range(POP_SIZE)]
    trace = np.zeros(N_GEN)
    best_ind_run = None
    best_fit_run = -np.inf

    for gen in range(N_GEN):
        fits = [fitness_ga1(ind) for ind in pop]
        best_idx = np.argmax(fits)
        
        if fits[best_idx] > best_fit_run:
            best_fit_run = fits[best_idx]
            best_ind_run = pop[best_idx].copy()

        trace[gen] = get_sensing_snr(pop[best_idx]) if fits[best_idx] > -1.0 else 0

        # --- ELITISM ---
        sorted_indices = np.argsort(fits)[::-1]
        new_pop = [pop[i].copy() for i in sorted_indices[:ELITE]] # Giữ lại 2 cá thể ưu tú

        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            child = mutation(crossover(p1, p2)) # Gọi đúng tên hàm mutation
            new_pop.append(repair_power(child))
            
        pop = new_pop
    return trace, best_ind_run, best_fit_run

if __name__ == "__main__":
    print(f"Đang thực hiện GA1 (Elitism) với {N_RUNS} lần chạy...")
    all_traces = np.zeros((N_RUNS, N_GEN))
    global_best_p = None
    global_best_fit = -np.inf

    for r in range(N_RUNS):
        trace, best_p, best_fit = run_ga1()
        all_traces[r, :] = trace
        if best_fit > global_best_fit:
            global_best_fit = best_fit
            global_best_p = best_p
    
    print_report(global_best_p, global_best_fit, "GA1 (ELITISM)")

    avg_trace = np.mean(all_traces, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(avg_trace, color='green', linewidth=2, label='GA1 (Elitism)')
    plt.xlabel("Generation")
    plt.ylabel("Average Best Sensing SNR")
    plt.title(f"Hội tụ của GA1 với cơ chế Elitism\n(Gamma* = {gamma:.4f}, Elite={ELITE})")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()
