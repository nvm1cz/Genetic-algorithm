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

# Tham số GA
POP_SIZE = 40            # Kích thước quần thể
N_GEN = 120              # Số thế hệ
PC = 0.9                 # Xác suất lai ghép
PM_BASE = 0.12           # Xác suất đột biến cơ sở
ELITE = 0                # GA0 không sử dụng Elitism
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

def fitness_func(p, variant='penalty'):
    sinr = compute_sinr(p)
    snr_s = get_sensing_snr(p)
    if variant == 'penalty':
        penalty = LAMBDA_SINR * np.sum(np.maximum(0.0, gamma - sinr) ** 2)
        return snr_s - penalty
    else: 
        if np.any(sinr < gamma):
            return -1e6 - np.sum(np.maximum(0.0, gamma - sinr)) * 1e3
        return snr_s

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

def mutate(p, pm_rate=PM_BASE):
    if np.random.rand() < pm_rate:
        m, s = np.random.randint(M), np.random.randint(S)
        p[m, s] += 0.08 * np.random.randn()
    return p

def tournament_selection(pop, fits, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax([fits[i] for i in idx])]]

def print_report(best_p, best_fit, title):
    sinr = compute_sinr(best_p)
    snr_s = get_sensing_snr(best_p)
    is_feasible = "Có" if np.all(sinr >= gamma - 1e-6) else "Không"
    
    print("-" * 50)
    print(f"BÁO CÁO KẾT QUẢ TỐI ƯU: {title}")
    print("-" * 50)
    print(f"{'Chỉ tiêu':<30} | {'Giá trị':<15}")
    print("-" * 50)
    print(f"{'Best fitness':<30} | {best_fit:.4f}")
    print(f"{'Best sensing SNR':<30} | {snr_s:.4f}")
    print(f"{'SINR user 1':<30} | {sinr[0]:.4f}")
    print(f"{'SINR user 2':<30} | {sinr[1]:.4f}")
    print(f"{'SINR nhỏ nhất':<30} | {np.min(sinr):.4f}")
    print(f"{'Ngưỡng SINR (gamma)':<30} | {gamma:.4f}")
    print(f"{'Tính khả thi (Feasible)':<30} | {is_feasible}")
    print("-" * 50)
    print("\nMa trận phân bổ công suất tương ứng:")
    print(f"{'AP':<5} | {'User 1':<10} | {'User 2':<10} | {'Sensing':<10} | {'Tổng':<10}")
    print("-" * 50)
    for m in range(M):
        total = np.sum(best_p[m])
        print(f"AP{m+1:<3} | {best_p[m,0]:<10.4f} | {best_p[m,1]:<10.4f} | {best_p[m,2]:<10.4f} | {total:<10.4f}")
    print("-" * 50)

def run_ga():
    pop = [repair_power(np.random.rand(M, S)) for _ in range(POP_SIZE)]
    trace = np.zeros(N_GEN)
    best_ind = None
    best_fit_val = -np.inf

    for gen in range(N_GEN):
        fits = [fitness_func(ind, 'penalty') for ind in pop]
        idx = np.argmax(fits)
        if fits[idx] > best_fit_val:
            best_fit_val = fits[idx]
            best_ind = pop[idx].copy()
        
        trace[gen] = get_sensing_snr(pop[idx]) if fits[idx] > -1.0 else 0
        
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            child = mutate(crossover(p1, p2))
            new_pop.append(repair_power(child))
        pop = new_pop
    return trace, best_ind, best_fit_val

if __name__ == "__main__":
    print(f"Đang thực hiện GA0 với {N_RUNS} lần chạy...")
    all_traces = np.zeros((N_RUNS, N_GEN))
    global_best_p = None
    global_best_fit = -np.inf

    for r in range(N_RUNS):
        trace, best_p, best_fit = run_ga()
        all_traces[r, :] = trace
        if best_fit > global_best_fit:
            global_best_fit = best_fit
            global_best_p = best_p
    
    avg_trace = np.mean(all_traces, axis=0)
    print_report(global_best_p, global_best_fit, "GA0")

    plt.figure(figsize=(10, 5))
    plt.plot(avg_trace, linewidth=2, label='GA0')
    plt.xlabel("Generation")
    plt.ylabel("Average Best Sensing SNR")
    plt.title(f"Hội tụ của GA0\n(Gamma* = {gamma:.4f})")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()
