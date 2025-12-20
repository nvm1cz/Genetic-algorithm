import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. THIẾT LẬP HỆ THỐNG (M=3, U=2, Q=1)
# =====================================================
np.random.seed(1)
M, U, Q = 3, 2, 1
S = U + Q
P_AP = 1.0
noise_u = 0.1
gamma = 0.35446747231755743  # Ngưỡng Gamma*

# Tham số GA2 (Steady-State)
POP_SIZE = 40
N_GEN = 120              # Số khối lặp (tương đương thế hệ)
PC = 0.9
PM = 0.12
N_RUNS = 10

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

def fitness_ga2(p):
    # Tiêu chuẩn Feasibility-first (Lexicographic) giống bản template của ông
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
        p[:, U:] *= 0.5 # Ưu tiên truyền thông bằng cách cắt giảm công suất Radar
    return repair_power(p)

def crossover(a, b):
    if np.random.rand() < PC:
        alpha = np.random.rand()
        return alpha * a + (1 - alpha) * b
    return a.copy()

def mutation(p):
    if np.random.rand() < PM:
        p[np.random.randint(M), np.random.randint(S)] += 0.08 * np.random.randn()
    return p

def tournament_selection(pop, fits, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax([fits[i] for i in idx])]]

# =====================================================
# 4. THUẬT TOÁN GA2 (STEADY-STATE)
# =====================================================
def run_ga2():
    # Khởi tạo quần thể ban đầu
    pop = [repair_power(np.random.rand(M, S)) for _ in range(POP_SIZE)]
    fits = [fitness_ga2(ind) for ind in pop]
    
    trace = np.zeros(N_GEN)

    for gen in range(N_GEN):
        # Thực hiện POP_SIZE lần cập nhật để tương đương khối lượng tính toán
        for _ in range(POP_SIZE):
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            
            child = mutation(crossover(p1, p2))
            child = repair_feasible(child)
            child_fit = fitness_ga2(child)
            
            # THAY THẾ CÁ THỂ YẾU NHẤT
            worst_idx = np.argmin(fits)
            if child_fit > fits[worst_idx]:
                pop[worst_idx] = child
                fits[worst_idx] = child_fit

        best_idx = np.argmax(fits)
        trace[gen] = get_sensing_snr(pop[best_idx]) if fits[best_idx] > -1.0 else 0

    return trace

# =====================================================
# 5. THỰC THI & VẼ ĐỒ THỊ
# =====================================================
if __name__ == "__main__":
    print(f"Đang chạy GA2 (Steady-State) trung bình {N_RUNS} lần...")
    results = np.zeros((N_RUNS, N_GEN))
    for r in range(N_RUNS):
        results[r, :] = run_ga2()
    
    avg_trace = np.mean(results, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_trace, color='orange', linewidth=2, label='GA2 (Steady-State)')
    plt.xlabel("Generation (Iteration Blocks)")
    plt.ylabel("Average Best Sensing SNR")
    plt.title(f"Hội tụ của GA2 (Steady-State GA)\n(Gamma* = {gamma:.4f})")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()