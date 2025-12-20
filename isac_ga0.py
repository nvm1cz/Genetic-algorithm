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
    # Kênh truyền H giữa người dùng và AP
    h = rng.standard_normal((U, M)) + 1j * rng.standard_normal((U, M))
    # Hướng búp sóng cố định cho truyền thông
    fbar_comm = rng.standard_normal((M, U)) + 1j * rng.standard_normal((M, U))
    fbar_comm /= np.linalg.norm(fbar_comm, axis=1, keepdims=True)
    # Hướng búp sóng cho cảm biến
    rng_s = np.random.default_rng(2)
    fbar_sens = rng_s.standard_normal((M, Q)) + 1j * rng_s.standard_normal((M, Q))
    fbar_sens /= np.linalg.norm(fbar_sens, axis=1, keepdims=True)
    
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
    # Fitness = Reward - Penalty
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

def mutation(p):
    if np.random.rand() < PM:
        m, s = np.random.randint(M), np.random.randint(S)
        p[m, s] += 0.08 * np.random.randn()
    return p

def tournament_selection(pop, fits, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax([fits[i] for i in idx])]]

# =====================================================
# 4. VÒNG LẶP TIẾN HÓA GA1
# =====================================================
def run_ga1():
    # Khởi tạo quần thể ban đầu
    pop = []
    for _ in range(POP_SIZE):
        p = np.random.rand(M, S)
        for m in range(M):
            p[m] = p[m] / (np.sum(p[m]) + 1e-9) * P_AP
        pop.append(p)

    trace = np.zeros(N_GEN)

    for gen in range(N_GEN):
        fits = [fitness_ga1(ind) for ind in pop]
        
        best_idx = np.argmax(fits)
        # Chỉ lưu Sensing SNR nếu tìm được nghiệm gần vùng khả thi
        trace[gen] = get_sensing_snr(pop[best_idx]) if fits[best_idx] > -1.0 else 0

        # --- CƠ CHẾ ELITISM (Điểm khác biệt của GA1) ---
        # Sắp xếp chỉ số cá thể theo fitness giảm dần
        sorted_indices = np.argsort(fits)[::-1]
        new_pop = []
        for i in range(ELITE):
            new_pop.append(pop[sorted_indices[i]].copy())

        # Tạo các cá thể còn lại thông qua lai ghép và đột biến
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            child = mutation(crossover(p1, p2))
            child = repair_power(child)
            new_pop.append(child)
            
        pop = new_pop

    return trace

# =====================================================
# 5. THỰC THI VÀ VẼ ĐỒ THỊ
# =====================================================
if __name__ == "__main__":
    print(f"Đang thực hiện GA1 (Elitism) với {N_RUNS} lần chạy...")
    results = np.zeros((N_RUNS, N_GEN))
    for r in range(N_RUNS):
        results[r, :] = run_ga1()
    
    avg_trace = np.mean(results, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_trace, color='green', linewidth=2, label='GA1 (Elitism)')
    plt.xlabel("Generation")
    plt.ylabel("Average Best Sensing SNR")
    plt.title(f"Hội tụ của GA1 với cơ chế Elitism\n(Gamma* = {gamma:.4f}, Elite={ELITE})")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()