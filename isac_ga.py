import numpy as np
import matplotlib.pyplot as plt
import copy

# =====================================================
# 1. CẤU HÌNH HỆ THỐNG ISAC (Theo yêu cầu)
# =====================================================
np.random.seed(1)
M, U, Q = 3, 2, 1
S = U + Q
P_AP = 1.0
noise_u = 0.1
gamma = 0.35446747231755743  # Gamma star

# Tham số GA
POP_SIZE = 40
N_GEN = 120
PC = 0.9
PM = 0.12
ELITE = 2
LAMBDA_SINR = 80.0
N_RUNS = 10 

# Khởi tạo kênh truyền và Beamforming cố định
def make_system():
    rng = np.random.default_rng(1)
    # Kênh truyền h_um (U x M)
    h = rng.standard_normal((U, M)) + 1j * rng.standard_normal((U, M))
    # Beamforming cố định (MRT style)
    fbar_comm = rng.standard_normal((M, U)) + 1j * rng.standard_normal((M, U))
    fbar_comm /= np.linalg.norm(fbar_comm, axis=1, keepdims=True)
    
    rng_s = np.random.default_rng(2)
    fbar_sens = rng_s.standard_normal((M, Q)) + 1j * rng_s.standard_normal((M, Q))
    fbar_sens /= np.linalg.norm(fbar_sens, axis=1, keepdims=True)
    
    f_bar = np.concatenate([fbar_comm, fbar_sens], axis=1)
    return h, f_bar

h, f_bar = make_system()

# =====================================================
# 2. CÁC HÀM TÍNH TOÁN HIỆU NĂNG
# =====================================================
def compute_sinr(p):
    sinr = np.zeros(U)
    for u in range(U):
        sig, interf = 0.0, 0.0
        for m in range(M):
            sig += p[m, u] * abs(np.conj(h[u, m]) * f_bar[m, u])**2
            for s in range(S):
                if s != u:
                    interf += p[m, s] * abs(np.conj(h[u, m]) * f_bar[m, s])**2
        sinr[u] = sig / (interf + noise_u)
    return sinr

def sensing_snr(p):
    return float(np.sum(p[:, U:]))

# =====================================================
# 3. CÁC TOÁN TỬ GA & REPAIR
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
        p[:, U:] *= 0.5 # Giảm công suất sensing để cứu SINR UE
    return repair_power(p)

def crossover(a, b):
    if np.random.rand() < PC:
        alpha = np.random.rand()
        return alpha * a + (1 - alpha) * b
    return a.copy()

def mutation(p):
    if np.random.rand() < PM:
        m_idx = np.random.randint(M)
        s_idx = np.random.randint(S)
        p[m_idx, s_idx] += 0.08 * np.random.randn()
    return p

def tournament(pop, fits, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax([fits[i] for i in idx])]]

# =====================================================
# 4. HÀM THÍCH NGHI (FITNESS) THEO TỪNG BIẾN THỂ
# =====================================================
def get_fitness(p, variant, **kwargs):
    sinr = compute_sinr(p)
    snr_s = sensing_snr(p)
    
    if variant == "GA0":
        return snr_s - LAMBDA_SINR * np.sum(np.maximum(0.0, gamma - sinr) ** 2)
    
    elif variant == "GA1":
        penalty = kwargs.get('penalty', LAMBDA_SINR)
        return snr_s - penalty * np.sum(np.maximum(0.0, gamma - sinr) ** 2)
    
    elif variant in ["GA2", "GA3"]:
        # Lexicographic: Infeasible < 0, Feasible > 0
        if np.any(sinr < gamma):
            return -1e6 - np.sum(np.maximum(0.0, gamma - sinr)) * 1e3
        return snr_s

# =====================================================
# 5. VÒNG LẶP GA CHÍNH
# =====================================================
def run_single_ga(variant):
    # Khởi tạo quần thể ngẫu nhiên và chuẩn hóa công suất
    pop = []
    for _ in range(POP_SIZE):
        p = np.random.rand(M, S)
        for m in range(M):
            p[m] = p[m] / (np.sum(p[m]) + 1e-9) * P_AP
        pop.append(p)

    trace = np.zeros(N_GEN)

    for gen in range(N_GEN):
        # Tính toán Penalty động cho GA1
        penalty = LAMBDA_SINR
        if variant == "GA1":
            viol_ratio = np.mean([np.any(compute_sinr(ind) < gamma) for ind in pop])
            penalty = LAMBDA_SINR * (1 + 5 * viol_ratio)

        # Tính Fitness
        fits = [get_fitness(ind, variant, penalty=penalty) for ind in pop]
        
        # Lưu kết quả tốt nhất (chỉ lấy phần Sensing SNR thực tế để vẽ)
        best_idx = np.argmax(fits)
        trace[gen] = sensing_snr(pop[best_idx]) if fits[best_idx] > -1e5 else 0

        # Elitism
        elite_indices = np.argsort(fits)[-ELITE:]
        new_pop = [pop[i].copy() for i in elite_indices]

        # Sinh sản
        while len(new_pop) < POP_SIZE:
            p1 = tournament(pop, fits)
            p2 = tournament(pop, fits)
            child = mutation(crossover(p1, p2))

            # Repair logic
            if variant in ["GA2", "GA3"]:
                child = repair_feasible(child)
            else:
                child = repair_power(child)
            
            new_pop.append(child)
        
        pop = new_pop

    return trace

# =====================================================
# 6. CHẠY SO SÁNH & VẼ ĐỒ THỊ
# =====================================================
if __name__ == "__main__":
    variants = ["GA0", "GA1", "GA2", "GA3"]
    all_results = {}

    for var in variants:
        print(f"Đang chạy {var} (Trung bình {N_RUNS} lần)...")
        results = np.zeros((N_RUNS, N_GEN))
        for r in range(N_RUNS):
            results[r, :] = run_single_ga(var)
        all_results[var] = np.mean(results, axis=0)

    # Vẽ đồ thị Paper-Ready (Feasible only)
    plt.figure(figsize=(10, 6))
    for var in variants:
        plt.plot(all_results[var], label=var, linewidth=2)

    plt.xlabel("Generation")
    plt.ylabel("Average Best Sensing SNR (Feasible)")
    plt.title(f"Comparison of GA Variants in Cell-Free ISAC MIMO\n(Gamma* = {gamma:.4f})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()