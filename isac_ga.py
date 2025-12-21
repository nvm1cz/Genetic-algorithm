import numpy as np
import matplotlib.pyplot as plt
import copy

# =====================================================
# 1. CẤU HÌNH HỆ THỐNG ISAC (Đồng bộ)
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
PM_BASE = 0.12
ELITE = 2
LAMBDA_SINR = 80.0
N_RUNS = 10 

def make_system():
    rng = np.random.default_rng(1)
    h = rng.standard_normal((U, M)) + 1j * rng.standard_normal((U, M))
    fbar_comm = rng.standard_normal((M, U)) + 1j * rng.standard_normal((M, U))
    fbar_comm /= (np.linalg.norm(fbar_comm, axis=1, keepdims=True) + 1e-9)
    rng_s = np.random.default_rng(2)
    fbar_sens = rng_s.standard_normal((M, Q)) + 1j * rng_s.standard_normal((M, Q))
    fbar_sens /= (np.linalg.norm(fbar_sens, axis=1, keepdims=True) + 1e-9)
    f_bar = np.concatenate([fbar_comm, fbar_sens], axis=1)
    return h, f_bar

H_CHANNELS, F_BAR = make_system()

# =====================================================
# 2. CÁC HÀM TÍNH TOÁN HIỆU NĂNG
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

def sensing_snr(p):
    return float(np.sum(p[:, U:]))

# =====================================================
# 3. CÁC TOÁN TỬ GA & REPAIR
# =====================================================
def repair_power(p):
    p = np.maximum(p, 0.0)
    for m in range(M):
        s = np.sum(p[m])
        if s > P_AP: p[m] *= P_AP / (s + 1e-9)
    return p

def repair_feasible(p):
    p = repair_power(p)
    if np.any(compute_sinr(p) < gamma): p[:, U:] *= 0.5
    return repair_power(p)

def crossover(a, b):
    if np.random.rand() < PC:
        alpha = np.random.rand()
        return alpha * a + (1 - alpha) * b
    return a.copy()

def mutation(p, pm_rate):
    if np.random.rand() < pm_rate:
        p[np.random.randint(M), np.random.randint(S)] += 0.08 * np.random.randn()
    return p

def tournament(pop, fits, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax([fits[i] for i in idx])]]

def print_report(best_p, best_fit, title):
    sinr = compute_sinr(best_p)
    snr_s = sensing_snr(best_p)
    is_feasible = "Có" if np.all(sinr >= gamma - 1e-6) else "Không"
    print(f"\n--- KẾT QUẢ TỐI ƯU: {title} ---")
    print(f"Sensing SNR: {snr_s:.4f} | SINR min: {np.min(sinr):.4f} | Feasible: {is_feasible}")
    print("Ma trận công suất p_ms:")
    print(best_p)

# =====================================================
# 4. VÒNG LẶP GA CHÍNH
# =====================================================
def run_single_ga(variant):
    pop = [repair_power(np.random.rand(M, S)) for _ in range(POP_SIZE)]
    trace = np.zeros(N_GEN)
    best_ind_run = None
    best_fit_run = -np.inf
    pm = PM_BASE
    no_improve = 0

    for gen in range(N_GEN):
        # Tính Penalty động cho GA1
        penalty = LAMBDA_SINR
        if variant == "GA1":
            viol_ratio = np.mean([np.any(compute_sinr(ind) < gamma) for ind in pop])
            penalty = LAMBDA_SINR * (1 + 5 * viol_ratio)

        # Tính Fitness
        fits = []
        for ind in pop:
            sinr = compute_sinr(ind)
            snr_s = sensing_snr(ind)
            if variant in ["GA0", "GA1"]:
                f = snr_s - penalty * np.sum(np.maximum(0.0, gamma - sinr) ** 2)
            else: # Lexicographic cho GA2, GA3
                f = snr_s if np.all(sinr >= gamma) else -1e6 - np.sum(np.maximum(0.0, gamma - sinr)) * 1e3
            fits.append(f)

        # Cập nhật Best và Adaptive PM cho GA3
        idx = np.argmax(fits)
        if fits[idx] > best_fit_run:
            best_fit_run = fits[idx]; best_ind_run = pop[idx].copy(); no_improve = 0
            if variant == "GA3": pm = max(0.05, pm - 0.01)
        else:
            no_improve += 1
            if variant == "GA3" and no_improve > 5: pm = min(0.6, pm + 0.05)

        trace[gen] = sensing_snr(pop[idx]) if fits[idx] > -1e5 else 0

        # Sinh sản (GA2 dùng Steady-State logic riêng, ở đây gộp chung để dễ so sánh)
        new_pop = []
        if variant != "GA0":
            sorted_idx = np.argsort(fits)[-ELITE:]
            for i in sorted_idx: new_pop.append(pop[i].copy())

        while len(new_pop) < POP_SIZE:
            p1 = tournament(pop, fits); p2 = tournament(pop, fits)
            child = mutation(crossover(p1, p2), pm)
            child = repair_feasible(child) if variant in ["GA2", "GA3"] else repair_power(child)
            new_pop.append(child)
        pop = new_pop

    return trace, best_ind_run, best_fit_run

# =====================================================
# 5. THỰC THI
# =====================================================
if __name__ == "__main__":
    variants = ["GA0", "GA1", "GA2", "GA3"]
    plt.figure(figsize=(10, 6))

    for var in variants:
        print(f"Đang chạy {var}...")
        results = np.zeros((N_RUNS, N_GEN))
        v_best_p = None; v_best_fit = -np.inf
        for r in range(N_RUNS):
            trace, b_p, b_f = run_single_ga(var)
            results[r, :] = trace
            if b_f > v_best_fit: v_best_fit = b_f; v_best_p = b_p
        
        print_report(v_best_p, v_best_fit, var)
        plt.plot(np.mean(results, axis=0), label=var, linewidth=2)

    plt.xlabel("Generation"); plt.ylabel("Avg Best Sensing SNR (Feasible)")
    plt.title(f"Comparison of GA Variants (Gamma* = {gamma:.4f})")
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend(); plt.show()
