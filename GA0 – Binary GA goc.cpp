// GA0: Binary Genetic Algorithm (GA goc + thong so danh gia)
#include <bits/stdc++.h>
using namespace std;

double f(double x) {
    return x * x - 5 * x + 6;
}

// =========================
// Tham so GA
// =========================
const int    POP_SIZE = 100;
const int    MAX_GEN  = 500;
const double PC       = 0.8;
const double PM       = 0.05;     // nho, co dinh
const double EPS      = 1e-6;
const double X_MIN    = -100.0;
const double X_MAX    =  100.0;
const int    BITS     = 16;       // so bit bieu dien

mt19937 rng((unsigned) time(nullptr));

double randDouble(double a, double b) {
    uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

// sinh so nguyen ngau nhien trong [lo, hi]
unsigned int randUInt(unsigned int lo, unsigned int hi) {
    uniform_int_distribution<unsigned int> dist(lo, hi);
    return dist(rng);
}

// Dem so lan evaluate fitness (chi phi tinh toan)
long long fitness_eval_total = 0;

// =========================
// Cau truc ca the
// =========================
struct Individual {
    unsigned int gene; // 16 bit thap
    double x;
    double fitness;

    void decode() {
        unsigned int maxVal = (1u << BITS) - 1u;
        double ratio = (double)gene / (double)maxVal;
        x = X_MIN + ratio * (X_MAX - X_MIN);
    }

    void evaluate() {
        decode();
        double fx = fabs(f(x));
        fitness   = 1.0 / (1.0 + fx);
        fitness_eval_total++;
    }
};

// Roulette wheel selection
Individual rouletteSelect(const vector<Individual>& pop) {
    double sumFit = 0.0;
    for (auto &ind : pop) sumFit += ind.fitness;
    double r   = randDouble(0.0, sumFit);
    double acc = 0.0;
    for (auto &ind : pop) {
        acc += ind.fitness;
        if (acc >= r) return ind;
    }
    return pop.back();
}

// One-point crossover tren BITS bit
Individual crossover(const Individual &p1, const Individual &p2) {
    Individual child = p1;
    int point = randUInt(0, BITS - 1);  // cat sau bit [0..BITS-1]
    unsigned int maskLeft  = (~0u) << point;
    unsigned int maskRight = ~maskLeft;
    child.gene = (p1.gene & maskLeft) | (p2.gene & maskRight);
    return child;
}

// Bit-flip mutation
void mutate(Individual &ind) {
    for (int i = 0; i < BITS; ++i) {
        if (randDouble(0.0, 1.0) < PM) {
            ind.gene ^= (1u << i);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    using namespace std::chrono;
    auto t_start = high_resolution_clock::now();

    // Khoi tao quan the
    vector<Individual> pop(POP_SIZE);
    unsigned int maxGeneVal = (1u << BITS) - 1u;
    for (int i = 0; i < POP_SIZE; ++i) {
        pop[i].gene = randUInt(0u, maxGeneVal);
        pop[i].evaluate();
    }

    Individual best = pop[0];
    for (auto &ind : pop)
        if (ind.fitness > best.fitness) best = ind;

    // Bien theo doi hoi tu & stall
    int    gen_converged   = -1;
    double last_best_fit   = best.fitness;
    int    stall_gen       = 0;
    int    max_stall_gen   = 0;

    for (int gen = 0; gen < MAX_GEN; ++gen) {
        // Cap nhat best trong quan the hien tai
        for (auto &ind : pop)
            if (ind.fitness > best.fitness) best = ind;

        cout << "The he: " << gen
             << "   x tot nhat: " << best.x
             << "   f(x): " << f(best.x)
             << "   fitness: " << best.fitness << "\n";

        // Ghi nhan the he hoi tu lan dau
        if (fabs(f(best.x)) < EPS && gen_converged == -1) {
            gen_converged = gen;
            cout << "\nTim thay nghiem gan dung tai the he " << gen << "\n";
            // Neu chi muon dung som thi mo comment:
            // break;
        }

        // Cap nhat stall
        if (best.fitness > last_best_fit + 1e-12) {
            last_best_fit = best.fitness;
            stall_gen     = 0;
        } else {
            stall_gen++;
            max_stall_gen = max(max_stall_gen, stall_gen);
        }

        // Tao quan the moi (khong elitism)
        vector<Individual> newPop;
        while ((int)newPop.size() < POP_SIZE) {
            Individual p1 = rouletteSelect(pop);
            Individual p2 = rouletteSelect(pop);

            Individual c1 = p1;
            Individual c2 = p2;

            if (randDouble(0.0, 1.0) < PC) {
                c1 = crossover(p1, p2);
                c2 = crossover(p2, p1);
            }

            mutate(c1);
            mutate(c2);
            c1.evaluate();
            c2.evaluate();

            newPop.push_back(c1);
            if ((int)newPop.size() < POP_SIZE)
                newPop.push_back(c2);
        }

        pop = newPop;
    }

    auto t_end  = high_resolution_clock::now();
    auto dur_ms = duration_cast<milliseconds>(t_end - t_start).count();

    cout << "\n===== KET QUA =====\n";
    cout << "Nghiem x gan dung: " << best.x << "\n";
    cout << "f(x) = " << f(best.x) << "\n";
    cout << "fitness = " << best.fitness << "\n";
    cout << "Thoi gian chay (ms): " << dur_ms << "\n";
    cout << "Tong so lan danh gia fitness: " << fitness_eval_total << "\n";
    cout << "The he hoi tu (neu co, -1 neu khong): " << gen_converged << "\n";
    cout << "Max so the he lien tiep khong cai thien (max stall): " << max_stall_gen << "\n";

    return 0;
}
