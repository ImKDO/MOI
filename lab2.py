import numpy as np

# --- 1. Базовые функции ---
def f(x):
    return x**2

a, b = 2.0, 5.0
I_true = 39.0

def eval_error(N):
    return I_true / np.sqrt(N)

# Плотности вероятности и обратные функции (Inverse Transform Sampling)
def p1(x): return 2*x / 21
def inv_p1(xi): return np.sqrt(21*xi + 4)

def p2(x): return x**2 / 39
def inv_p2(xi): return np.cbrt(117*xi + 8)

def p3(x): return 4*x**3 / 609
def inv_p3(xi): return (609*xi + 16)**0.25

# --- 2. Простой метод Монте-Карло ---
def mc_simple(N):
    x = np.random.uniform(a, b, N)
    return (b - a) * np.mean(f(x))

# --- 3. Метод со стратификацией ---
def mc_stratified(N, step):
    K = int(np.round((b - a) / step))
    # Разбиваем N на K слоев поровну (плюс остаток, если есть)
    sizes = [N // K + (1 if i < N % K else 0) for i in range(K)]
    
    integral = 0.0
    for i in range(K):
        x_start = a + i * step
        x_end = x_start + step
        x = np.random.uniform(x_start, x_end, sizes[i])
        integral += step * np.mean(f(x))
    return integral

# --- 4. Выборка по значимости ---
def mc_importance(N, inv_cdf, pdf):
    xi = np.random.uniform(0, 1, N)
    x = inv_cdf(xi)
    return np.mean(f(x) / pdf(x))

# --- 5. Многократная выборка по значимости (MIS) ---
def mc_mis(N, weight_type='balance'):
    n1 = N // 2
    n2 = N - n1
    
    # Сэмплы из p1(x) ~ x и p3(x) ~ x^3 (в задании названа p_2(x))
    x1 = inv_p1(np.random.uniform(0, 1, n1))
    x2 = inv_p3(np.random.uniform(0, 1, n2))
    
    if weight_type == 'balance':
        w1_x1 = p1(x1) / (p1(x1) + p3(x1))
        w2_x2 = p3(x2) / (p1(x2) + p3(x2))
    elif weight_type == 'power':
        w1_x1 = p1(x1)**2 / (p1(x1)**2 + p3(x1)**2)
        w2_x2 = p3(x2)**2 / (p1(x2)**2 + p3(x2)**2)

    # I = 1/n1 * sum(f*w1/p1) + 1/n2 * sum(f*w2/p2)
    est1 = np.mean(f(x1) * w1_x1 / p1(x1))
    est2 = np.mean(f(x2) * w2_x2 / p3(x2))
    return est1 + est2

# --- 6. Русская рулетка ---
def mc_russian_roulette(N, R):
    x = np.random.uniform(a, b, N)
    xi = np.random.uniform(0, 1, N)
    
    # Оцениваем функцию только если xi <= R, иначе 0. При оценке делим на R.
    eval_mask = xi <= R
    vals = np.where(eval_mask, f(x)/R, 0)
    return (b - a) * np.mean(vals)

# --- Главная функция и вывод ---
def run_experiments():
    N_list = [100, 1000, 10000, 100000]
    
    experiments = [
        ("Простой Монте-Карло", lambda N: mc_simple(N)),
        ("Стратификация (шаг 1.0)", lambda N: mc_stratified(N, 1.0)),
        ("Стратификация (шаг 0.5)", lambda N: mc_stratified(N, 0.5)),
        ("По значимости p(x) ~ x", lambda N: mc_importance(N, inv_p1, p1)),
        ("По значимости p(x) ~ x^2", lambda N: mc_importance(N, inv_p2, p2)),
        ("По значимости p(x) ~ x^3", lambda N: mc_importance(N, inv_p3, p3)),
        ("MIS (Баланс. веса)", lambda N: mc_mis(N, 'balance')),
        ("MIS (Степенные веса)", lambda N: mc_mis(N, 'power')),
        ("Русская рулетка (R=0.5)", lambda N: mc_russian_roulette(N, 0.50)),
        ("Русская рулетка (R=0.75)", lambda N: mc_russian_roulette(N, 0.75)),
        ("Русская рулетка (R=0.95)", lambda N: mc_russian_roulette(N, 0.95)),
    ]
    
    print(f"{'Метод':<25} | {'N':>7} | {'I вычисл.':>10} | {'Погрешность':>12} | {'Оценка ΔI':>10}")
    print("-" * 72)
    
    for name, func in experiments:
        for N in N_list:
            I_calc = func(N)
            error = abs(I_calc - I_true)
            delta_I = eval_error(N)
            print(f"{name:<25} | {N:>7} | {I_calc:>10.5f} | {error:>12.5f} | {delta_I:>10.5f}")
        print("-" * 72)

if __name__ == "__main__":
    np.random.seed(42)
    run_experiments()