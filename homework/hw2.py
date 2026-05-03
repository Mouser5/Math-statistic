import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ========================= НАСТРОЙКИ =========================
n = 300  # Объем выборки
Lambda = 1.2  # Параметр масштаба λ
c = 1.5  # Параметр формы c
np.random.seed(42)


def pdf(x):
    """Плотность распределения Вейбулла-Гнеденко."""
    x = np.asarray(x)
    res = np.zeros_like(x)
    mask = x > 0
    # f(x) = λ * c * x^(c-1) * exp(-λ * x^c)
    res[mask] = Lambda * c * (x[mask] ** (c - 1)) * np.exp(-Lambda * (x[mask] ** c))
    return res


# Вычисляем квадрат L2-нормы теоретической плотности для ОИСКО численно
Norm_L2_sq, _ = quad(lambda x: pdf(x) ** 2, 0, np.inf, epsabs=1e-8)

# ====================== 1. ГЕНЕРАЦИЯ ОДНОЙ ВЫБОРКИ ======================
# Метод обратных функций
u = np.random.uniform(0, 1, n)
sample_single = (-np.log(u) / Lambda) ** (1 / c)

x_min = np.min(sample_single)
x_max = np.max(sample_single)
Delta = x_max - x_min

print(f"Выборка сгенерирована (n={n}):")
print(f"   x_min = {x_min:.6f},  x_max = {x_max:.6f},  Δ = {Delta:.6f}\n")


# ====================== ФУНКЦИЯ ДЛЯ ТОЧНОГО ОИСКО ======================
def compute_delta_n(m, sample, x_min, x_max):
    if m < 1:
        return np.nan
    h = (x_max - x_min) / m
    bins = np.linspace(x_min, x_max, m + 1)
    counts, _ = np.histogram(sample, bins=bins)

    ise = 0.0
    for i in range(m):
        left, right = bins[i], bins[i + 1]
        f_hat = counts[i] / (n * h)
        integrand = lambda x: (f_hat - pdf(x)) ** 2
        integral, _ = quad(integrand, left, right, epsabs=1e-8, limit=100)
        ise += integral

    # Хвосты за пределами выборки
    if x_min > 0:
        left_tail, _ = quad(lambda x: pdf(x) ** 2, 0, x_min, epsabs=1e-8)
        ise += left_tail
    right_tail, _ = quad(lambda x: pdf(x) ** 2, x_max, np.inf, epsabs=1e-8)
    ise += right_tail

    return ise / Norm_L2_sq


# ====================== ПЕРВАЯ ФИГУРА: ГИСТОГРАММЫ m=20, 30, 40, 50 ======================
fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10))
axs1 = axs1.flatten()

m_hist = [20, 30, 40, 50]
for idx, m in enumerate(m_hist):
    h = Delta / m
    bins = np.linspace(x_min, x_max, m + 1)
    counts, _ = np.histogram(sample_single, bins=bins)
    bin_centers = x_min + (np.arange(m) + 0.5) * h
    empirical_density = counts / (n * h)

    ax = axs1[idx]
    ax.bar(bin_centers, empirical_density, width=h, alpha=0.7,
           color='skyblue', edgecolor='black', label='Гистограмма')

    x_plot = np.linspace(0, x_max + 0.5, 1000)
    ax.plot(x_plot, pdf(x_plot), 'r-', lw=2.5, label='Теоретическая f(x)')

    ax.set_title(f'Гистограмма, m = {m}\n(h = {h:.4f})', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('graphics/graphic_2_1.png', dpi=300, bbox_inches='tight')
plt.show()

# Вывод в консоль статистики по гистограммам
for m in m_hist:
    h = Delta / m
    empty_bins = np.sum(np.histogram(sample_single, bins=np.linspace(x_min, x_max, m + 1))[0] == 0)
    delta_val = compute_delta_n(m, sample_single, x_min, x_max)
    print(f"m = {m:2d} | h = {h:.5f} | пустых разрядов = {empty_bins:2d} | ОИСКО δ_n = {delta_val:.6f}")

# ====================== ВТОРАЯ ФИГУРА: УСРЕДНЕННАЯ ОИСКО ======================
# Для построения стабильного графика зависимости вычислим усредненную ОИСКО по N симуляциям
N_sim = 100
m_all = list(range(5, 61))  # Диапазон от 5 до 60 разрядов
delta_avg = np.zeros(len(m_all))

print(f"\nВычисляем усредненную ОИСКО (по {N_sim} независимым выборкам) для построения кривой...")

for _ in range(N_sim):
    u = np.random.uniform(0, 1, n)
    sample = (-np.log(u) / Lambda) ** (1 / c)
    x_min_s = np.min(sample)
    x_max_s = np.max(sample)

    for i, m in enumerate(m_all):
        delta_avg[i] += compute_delta_n(m, sample, x_min_s, x_max_s)

delta_avg /= N_sim

# Построение графика
fig2 = plt.figure(figsize=(10, 6))
plt.plot(m_all, delta_avg, 'o-', markersize=4, lw=2, color='navy')

plt.title(f'Зависимость ОИСКО от числа разрядов m (n = {n})')
plt.xlabel('Число разрядов m')
plt.ylabel(r'Усредненная $\overline{\delta}_n(m)$')
plt.grid(True, alpha=0.3)
plt.xticks(range(5, 61, 5))

# Отрисовка оптимального m
min_m_idx = np.argmin(delta_avg)
opt_m = m_all[min_m_idx]
plt.axvline(x=opt_m, color='red', linestyle='--', label=f'Оптимальное m ≈ {opt_m}')

plt.legend()
plt.tight_layout()
plt.savefig('graphics/graphic_2_2.png', dpi=300, bbox_inches='tight')
plt.show()

# ====================== ИТОГ ======================
print(f"\nМИНИМУМ усредненной ОИСКО достигается при m = {opt_m} "
      f"(δ_n ≈ {delta_avg[min_m_idx]:.6f})")