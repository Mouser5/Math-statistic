import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, IntegrationWarning
import warnings

# Отключаем предупреждения интегратора (возникают из-за разрывов прямоугольного ядра)
warnings.filterwarnings("ignore", category=IntegrationWarning)

# ========================= НАСТРОЙКИ =========================
n = 300  # Объем выборки
Lambda = 1.0  # Параметр λ
c_param = 1.5  # Параметр формы c
np.random.seed(42)

# Генерация выборки из распределения Вейбулла-Гнеденко (метод обратного преобразования)
u = np.random.uniform(0, 1, n)
sample = (-np.log(u) / Lambda) ** (1 / c_param)


def f(x):
    """Теоретическая плотность распределения Вейбулла-Гнеденко."""
    x = np.asarray(x)
    res = np.zeros_like(x)
    mask = x > 0
    res[mask] = Lambda * c_param * (x[mask] ** (c_param - 1)) * np.exp(-Lambda * (x[mask] ** c_param))
    return res


# Прямоугольное ядро
def K(u):
    a = np.sqrt(3)
    return np.where(np.abs(u) <= a, 1 / (2 * a), 0.0)


# Векторизованная ядерная оценка плотности
def f_hat(x, h, sample):
    return np.sum(K((x[:, None] - sample) / h), axis=1) / (n * h)


# Вычисляем квадрат L2-нормы теоретической плотности численно
Norm_L2, _ = quad(lambda x: f(x) ** 2, 0, np.inf, epsabs=1e-9, limit=500)

# ====================== ОИСКО ======================
# Используем две сетки: детальную для поиска минимума и широкую для асимптотики
h_values = np.arange(0.05, 3.01, 0.05)
h_values_2 = np.arange(1.0, 20.1, 1.0)
delta_values = []
delta_values_2 = []

print("Вычисляем ОИСКО для детальной сетки h...")
for h in h_values:
    def integrand(x):
        return (f_hat(np.array([x]), h, sample)[0] - f(x)) ** 2


    ise, _ = quad(integrand, 0, 10, limit=300)

    # Теоретический хвост (для точности на бесконечности)
    tail, _ = quad(lambda x: f(x) ** 2, 10, np.inf)
    delta_values.append((ise + tail) / Norm_L2)

print("Вычисляем ОИСКО для широкой сетки h...")
for h_2 in h_values_2:
    def integrand(x):
        return (f_hat(np.array([x]), h_2, sample)[0] - f(x)) ** 2


    ise_2, _ = quad(integrand, 0, 15 + h_2, limit=300)

    tail_2, _ = quad(lambda x: f(x) ** 2, 15 + h_2, np.inf)
    delta_values_2.append((ise_2 + tail_2) / Norm_L2)

delta_values = np.array(delta_values)
best_idx = np.argmin(delta_values)
best_h = h_values[best_idx]

delta_values_2 = np.array(delta_values_2)
best_idx_2 = np.argmin(delta_values_2)
best_h_2 = h_values_2[best_idx_2]

# ====================== ТАБЛИЦА ======================
print("\n" + "=" * 55)
print("  h     |   ОИСКО δ_n")
print("-" * 55)
for i in range(0, len(h_values), 4):
    print(f"{h_values[i]:.3f}   |   {delta_values[i]:.6f}")
print("=" * 55)
print(f"МИНИМУМ в детальной сетке: h = {best_h:.3f},  δ_n = {delta_values[best_idx]:.6f}\n")

# ====================== ГРАФИКИ ======================
fig, axs = plt.subplots(1, 3, figsize=(16, 6))

# 1. Зависимость ОИСКО от h (Детальный вид)
axs[0].plot(h_values, delta_values, 'o-', color='purple', markersize=4, lw=2.5)
axs[0].plot(best_h, delta_values[best_idx], 'o', color='green', markersize=10)
axs[0].set_title('ОИСКО от ширины окна (0.05 - 3.0)')
axs[0].set_xlabel('Параметр сглаживания h')
axs[0].set_ylabel(r'$\overline{\delta}_n(h)$')
axs[0].grid(True, alpha=0.3)
axs[0].legend(['ОИСКО', f'Минимум при h={best_h:.3f}'])

# 2. Зависимость ОИСКО от h (Широкий вид)
axs[1].plot(h_values_2, delta_values_2, 'o-', color='purple', markersize=4, lw=2.5)
axs[1].set_title('ОИСКО от ширины окна (1.0 - 20.0)')
axs[1].set_xlabel('Параметр сглаживания h')
axs[1].set_ylabel(r'$\overline{\delta}_n(h)$')
axs[1].grid(True, alpha=0.3)
axs[1].legend(['Асимптотика ОИСКО'])

# 3. Сравнение лучшей оценки + ГИСТОГРАММА
x_plot = np.linspace(0, 4.5, 1000)
y_hat = f_hat(x_plot, best_h, sample)

axs[2].plot(x_plot, f(x_plot), 'r-', lw=2.5, label='Теоретическая f(x)')
axs[2].plot(x_plot, y_hat, 'b--', lw=2.2, label=f'Ядерная оценка (h={best_h:.3f})')
axs[2].hist(sample, bins=25, density=True, alpha=0.7, color='skyblue',
            edgecolor='black', label='Гистограмма')
axs[2].set_title('Сравнение лучшей ядерной оценки')
axs[2].set_xlabel('x')
axs[2].set_ylabel('Плотность')
axs[2].grid(True, alpha=0.3)
axs[2].legend()

plt.tight_layout()
plt.savefig('graphics/graphic_4.png', dpi=300, bbox_inches='tight')
plt.show()