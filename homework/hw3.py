import numpy as np
from scipy.special import laguerre
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ========================= НАСТРОЙКИ =========================
np.random.seed(42)
n = 300  # Объем выборки по условию
Lambda = 1.0  # Параметр λ
c_param = 1.5  # Параметр формы c

# Генерация выборки из распределения Вейбулла-Гнеденко (метод обратного преобразования)
u = np.random.uniform(0, 1, n)
sample = (-np.log(u) / Lambda) ** (1 / c_param)


def f(x):
    """Плотность распределения Вейбулла-Гнеденко."""
    x = np.asarray(x)
    res = np.zeros_like(x)
    mask = x > 0
    # f(x) = λ * c * x^(c-1) * exp(-λ * x^c)
    res[mask] = Lambda * c_param * (x[mask] ** (c_param - 1)) * np.exp(-Lambda * (x[mask] ** c_param))
    return res


def phi(i, x):
    """Система ортонормированных функций Лагерра."""
    L = laguerre(i)
    return np.exp(-x / 2.0) * L(x)


# Вычисляем квадрат L2-нормы теоретической плотности (для точного расчета ОИСКО)
Norm_L2_sq, _ = quad(lambda x: f(x) ** 2, 0, np.inf, epsabs=1e-10, limit=1000)

# ====================== КОЭФФИЦИЕНТЫ ======================
MAX_N = 40  # Максимальное N по задаче: 40
true_c = np.zeros(MAX_N + 1)
hat_c = np.zeros(MAX_N + 1)

print("Вычисляем коэффициенты")
for i in range(MAX_N + 1):
    integrand = lambda x: f(x) * phi(i, x)
    # Повышенные лимиты интегрирования для устранения артефактов осцилляции
    true_c[i], _ = quad(integrand, 0, np.inf, epsabs=1e-9, limit=1000)
    hat_c[i] = np.mean(phi(i, sample))

# ====================== КРАСИВАЯ ТАБЛИЦА КОЭФФИЦИЕНТОВ ======================
print("\n" + "=" * 50)
print(" i | true_c_i      | hat_c_i")
print("-" * 50)
for i in range(11):  # Выведем первые 10 для наглядности
    print(f"{i:2d} | {true_c[i]:.10f} | {hat_c[i]:.10f}")
print("...| ...           | ...")
print("=" * 50)

# ====================== ОИСКО ДЛЯ N = 10(5)40 ======================
N_oisco = list(range(10, 45, 5))  # 10, 15, 20, 25, 30, 35, 40
delta_n = []

print("\nВычисляем ОИСКО для N = 10(5)40...")
for N in N_oisco:
    # Ошибка на вычисленных коэффициентах
    disp = np.sum((hat_c[:N + 1] - true_c[:N + 1]) ** 2)
    # Хвост ряда: точная норма минус учтенные теоретические коэффициенты
    tail = Norm_L2_sq - np.sum(true_c[:N + 1] ** 2)
    tail = max(0, tail)  # Защита от микропогрешностей вычислений

    ise = disp + tail
    delta_n.append(ise / Norm_L2_sq)  # Относительная ОИСКО

delta_n = np.array(delta_n)
best_N = N_oisco[np.argmin(delta_n)]

# ====================== КРАСИВАЯ ТАБЛИЦА ОИСКО ======================
print("\n" + "=" * 30)
print(" N |  ОИСКО δ_n(N)")
print("-" * 30)
for N, d in zip(N_oisco, delta_n):
    print(f"{N:2d} |  {d:.6f}")
print("=" * 30)

# ====================== ПРОЕКЦИОННЫЕ ОЦЕНКИ ======================
# Выбираем 5 значений N для отрисовки графиков проекции (по аналогии со старым кодом)
N_plot = [10, 15, 20, 30, 40]
x_plot = np.linspace(0, 4.5, 1000)
f_true = f(x_plot)

f_hats = []
for N in N_plot:
    f_hat = np.zeros_like(x_plot)
    for i in range(N + 1):
        f_hat += hat_c[i] * phi(i, x_plot)
    f_hats.append(f_hat)

# ====================== ГРАФИКИ: 2 СТРОКИ ПО 3 ======================
fig, axs = plt.subplots(2, 3, figsize=(18, 11))

# Первая строка: N=10, 15, 20
for idx, N in enumerate(N_plot[:3]):
    ax = axs[0, idx]
    ax.plot(x_plot, f_true, 'r-', lw=2.5, label='Теоретическая f(x)')
    ax.plot(x_plot, f_hats[idx], 'b--', lw=2.2, label=f'$\\widehat{{f}}_{{{N}}}(x)$')
    ax.set_title(f'Проекционная оценка, N = {N}', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

# Вторая строка: N=30, 40 + График ОИСКО
for idx, N in enumerate(N_plot[3:]):
    ax = axs[1, idx]
    ax.plot(x_plot, f_true, 'r-', lw=2.5, label='Теоретическая f(x)')
    ax.plot(x_plot, f_hats[3 + idx], 'b--', lw=2.2, label=f'$\\widehat{{f}}_{{{N}}}(x)$')
    ax.set_title(f'Проекционная оценка, N = {N}', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

# График ОИСКО (правый нижний)
ax_delta = axs[1, 2]
ax_delta.plot(N_oisco, delta_n, 'o-', color='purple',
              markersize=6, linewidth=2.5, label=r'$\overline{\delta}_n(N)$')

min_idx = np.argmin(delta_n)
ax_delta.plot(N_oisco[min_idx], delta_n[min_idx], 'o', color='green',
              markersize=12, label=f'Минимум при N={N_oisco[min_idx]}')

ax_delta.set_title('Зависимость ОИСКО $\\overline{\\delta}_n$ от N', fontsize=13)
ax_delta.set_xlabel('Число членов разложения N')
ax_delta.set_ylabel(r'$\overline{\delta}_n(N)$')
ax_delta.grid(True, alpha=0.3)
ax_delta.set_xticks(N_oisco)
ax_delta.legend(fontsize=11)

plt.tight_layout()
plt.savefig('graphics/graphic_3.png', dpi=300, bbox_inches='tight')
plt.show()

# ====================== ИТОГ ======================
print(f"\n{'=' * 50}")
print(f"МИНИМУМ ОИСКО достигается при N = {best_N}")
print(f"Значение: δ_n = {delta_n[min_idx]:.6f}")