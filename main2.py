import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

# Определение параметров
m1 = 2.0
m2 = 1.0
r = 0.1
R = 0.5
l = 0.5
g = 9.81
t = sp.Symbol('t')

# Обобщенные координаты
phi = sp.Function('phi')(t)
theta = sp.Function('theta')(t)

# Скорости и ускорения
phi_dot = sp.diff(phi, t)
phi_ddot = sp.diff(phi_dot, t)
theta_dot = sp.diff(theta, t)
theta_ddot = sp.diff(theta_dot, t)

# Координаты точек
x2 = (R - r) * sp.sin(theta)
y2 = -(R - r) * sp.cos(theta)
x3 = x2 + l * sp.sin(phi)
y3 = y2 - l * sp.cos(phi)

# Кинетическая энергия
T1 = (1/2) * m1 * (sp.diff(x2, t)**2 + sp.diff(y2, t)**2)
T2 = (1/2) * m2 * (sp.diff(x3, t)**2 + sp.diff(y3, t)**2)
T = T1 + T2

# Потенциальная энергия
V1 = m1 * g * y2
V2 = m2 * g * y3
V = V1 + V2

# Лагранжиан
L = T - V

# Уравнения Лагранжа
lagrange_phi = sp.diff(sp.diff(L, phi_dot), t) - sp.diff(L, phi)
lagrange_theta = sp.diff(sp.diff(L, theta_dot), t) - sp.diff(L, theta)

# Решение уравнений движения
solutions = sp.solve([lagrange_phi, lagrange_theta], (phi_ddot, theta_ddot))
phi_ddot_expr = solutions[phi_ddot]
theta_ddot_expr = solutions[theta_ddot]

# Дискретизация времени
T_vals = np.linspace(0, 10, 500)
phi_vals = np.zeros_like(T_vals)
theta_vals = np.zeros_like(T_vals)
phi_dot_vals = np.zeros_like(T_vals)
theta_dot_vals = np.zeros_like(T_vals)

# Начальные условия
phi_vals[0] = np.pi / 6
theta_vals[0] = np.pi / 4
phi_dot_vals[0] = 0
theta_dot_vals[0] = 0

# интегрирование
dt = T_vals[1] - T_vals[0]
for i in range(1, len(T_vals)):
    phi_ddot_val = float(phi_ddot_expr.subs({phi: phi_vals[i-1], theta: theta_vals[i-1], phi_dot: phi_dot_vals[i-1], theta_dot: theta_dot_vals[i-1]}))
    theta_ddot_val = float(theta_ddot_expr.subs({phi: phi_vals[i-1], theta: theta_vals[i-1], phi_dot: phi_dot_vals[i-1], theta_dot: theta_dot_vals[i-1]}))

    phi_dot_vals[i] = phi_dot_vals[i-1] + phi_ddot_val * dt
    theta_dot_vals[i] = theta_dot_vals[i-1] + theta_ddot_val * dt
    phi_vals[i] = phi_vals[i-1] + phi_dot_vals[i] * dt
    theta_vals[i] = theta_vals[i-1] + theta_dot_vals[i] * dt

# Координаты для анимации
x2_vals = (R - r) * np.sin(theta_vals)
y2_vals = -(R - r) * np.cos(theta_vals)
x3_vals = x2_vals + l * np.sin(phi_vals)
y3_vals = y2_vals - l * np.cos(phi_vals)

# Функции для рисования окружностей
def circle(X, Y, radius):
    angles = np.linspace(0, 2 * np.pi, 100)
    return X + radius * np.cos(angles), Y + radius * np.sin(angles)

# Анимация
def update(frame):
    Beam_1.set_data([x2_vals[frame], x3_vals[frame]], [y2_vals[frame], y3_vals[frame]])
    Beam_2.set_data([0, x2_vals[frame]], [0, y2_vals[frame]])
    circle1.set_data(*circle(0, 0, R))
    circle2.set_data(*circle(x2_vals[frame], y2_vals[frame], r))
    circle3.set_data(*circle(x3_vals[frame], y3_vals[frame], r * 0.3))
    return Beam_1, Beam_2, circle1, circle2, circle3

# Построение графиков
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Анимация слева
ax[0].axis('equal')
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)
Beam_1, = ax[0].plot([], [], 'b')
Beam_2, = ax[0].plot([], [], 'g')
circle1, = ax[0].plot([], [], 'r')
circle2, = ax[0].plot([], [], 'orange')
circle3, = ax[0].plot([], [], 'purple')

ani = FuncAnimation(fig, update, frames=len(T_vals), interval=20, blit=True)

# Графики обобщённых координат справа
ax[1].plot(T_vals, phi_vals, label='$ph(t)$')
ax[1].plot(T_vals, theta_vals, label='$\theta(t)$')
ax[1].set_xlabel('Время (с)')
ax[1].set_ylabel('Угол (рад)')
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()
