import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Сircle1(X, Y):
    CX = [X + R * math.cos(i / 100) for i in range(0, 628)]
    CY = [Y + R * math.sin(i / 100) for i in range(0, 628)]
    return CX, CY

def Сircle2(X, Y):
    CX = [X + r * math.cos(i / 100) for i in range(0, 628)]
    CY = [Y + r * math.sin(i / 100) for i in range(0, 628)]
    return CX, CY

def Сircle3(X, Y):
    CX = [X + r * 0.3 * math.cos(i / 100) for i in range(0, 628)]
    CY = [Y + r * 0.3 * math.sin(i / 100) for i in range(0, 628)]
    return CX, CY

def anima(i):
    Beam_1.set_data([X20[i], X30[i]], [Y20[i], Y30[i]])
    Beam_2.set_data([X10[i], X20[i]], [Y10[i], Y20[i]])
    Beam_3.set_data([X20[i], X20[i]], [Y20[i], -0.9])
    circle1.set_data(*Сircle1(X10[i], Y10[i]))
    circle2.set_data(*Сircle2(X20[i], Y20[i]))
    circle3.set_data(*Сircle3(X30[i], Y30[i]))
    return circle1, circle2, circle3, Beam_1, Beam_2, Beam_3

# Определение параметров
t = sp.Symbol('t')
m1 = 2.0
m2 = 1.0
r = 0.1
R = 0.5
l = 0.5
g = 9.81

phi = sp.sin(math.pi / 6 * t)
dphi = sp.diff(phi, t)
ddphi = sp.diff(dphi, t)
omega = sp.sin(math.pi / 4 * t)
domega = sp.diff(omega, t)
ddomega = sp.diff(domega, t)

x2 = (R - r) * sp.sin(ddomega)
y2 = -(R - r) * sp.cos(ddomega)
x3 = x2 + l * sp.sin(ddphi * sp.cos(phi - omega) + dphi ** 2 * sp.sin(phi - omega))
y3 = y2 - l * sp.cos(ddphi * sp.cos(phi - omega) + dphi ** 2 * sp.sin(phi - omega))

T = np.linspace(0, 50, 800)
X10 = np.zeros_like(T)
Y10 = np.zeros_like(T)
X20 = np.zeros_like(T)
Y20 = np.zeros_like(T)
X30 = np.zeros_like(T)
Y30 = np.zeros_like(T)

for i in np.arange(len(T)):
    X20[i] = sp.Subs(x2, t, T[i])
    Y20[i] = sp.Subs(y2, t, T[i])
    X30[i] = sp.Subs(x3, t, T[i])
    Y30[i] = sp.Subs(y3, t, T[i])

# Создание фигуры и анимации
fig, ax1 = plt.subplots()
ax1.axis('equal')
ax1.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
ax1.plot([0, 0], [-0.6, 0], linestyle='--', linewidth=1, color='grey')

Beam_1, = ax1.plot([], [], 'black')
Beam_2, = ax1.plot([], [], linestyle='--', linewidth=1, color='grey')
Beam_3, = ax1.plot([], [], linestyle='--', linewidth=1, color='grey')
circle1, = ax1.plot([], [], 'black')
circle2, = ax1.plot([], [], 'black')
circle3, = ax1.plot([], [], 'black')

anim = FuncAnimation(fig, anima, frames=800, interval=10, blit=True)
plt.show()