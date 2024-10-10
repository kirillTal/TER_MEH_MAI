import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

Scale = 5 # Устанавливаем масштаб для графиков
Pi = math.acos(-1)  # Вычисление Pi с помощью арккосинуса от -1

# Функция для поворота координат в 2D на угол Alpha
def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)  # Новый X после поворота
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)  # Новый Y после поворота
    return RX, RY

# Объявляем символическую переменную времени t и временной массив T
t = sp.Symbol('t')
T = np.linspace(0, 10, 1000)  # Временной интервал с 1000 точками

# Определяем закон движения в полярных координатах
r = 1 + 1.5 * sp.sin(12 * t)  # Радиус как функция от времени
phi = 1.2 * t + 0.2 * sp.cos(12 * t)  # Угловая скорость как функция от времени
# Перевод в декартову систему
x = r * sp.cos(phi)  # Координата x в декартовой системе
y = r * sp.sin(phi)  # Координата y в декартовой системе

# Вычисляем производные по времени для скоростей и ускорения
Vx = sp.diff(x, t)  # Проекция скорости по X
Vy = sp.diff(y, t)  # Проекция скорости по Y
Ax = sp.diff(Vx, t)  # Проекция ускорения по X
Ay = sp.diff(Vy, t)  # Проекция ускорения по Y
w = sp.diff(phi, t)  # Угловая скорость

# Преобразование символических выражений в функции для численного вычисления
F_x = sp.lambdify(t, x, 'numpy') # координата x
F_y = sp.lambdify(t, y, 'numpy') # координата y
F_Vx = sp.lambdify(t, Vx, 'numpy') # Проекция скорости по X
F_Vy = sp.lambdify(t, Vy, 'numpy') # Проекция скорости по Y
F_Ax = sp.lambdify(t, Ax, 'numpy') # Проекция ускорения по X
F_Ay = sp.lambdify(t, Ay, 'numpy') # Проекция ускорения по Y
F_w = sp.lambdify(t, w, 'numpy') # Угловая скорость

# Вычисление значений координат, скорости и ускорения для всех моментов времени
X = F_x(T)
Y = F_y(T)
VX = F_Vx(T)
VY = F_Vy(T)
AX = F_Ax(T)
AY = F_Ay(T)
W = F_w(T)

#если производная от phi равняется константе, то W не будет массивом, отлавливаем этот случай для корректной работы
if type(W) == float:
    W = np.full_like(T, W)

# Создание фигуры для графиков
fig = plt.figure()

# Добавление субплота, на котором будет строиться анимация
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')  # Одинаковый масштаб по осям
ax1.set(xlim=[-Scale, Scale], ylim=[-Scale, Scale])  # Установка границ осей

# Начальное положение объекта (точки)
P, = ax1.plot(X[0], Y[0])

# Шаблон стрелки для отображения направления
ArrowX = np.array([-0.2, 0, -0.2])
ArrowY = np.array([0.1, 0, -0.1])

# Отрисовка линии радиус-вектора
RVLine, = ax1.plot([0, X[0]], [0, Y[0]], 'grey')

# Построение стрелки для радиус-вектора
RVArrowX, RVArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[0], X[0])) #разворот шаблона стрелки
RVArrow, = ax1.plot(RVArrowX+X[0], RVArrowY+Y[0], 'grey') #отрисовка со сдвигом из точки 0, 0

# Отрисовка линии вектора скорости
VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], 'red')

# Стрелка для скорости
VArrowX, VArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0])) #разворот шаблона стрелки
VArrow, = ax1.plot(VArrowX+X[0]+VX[0], VArrowY+Y[0]+VY[0], 'red') #отрисовка со сдвигом из точки 0, 0

# Отрисовка линии вектора ускорения
ALine, = ax1.plot([X[0], X[0]+AX[0]], [Y[0], Y[0]+AY[0]], 'green')

# Стрелка для ускорения
AArrowX, AArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[0], AX[0])) #разворот шаблона стрелки
AArrow, = ax1.plot(AArrowX+X[0]+AX[0], AArrowY+Y[0]+AY[0], 'green') #отрисовка со сдвигом из точки 0, 0

# Построение линии радиуса кривизны
RX, RY = Rot2D(X[0]+VX[0]/W[0], Y[0]+VY[0]/W[0], Pi/2)
RLine, = ax1.plot([X[0], RX], [Y[0], RY], 'black')

# Стрелка для радиуса кривизны
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(RY, RX))
RArrow, = ax1.plot(RArrowX+RX, RArrowY+RY, 'black')

# Функция для обновления данных на каждом кадре анимации
def anima(i):
    P.set_data(X, Y)  # Обновление положения точки
    RVLine.set_data([0, X[i]], [0, Y[i]])  # Обновление радиус-вектора
    RVArrowX, RVArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RVArrow.set_data(RVArrowX+X[i], RVArrowY+Y[i])
    VLine.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])  # Вектор скорости
    VArrowX, VArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(VArrowX+X[i]+VX[i], VArrowY+Y[i]+VY[i])
    ALine.set_data([X[i], X[i]+AX[i]], [Y[i], Y[i]+AY[i]])  # Вектор ускорения
    AArrowX, AArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    AArrow.set_data(AArrowX+X[i]+AX[i], AArrowY+Y[i]+AY[i])
    RX, RY = Rot2D(VX[i] / W[i], VY[i] / W[i], Pi / 2)  # Радиус кривизны
    RLine.set_data([X[i], X[i]+RX], [Y[i], Y[i]+RY])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(RY, RX))
    RArrow.set_data(RArrowX+X[i]+RX, RArrowY+Y[i]+RY)
    return P, VLine, VArrow, ALine, AArrow, RLine, RArrow

# Анимация с 1000 кадрами и интервалом в 100 мс между кадрами
anim = FuncAnimation(fig, anima, frames=1000, interval=100, repeat=False)

# Показываем график
plt.show()