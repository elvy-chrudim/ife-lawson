import sympy as sy
import numpy as np

kB = 1.0
Gammai = 1.0
Gammae = 1.0
Gammaa = 1.0
me = 1.0
mi = 1.0
ma = 1.0

ni, ne, na, Ti, Ms, vs, Th, R = sy.symbols('ni, ne, nalpha, Ti, Ms, vs, Th, R')

Y = sy.Matrix([ni, ne, na, Ti, Ms, vs, Th, R])

def vth(T, m):
    return (kB * T / m)**0.5

def Kn(n, T, m, R, Gamma):
    return vth(T, m)**4.0 / Gamma / n / R

def vl(n, T, m, R, Gamma):
    return vth(T, m) * Kn(n, T, m, R, Gamma) * (2.0 / sy.pi)**0.5

def press(n, T):
    return n * kB * T

def eps(n, T):
    return 3.0 / 2.0 * n * kB * T

def mass(n, m):
    return m * n

def energy(m, vd, n, T, R, Gamma):
    return 1.0 / 2.0 * m * vd**2.0 + m * n * vl(n, T, m, R, Gamma) * vd + eps(n, T)

G = sy.Matrix([mass(ni, mi), energy(mi, vs, ni, Ti, R, Gammai), mass(ne, me), energy(me, vs, ne, Ti, R, Gammae), mass(na, ma), energy(ma, vs, na, Ti, R, Gammaa)])

G = sy.Matrix([ni, ne, na, Ti, Ms, vs, Th, R])

J = G.jacobian(Y)

print(J)

Jnp = sy.lambdify(Y, J, "numpy")

print(Jnp(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

print(np.linalg.inv(Jnp(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
