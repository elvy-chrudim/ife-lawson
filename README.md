# ife-lawson
Simplified 1D spherical model of capsule compression and burn evaluating the Lawson criterion

# IFE-LAWSON-MODEL
Sufficient, but simplistic model of inertial fusion energy (IFE) burn.

https://en.wikipedia.org/wiki/Lawson_criterion

Wroking 1) through 4)

# TODOs:
0) Decide how to type math equations

1) First, define the model in terms of Euler equations in 3D.
   Red square "Euler equation(s) (original conservation or Eulerian form)"
   https://en.wikipedia.org/wiki/Euler_equations_(fluid_dynamics)

2) Second, transform to spherical coordinates (use spherical form of divergence) and switch to 1D problem in `R`.
   https://mathworld.wolfram.com/SphericalCoordinates.html
   
3) Apply assumptions: `n`, `T`, and derived (ideal gas equation of state) pressure `p * V = n * k_B * T` and internal energy density `\epsilon = 3/2 * p` are constant within the fuel capsule while the radius `R` of the capsule varies in time `R(t)`.

4) Use calculus integration by parts to switch to a system of two ordinary differential equations for `n` and `T`.

5) Add model equations for `R(t)`, where `R` is the position of the inward flying/compressing shell of mass `m_I`. The equations come from Newton's second law: `dR(t)dt = v_I(t), m_I dv_I(t)dt = \int p(R(t)) dS`, where `p(R(t)) = n(t) k_B T(t) / V`.

6) Solve the set of ODEs using numerical solutin via python
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

# INTRODUCTION
We aim to model compression of the deuterium-tritium fuel (capsule) by an inward-flying shell. Most importantly, we need to model particle density $n_f$, temperature $T_f$, and radius $R_f$ of the fuel (DT-capsule) to understand the conditions of fusion burn, and more precisely, to demonstrate the dynamics of **Lawson-criterion** of competition between the $\alpha$-heating and fuel leakage through thermal transport via thermal velocity $`v_{th}`$.

# Boltzmann kinetics model
$`\frac{\partial}{\partial t} ( \int \psi(\mathbf{v}) f d^3v) + \nabla \cdot ( \int \psi(\mathbf{v}) \mathbf{v} f(\mathbf{v}) d^3v) \overset{\int dV}{\Rightarrow} \int \left( \frac{\partial}{\partial t} ( \int \psi(\mathbf{v}) f(\mathbf{v}) d^3v) \right) dV + \int \left( \int \psi(\mathbf{v}) f(\mathbf{v}) \mathbf{v} \cdot \mathbf{n} d^3v \right) dS`$ 

*Assumption-1* Drift velocity $\mathbf{v}_d$ and microscopic velocity $\mathbf{w}$ in the drift frame decomposition

$`\overset{\mathbf{v} = \mathbf{v}_d + \mathbf{w}}{=} \int \left( \frac{\partial}{\partial t} ( \int \psi(\mathbf{v}_d + \mathbf{w}) f(\mathbf{v}_d + \mathbf{w}) d^3\mathbf{w}) \right) dV +`$ 
$`\int \left( \int \psi(\mathbf{v}_d + \mathbf{w}) f(\mathbf{v}_d + \mathbf{w}) d^3\mathbf{w} \right) \mathbf{v}_d \cdot \mathbf{n} dS + \int \left( \int \psi(\mathbf{v}_d + \mathbf{w}) f(\mathbf{v}_d + \mathbf{w}) \mathbf{w} \cdot \mathbf{n} d^3\mathbf{w} \right) dS`$

*Assumption-2* Thermalized and isotropic distribution function (Maxwell-Boltzmann) in co-moving frame
$`\overset{^{f(\mathbf{v}) = f_M(|\mathbf{v} - 
\mathbf{v}_f|)}_{\Rightarrow f_M(|\mathbf{w}|)}}{=}
\int \left( \frac{\partial}{\partial t} ( \int \psi(\mathbf{v}_d + \mathbf{w}) f_M(|\mathbf{w}|) d^3\mathbf{w}) \right) dV +`$ 
$`\int \left( \int \psi(\mathbf{v}_d + \mathbf{w}) f_M(|\mathbf{w}|) d^3\mathbf{w} \right) \mathbf{v}_d \cdot \mathbf{n} dS + \int \left( \int \psi(\mathbf{v}_d + \mathbf{w}) f_M(|\mathbf{w}|) \mathbf{w} \cdot \mathbf{n} d^3\mathbf{w} \right) dS`$

*Assumption-3* Switch to spherical coordiantes of $\mathbf{w}$ given the azimuthal symmetry assumption, hence variation in the pitch angle with respect to the radial direction

$`\overset{_{\frac{\partial f}{\partial \omega} = 0}}{=} 2\pi \Big[ \int \left( \frac{\partial}{\partial t} ( \int \psi(v_d, w, \mu) f_M(w) w^2 dw d\mu) \right) dV +`$ 
$`\int \left( \int_{\mu} \psi(v_d, w, \mu) f_M(w) w^2 dw d\mu \right) v_d dS + \int \left( \int_{\mu} \psi(v_d, w, \mu) f_M(w) w \mu w^2 dw d\mu \right) dS \Big]`$

*Assumption-4* The boundary layer effect (isotropic vs. streaming transport close to the sphere surface) can be modeled using the Knudsen number $K_n(n, T, R) = \frac{\lambda(n, T)}{R}$, allowing us to include *thermal* leakage through the outer boundary of the fuel
 
$`\overset{_{f_M^{K_n}(\mu<0)dS \rightarrow 0}^{K_n \rightarrow 1}}{=} 2\pi \Big[ \int \frac{\partial}{\partial t} \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} \psi(v_d, w, \mu) f_M^{K_n}(w) w^2 dw d\mu \right) dV +`$ 
$`\int \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} \psi(v_d, w, \mu) f_M^{K_n}(w) w^2 dw d\mu \right) v_d dS + \int \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} \mu \psi(v_d, w, \mu) f_M^{K_n}(w) w^3 dw d\mu \right) dS \Big], \quad (p0)`$

where the Knudsen number $K_n$ spans the values in $(0, 1)$, taking value $0$ when highly collisional ($\lambda \ll R$) and value $1$ when streaming $\lambda = R$. 

Now, we will use *mass, momentum, and energy invariants* $`\psi(\mathbf{v}_d + \mathbf{w}) = (m, m (\mathbf{v}_d + \mathbf{w}), \frac{m}{2} (\mathbf{v}_d + \mathbf{w}) \cdot (\mathbf{v}_d + \mathbf{w}))`$  in (p0) to derive Euler equations in integral (surface divergence) form. Expressing $`\mathbf{v}_d + \mathbf{w}`$ in local Cartesian coordinates ($z$-axis aligned with drift velocity $`\mathbf{v}_d`$), we write $`\mathbf{v}_d = [0, 0, v_d]`$ and 
$`\mathbf{w} = [w \cos(\phi) \sqrt{1 - \mu^2}, w \sin(\phi) \sqrt{1 - \mu^2}, w \mu]`$, hence $`\mathbf{v}_d + \mathbf{w} = [w \cos(\phi) \sqrt{1 - \mu^2}, w \sin(\phi) \sqrt{1 - \mu^2}, v_d + w \mu]`$.

**Three invariants we will apply** then read $`\psi_{n} = m`$, $`\psi_{p_{z}} = m (v_d + w \mu)`$, and $`\psi_{T} = \frac{m}{2}(v_d^2 + 2 v_d w \mu + w^2)`$.

The **MASS MOMENT** of the Boltzmann model reads

$`2\pi \Big[ \int \frac{\partial}{\partial t} \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} m f_M^{K_n}(w) w^2 dw d\mu \right) dV +`$ 
$`\int \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} m f_M^{K_n}(w) w^2 dw d\mu \right) v_d dS + \int \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} m \mu f_M^{K_n}(w) w^3 dw d\mu \right) dS \Big] = 0, \quad (ppp1)`$

The **MOMENTUM MOMENT** of the Boltzmann model reads

$`2\pi \Big[ \int \frac{\partial}{\partial t} \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} m (v_d + w \mu) f_M^{K_n}(w) w^2 dw d\mu \right) dV +`$ 
$`\int \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} m (v_d + w \mu) f_M^{K_n}(w) w^2 dw d\mu \right) v_d dS + \int \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} m \mu (v_d + w \mu) f_M^{K_n}(w) w^3 dw d\mu \right) dS \Big] = 0, \quad (ppp2)`$

The **ENERGY MOMENT** of the Boltzmann model reads

$`2\pi \Big[ \int \frac{\partial}{\partial t} \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} \frac{m}{2}(v_d^2 + 2 v_d w \mu + w^2) f_M^{K_n}(w) w^2 dw d\mu \right) dV +`$ 
$`\int \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} \frac{m}{2}(v_d^2 + 2 v_d w \mu + w^2) f_M^{K_n}(w) w^2 dw d\mu \right) v_d dS + \int \left( \int_{K_n - 1}^{1} \int_{0}^{\infty} \frac{m}{2} \mu (v_d^2 + 2 v_d w \mu + w^2) f_M^{K_n}(w) w^3 dw d\mu \right) dS \Big] = 0. \quad (ppp3)`$

Notes: 

- $\lambda = \min(\lambda, 1)$ needs to be used in the implementation.

- *pseudo-equilibrium* distribution $f_M^{K_n}(w, n, T) = \gamma(n, T) f_M(w, n, T)$ is a scaled Maxwell Boltzmann by a *conservation factor* $\gamma(n, T, R) = \frac{2}{\int_{K_n(n, T, R) - 1}^1 d\mu} = \frac{1}{1 - \frac{K_n}{2}}$ (keep in mind that $`\frac{\partial f_M^{K_n}}{\partial \mu} = 0`$)

Math hints angle $\mu$: `https://www.wolframalpha.com/input?i=int+%28v%5E2+%2B+2+*+v+*+w+*+x+%2B+w%5E2%29+dx%2C+x%3DK-1..1`

- mass moment integration $`\int_{K_n - 1}^{1} d\mu = 2 (1 - \frac{K_n}{2}) = \frac{2}{\gamma}`$

- mass moment integration $`\int_{K_n - 1}^{1} \mu d\mu = K_n (1 - \frac{K_n}{2}) = \frac{2}{\gamma}\frac{K_n}{2}`$

- momentum moment integration $`\int_{K_n - 1}^{1} (v_d + w \mu) d\mu = 2 (1 - \frac{K_n}{2}) \frac{2 v_d + K_n w}{2} = \frac{2}{\gamma} \frac{2 v_d + K_n w}{2}`$
  
- momentum moment integration $`\int_{K_n - 1}^{1} \mu (v_d + w \mu) d\mu = 2 (1 - \frac{K_n}{2}) \frac{3 K_n v_d + 2 (K_n^2 - K_n + 1) w}{6} = \frac{2}{\gamma} \frac{3 K_n v_d + 2 (K_n^2 - K_n + 1) w}{6}`$
  
- energy moment integration $`\int_{K_n - 1}^{1} (v_d^2 + 2 v_d w \mu + w^2) d\mu = 2 (1 - \frac{K_n}{2}) (v_d^2 + K_n v_d w + w^2) = \frac{2}{\gamma} (v_d^2 + K_n v_d w + w^2)`$

- energy moment integration $`\int_{K_n - 1}^{1} \mu (v_d^2 + 2 v_d w \mu + w^2) d\mu = 2 (1 - \frac{K_n}{2}) \frac{4 (K_n^2 - K_n + 1) v_d w + 3 K_n v_d^2 + 3 K_n w^2}{6} = \frac{2}{\gamma} \frac{3 K_n v_d^2 + 4 (K_n^2 - K_n + 1) v_d w + 3 K_n w^2}{6}`$

The **MASS MOMENT** of the Boltzmann model reads

$`4\pi m \Big[ \int \frac{\partial}{\partial t} \left( \int_{0}^{\infty} f_M(w) w^2 dw \right) dV +`$ 
$`\int \left( \int_{0}^{\infty} f_M(w) w^2 dw \right) v_d dS + \int \left( \int_{0}^{\infty} \frac{K_n}{2} f_M(w) w^3 dw \right) dS \Big] = 0, \quad (pp1)`$

The **MOMENTUM MOMENT** of the Boltzmann model reads

$`4\pi m \Big[ \int \frac{\partial}{\partial t} \left( \int_{0}^{\infty} (v_d + \frac{K_n}{2} w) f_M(w) w^2 dw \right) dV +`$ 
$`\int \left( \int_{0}^{\infty} (v_d + \frac{K_n}{2} w) f_M(w) w^2 dw \right) v_d dS + \int \left( \int_{0}^{\infty} \frac{K_n v_d}{2} + \frac{(K_n^2 - K_n + 1)}{3} w) f_M(w) w^3 dw \right) dS \Big] = 0, \quad (pp2)`$

The **ENERGY MOMENT** of the Boltzmann model reads

$`4\pi \frac{m}{2} \Big[ \int \frac{\partial}{\partial t} \left( \int_{0}^{\infty} (v_d^2 + K_n v_d w + w^2) f_M(w) w^2 dw \right) dV +`$ 
$`\int \left( \int_{0}^{\infty} (v_d^2 + K_n v_d w + w^2) f_M(w) w^2 dw \right) v_d dS + \int \left( \int_{0}^{\infty} (\frac{K_n v_d^2}{2} + \frac{2(K_n^2 - K_n + 1) v_d}{3} w + \frac{K_n}{2} w^2) f_M(w) w^3 dw \right) dS \Big] = 0. \quad (pp3)`$

Math hints microscopic velocity $w$: `https://www.wolframalpha.com/input?i=int+4+pi+%2F+%282+pi%29%5E%283%2F2%29+%2F+a%5E3+t%5E5+e%5E%28-%28t%2Fa%29%5E2%2F2%29+dt%2C+t%3D0..infinity`

- Targeting $`\int_0^\infty w^\xi f_M(w) dw`$ for $\xi = 2, 3, 4, 5$, where $`f_M(w, n, v_{th}) = \frac{n}{(2\pi)^{\frac{3}{2}} v_{th}^3} \exp(-\frac{1}{2}\frac{w^2}{v_{th}^2})`$, where $`v_{th} = \sqrt{\frac{k_B T}{m}}`$.

- $`4\pi \int_0^\infty w^2 f_M(w) dw = n`$

- $`4\pi \int_0^\infty w^3 f_M(w) dw = 2 \sqrt{\frac{2}{\pi}} n v_{th}`$

- $`4\pi \int_0^\infty w^4 f_M(w) dw = 3 n v_{th}^2`$

- $`4\pi \int_0^\infty w^5 f_M(w) dw = 8 \sqrt{\frac{2}{\pi}} n v_{th}^3`$

The **MASS MOMENT** of the Boltzmann model reads

$`\int \frac{\partial}{\partial t} \left( m n \right) dV +`$ 
$`\int \left( m n \right) v_d dS + \int \left( K_n \sqrt{\frac{2}{\pi}} m n v_{th} \right) dS = 0, \quad (p1)`$

The **MOMENTUM MOMENT** of the Boltzmann model reads

$`\int \frac{\partial}{\partial t} \left( m n v_d + K_n \sqrt{\frac{2}{\pi}} m n v_{th} \right) dV +`$ 
$`\int \left( m n v_d + K_n \sqrt{\frac{2}{\pi}} m n v_{th} \right) v_d dS + \int \left( K_n v_d \sqrt{\frac{2}{\pi}} m n v_{th} + (K_n^2 - K_n + 1) m n v_{th}^2 \right) dS = 0, \quad (p2)`$

The **ENERGY MOMENT** of the Boltzmann model reads

$`\int \frac{\partial}{\partial t} \left( \frac{1}{2} m n v_d^2 + K_n v_d \sqrt{\frac{2}{\pi}} m n v_{th} + \frac{3}{2} m n v_{th}^2 \right) dV +`$ 
$`\int \left( \frac{1}{2} m n v_d^2 + K_n v_d \sqrt{\frac{2}{\pi}} m n v_{th} + \frac{3}{2} m n v_{th}^2 \right) v_d dS + \int \left( K_n v_d^2 \sqrt{\frac{2}{\pi}} \frac{1}{2} m n v_{th} + (K_n^2 - K_n + 1) v_d m n v_{th}^2 + 2 K_n \sqrt{\frac{2}{\pi}} m n v_{th}^3 \right) dS = 0. \quad (p3)`$

## Nonlocal $K_n$-model

The proposed model captures transition from a "pure" fluid to a streaming transport behavior of a finite medium within a sphere, where the transition between the two behaviors is captured by the Knudsen number $K_n$ comparing the mean free path of particles with respect to the diametr $R$ of the sphere.

Definitions:

- thermal velocity $`v_{th} = \sqrt{\frac{k_B T}{m}}`$

- Knudsen number $`K_n = \frac{\lambda}{R} = \frac{v_{th}^4}{\sum_\beta \left(\frac{4\pi q^2 q_\beta^2 \Lambda}{m^2} n_\beta \right) R}`$

- "leakage" velocity $`v_l = K_n \sqrt{\frac{2}{\pi}} v_{th}`$

- pressure $`p = n k_B T`$

- internal energy density $`\epsilon = \frac{3}{2} n k_B T`$

where particle mass $m$, Boltzmann constant $k_B$, and cross section $\Gamma$ are constant to be defined in correct units.

Application of the Reynold's transport theorem (https://en.wikipedia.org/wiki/Reynolds_transport_theorem) $`\frac{d}{dt} (\int F dV) = \int \frac{d F}{dt} dV + \int F (\mathbf{v}_f \cdot \mathbf{n}) dS`$ to $(p1)$, $(p2)$, and $(p3)$ with the definitions above lead to a simple model

**MASS CONSERVATION**

$`\frac{\partial}{\partial t} \int m n dV + \int m n v_l dS = \int S_M(n_\beta, T_\beta) dV, \quad (1)`$

**MOMENTUM CONSERVATION**

$`\frac{\partial}{\partial t} \int m n (v_d + v_l) dV + \int \left( m n v_l v_d + (K_n^2 - K_n + 1) p \right) dS = 0, \quad (2)`$

**ENERGY CONSERVATION**

$`\frac{\partial}{\partial t} \int \left( \frac{1}{2} m n v_d^2 + m n v_l v_d + \epsilon \right) dV + \int \left( \frac{1}{2} m n v_d^2 v_l + (K_n^2 - K_n + 1) p v_d + 2 p v_l \right) dS = \int S_E(n_\beta, T_\beta) dV. \quad (3)`$

## Discrete nonlocal $K_n$-model

Approximation assuming spatially constant $`n, T`$ and linear $`v_d(\rho) = \frac{\rho}{R} v_s`$

**mass conservation**

$`\frac{\partial}{\partial t} \left( \frac{m n 4\pi R^3}{3} \right) = - \left( m n {v_l}_{(m n_\beta, T, R)} \right) 4\pi R^2 + \frac{S_M(n_\beta, T_\beta, ...) 4\pi R^3}{3}, \quad (d1)`$

**momentum conservation**

$`\frac{\partial}{\partial t} \left( \frac{m n {v_l}_{(m, n_\beta, T, R)} 4\pi R^3}{3} + \frac{v_s 4\pi R^3}{4} \right) = - \left( m n {v_l}_{(m, n_\beta, T, R)} v_s + ({K_n}_{(m, n_\beta, T, R)}^2 - {K_n}_{(m, n_\beta, T, R)} + 1) p_{(n, T)} \right) 4\pi R^2 , \quad (d2)`$

**energy conservation**

$`\frac{\partial}{\partial t} \left( \frac{1}{2}\frac{m n v_s^2 4\pi R^3}{5} + \frac{m n {v_l}_{(m, n_\beta, T, R)} v_s 4\pi R^3}{4} + \frac{\epsilon_{(n, T)} 4\pi R^3}{3} \right) = - \left( \frac{1}{2} m n v_s^2 {v_l}_{(m, n_\beta, T, R)} + ({K_n}_{(m, n_\beta, T, R)}^2 - {K_n}_{(m, n_\beta, T, R)} + 1) p_{(n, T)} v_s + 2 p_{(n, T)} {v_l}_{(m, n_\beta, T, R)} \right) 4\pi R^2 + \frac{S_E(n_\beta, T_\beta, ...) 4\pi R^3}{3}. \quad (d3)`$

### Implementation

Vector of unknowns:

$`\mathbf{y} = [n_e, n_i, n_\alpha, T_e, T_i, T_\alpha, M_s, v_s, T_H, R]^T`$

Particle energy function:

$`E_e(\mathbf{y}) = \frac{1}{2}\frac{m_e n_e v_s^2 R^3}{5} + \frac{m_e n_e {v_l}(m_e, q_e, n_e, n_i, n_\alpha, T_e, R) v_s R^3}{4} + \frac{\epsilon(n_e, T_e) R^3}{3}`$

$`E_i(\mathbf{y}) = \frac{1}{2}\frac{m_i n_i v_s^2 R^3}{5} + \frac{m_i n_i {v_l}(m_i, q_i, n_e, n_i, n_\alpha, T_i, R) v_s R^3}{4} + \frac{\epsilon(n_i, T_i) R^3}{3}`$

$`E_\alpha(\mathbf{y}) = \frac{1}{2}\frac{m_\alpha n_\alpha v_s^2 R^3}{5} + \frac{m_\alpha n_\alpha {v_l}(m_\alpha, q_\alpha, n_e, n_i, n_\alpha, T_\alpha, R) v_s R^3}{4} + \frac{\epsilon(n_\alpha, T_\alpha) R^3}{3}`$

Leakage energy flux:

$`Q(m, q, n, T, R, v_s) = \frac{1}{2} m n v_s^2 v_l(m, q, n_e, n_i, n_\alpha, T, R) +`$ 
$`(K_n(m, q, n_e, n_i, n_\alpha, T, R)^2 - K_n(m, q, n_e, n_i, n_\alpha, T, R) + 1) p(n, T) v_s +`$
$`2 p(n, T) v_l(m, q, n_e, n_i, n_\alpha, T, R)`$

Force (pressure):

$`F_s(\mathbf{y}) =
\left( m_e n_e v_l(m_e, q_e, n_e, n_i, n_\alpha, T_e, R) + 
m_i n_i v_l(m_i, q_i, n_e, n_i, n_\alpha, T_i, R) + 
m_\alpha n_\alpha v_l(m_\alpha, q_\alpha, n_e, n_i, n_\alpha, T_\alpha, R) \right) v_s R^2 +`$
$`\Big((K_n(m_e, q_e, n_e, n_i, n_\alpha, T_e, R)^2 - K_n(m_e, q_e, n_e, n_i, n_\alpha, T_e, R) + 1) p(n_e, T_e) +`$ 
$`(K_n(m_i, q_i, n_e, n_i, n_\alpha, T_i, R)^2 - K_n(m_i, q_i, n_e, n_i, n_\alpha, T_i, R) + 1) p(n_i, T_i) +`$
$`(K_n(m_\alpha, q_\alpha, n_e, n_i, n_\alpha, T_\alpha, R)^2 - K_n(m_\alpha, q_\alpha, n_e, n_i, n_\alpha, T_\alpha, R) + 1) p(n_\alpha, T_\alpha) \Big) R^2`$

ODE system $`\frac{d}{dt} \mathbf{G}(\mathbf{y}) = f(\mathbf{y})`$

$`\mathbf{G}(\mathbf{y}) = \begin{bmatrix}
\frac{m_e n_e R^3}{3} \\ 
\frac{m_i n_i R^3}{3} \\ 
\frac{m_\alpha n_\alpha R^3}{3}
\\
E_e(\mathbf{y}) \\
E_i(\mathbf{y}) \\
E_\alpha(\mathbf{y}) 
\\
M_s \\
M_s v_s
\\
C_H T_H + \frac{1}{2} M_s v_s^2 + E_e(\mathbf{y}) + E_i(\mathbf{y}) + E_\alpha(\mathbf{y}) 
\\
R
\end{bmatrix}`$



$`\mathbf{f}(\mathbf{y}) = \begin{bmatrix} - 
m_e n_e v_l(m_e, q_e, n_e, n_i, n_\alpha, T_e, R) R^2 + \frac{S_M^e(n_\beta, T_\beta, ...) R^3}{3} \\ -
m_i n_i v_l(m_i, q_i, n_e, n_i, n_\alpha, T_i, R) R^2 + \frac{S_M^i(n_\beta, T_\beta, ...) R^3}{3} \\ -
m_\alpha n_\alpha v_l(m_\alpha, q_\alpha, n_e, n_i, n_\alpha, T_\alpha, R) R^2 + \frac{S_M^\alpha(n_\beta, T_\beta, ...) R^3}{3}
\\ -
Q(m_e, q_e, n_e, T_e, R, v_s) R^2 + \frac{S_E^e(n_\beta, T_\beta, ...)R^3}{3} \\ -
Q(m_i, q_i, n_i, T_i, R, v_s) R^2 + \frac{S_E^i(n_\beta, T_\beta, ...)R^3}{3} \\ -
Q(m_\alpha, q_\alpha, n_\alpha, T_\alpha, R, v_s) R^2 + \frac{S_E^\alpha(n_\beta, T_\beta, ...)R^3}{3} 
\\
\left( m_e n_e v_l(m_e, q_e, n_e, n_i, n_\alpha, T_e, R) +
m_i n_i v_l(m_i, q_i, n_e, n_i, n_\alpha, T_i, R) +
m_\alpha n_\alpha v_l(m_\alpha, q_\alpha, n_e, n_i, n_\alpha, T_\alpha, R) \right) R^2
\\
F_s(\mathbf{y})
\\
0
\\
v_s
\end{bmatrix}`$

# FUEL fluid model
In terms of inertial fusion energy model, deuterium-tritium ions, electrons, and Helium ($\alpha$ particles) need to be described by the **$K_n$-model**. It is a good approximation to consider that each fluid maintain the same drift velocity, i.e. $v_d^i = v_d^e = v_d^\alpha = v_d$, however, densities $n^i$, $n^e$, and $n^\alpha$, as well as temperatures $T^i$, $T^e$, and $T^\alpha$ deserve their own governing equations: 

**Ions mass conservation**

$`\frac{\partial}{\partial t} \int m^i n^i dV = - \int m^i n^i v_l^i dS - \int 2 \beta {T^i}^2 {n^i}^2 dV, \quad (4)`$

**Ions energy conservation**

$`\frac{\partial}{\partial t} \int \left( \frac{1}{2} m^i n^i v_d^2 + m^i n^i v_l^i v_d + \epsilon^i \right) dV = - \int \left( \frac{1}{2} m^i n^i v_d^2 v_l^i + ({K_n^i}^2 - K_n^i + 1) p^i v_d + 2 p^i v_l^i \right) dS, \quad (5)`$

**Electrons mass conservation**

$`\frac{\partial}{\partial t} \int m^e n^e dV = - \int m^e n^e v_l^e dS, \quad (6)`$

**Electrons energy conservation**

$`\frac{\partial}{\partial t} \int \left( \frac{1}{2} m^e n^e v_d^2 + m^e n^e v_l^e v_d + \epsilon^e \right) dV = - \int \left( \frac{1}{2} m^e n^e v_d^2 v_l^e + ({K_n^e}^2 - K_n^e + 1) p^e v_d + 2 p^e v_l^e \right) dS, \quad (7)`$

**Alphas mass conservation**

$`\frac{\partial}{\partial t} \int m^\alpha n^\alpha dV = - \int m^\alpha n^\alpha v_l^\alpha dS + \int \beta {T^i}^2 {n^i}^2 dV, \quad (8)`$

**Alphas energy conservation**

$`\frac{\partial}{\partial t} \int \left( \frac{1}{2} m^\alpha n^\alpha v_d^2 + m^\alpha n^\alpha v_l^\alpha v_d + \epsilon^\alpha \right) dV = - \int \left( \frac{1}{2} m^\alpha n^\alpha v_d^2 v_l^\alpha + ({K_n^\alpha}^2 - K_n^\alpha + 1) p^\alpha v_d + 2 p^\alpha v_l^\alpha \right) dS + \int E_\alpha \beta {T^i}^2 {n^i}^2 dV, \quad (9)`$

where fusion reaction has been added through the model of fusion burn $\beta n^2 T^2$.

Even though we are not going to model the~drift velocity of ions, electrons, and alphas separately, we will write their momentum conservation equations based on $(2)$ as 

**Ions momentum conservation**

$`\frac{\partial}{\partial t} \int m^i n^i (v_d + v_l^i) dV = - \int \left( m^i n^i v_l^i v_d + ({K_n^i}^2 - K_n^i + 1) p^i \right) dS, \quad (10)`$

**Electrons momentum conservation**

$`\frac{\partial}{\partial t} \int m^e n^e (v_d + v_l^e) dV = - \int \left( m^e n^e v_l^e v_d + ({K_n^e}^2 - K_n^e + 1) p^e \right) dS, \quad (11)`$

**Alphas momentum conservation**

$`\frac{\partial}{\partial t} \int m^\alpha n^\alpha (v_d + v_l^\alpha) dV = - \int \left( m^\alpha n^\alpha v_l^\alpha v_d + ({K_n^\alpha}^2 - K_n^\alpha + 1) p^\alpha \right) dS. \quad (12)`$

# SHELL flight model
The inflight model of the shell simply applies the second Newton's law $M^s a = F$. First, we assume that the mass of the shell is not constant, as it absorbs "leaking" ions, electrons, and alphas from the fuel sphere. This leakage is a sum of surface terms (dS) of mass conservation equations $(4)$, $(6)$, and $(8)$. Also, the composition of the force acting on the flying shell is obtained as a sum of "momentum leakeage" through surface terms (dS) of momentum conservation equations $(10)$, $(11)$, and $(12)$.

The mass conservation and momentum conservation model equations of the shell read:

**Shell mass conservation**

$`\frac{\partial}{\partial t} M_s = \int \left( m^i n^i v_l^i + m^e n^e v_l^e + m^\alpha n^\alpha v_l^\alpha \right) dS, \quad (13)`$

**Shell momentum conservation**

$`\frac{\partial}{\partial t} (M_s v_s) = \int \left( ( m^i n^i v_l^i + m^e n^e v_l^e + m^\alpha n^\alpha v_l^\alpha ) v_d + ({K_n^i}^2 - K_n^i + 1) p^i + ({K_n^e}^2 - K_n^e + 1) p^e + ({K_n^\alpha}^2 - K_n^\alpha + 1) p^\alpha \right) dS. \quad (14)`$

# HOHLRAUM heating model
Last piece of our model is the heating response of the hohlraum to the release of energy through the fusion burn. The approach is to assume that any "leaked energy" from the fuel not converted to the kinetic energy of the shell is absorbed to the hohlraum's internal energy, hence causing the heating of holraum. The heating rate of the hohlraum is obtained as a sum of "energy leakeage" through surface terms (dS) of energy conservation equations $(5)$, $(7)$, and $(9)$. Note that the rate of change of kinetic energy of the shell needs to be subtracted.

**Hohlraum energy conservation**

$`\frac{\partial}{\partial t} ( C_H T^H) = - \frac{\partial}{\partial t} \left( \frac{1}{2} M_s v_s^2 \right) +`$
$`\int \left( \frac{1}{2} ( m^i n^i v_l^i + m^e n^e v_l^e + m^\alpha n^\alpha v_l^\alpha ) v_d^2 + ( ({K_n^i}^2 - K_n^i + 1) p^i + ({K_n^e}^2 - K_n^e + 1) p^e + ({K_n^\alpha}^2 - K_n^\alpha + 1) p^\alpha ) v_d + 2 ( p^i v_l^i + p^e v_l^e + p^\alpha v_l^\alpha ) \right) dS. \quad (15)`$

# Summary IFE-Lawson-model
We assume one fluid ($v_d^i = v_d^e = v_d^\alpha = v_d$) model, and also, we assume one temperature ($T^i = T^e = T^\alpha$) model, which means that we substitute (5) by a sum of ions, electrons, and alphas energy equations where $T^e = T^i$ and $T^\alpha = T^i$

$`\frac{\partial}{\partial t} \int \left( \frac{1}{2} ( m^i n^i + m^e n^e + m^\alpha n^\alpha) v_d^2 + ( m^i n^i v_l^i + m^e n^e v_l^e + m^\alpha n^\alpha v_l^\alpha ) v_d + \epsilon^i + \epsilon^e + \epsilon^\alpha \right) dV =`$ 
$`\int E_\alpha \beta {T^i}^2 {n^i}^2 dV - \int \left( \frac{1}{2} ( m^i n^i v_l^i + m^e n^e v_l^e + m^\alpha n^\alpha v_l^\alpha ) v_d^2 + ( ({K_n^i}^2 - K_n^i + 1) p^i + ({K_n^e}^2 - K_n^e + 1) p^e + ({K_n^\alpha}^2 - K_n^\alpha + 1) p^\alpha) v_d + 2 ( p^i v_l^i + p^e v_l^e + p^\alpha v_l^\alpha ) \right) dS. \quad (16)`$

The set of 9 equations $(4)$, $(6)$, $(8)$ for $n^i$, $n^e$, $n^\alpha$, respectively, $(16)$ for $T^i$, $(13)$ for $M_s$, $(14)$ for $v_s$, and $(15)$ for $T^H$, complemented by

$`v_d(\rho) = \frac{\rho}{R} v_s, \quad (17)`$

$`\frac{\partial}{\partial t} R = v_s. \quad (18)`$

Note:

- The volumetric integration assumes spherical symmetry and uses $\rho$ as its coordinate as $`\int f dV = 4\pi \int_0^R f(\rho) \rho^2 d\rho`$

- The surface integration assumes spherical symmetry hence $`\int f dS = 4 \pi R^2 f(R)`$.

- Quantities $n^i$, $n^e$, $n^\alpha$, $v_l^i$, $v_l^e$, $v_l^\alpha$, $K_n^i$, $K_n^e$, $K_n^\alpha$, $T^i$ are considered spatially constant, and so $`\int_0^R T^i \rho^2 d\rho = T^i \frac{4\pi R^3}{3}`$.  

- Qunatity $v_d(\rho)$ is linearly vayring in space and $`\int v_d dV = \int_0^R \frac{\rho}{R} v_s \rho^2 d\rho = v_s \frac{4\pi R^3}{4}`$ or $`\int v_d^2 dV = \int_0^R \frac{\rho^2}{R^2} v_s^2 \rho^2 d\rho = v_s^2 \frac{4\pi R^3}{5}`$.

- It is useful to substitute electrons energy equation $(7)$ by $`\frac{\partial}{\partial t} T^e = \frac{\partial}{\partial t} T^i`$ and alphas energy equation $(9)$ by $`\frac{\partial}{\partial t} T^\alpha = \frac{\partial}{\partial t} T^i`$ in order to keep space for future "multi-temperature model".

# Implementation

Our equation set has the form

$`\frac{\partial}{\partial t}\mathbf{g}(\mathbf{y}) = \mathbf{f}(\mathbf{y})`$

where $\mathbf{y}$ is a vector of our unknowns, $\mathbf{g}$ is a vector of physical quantities on the left hand sides depeneding on $\mathbf{y}$, and $\mathbf{f}$ is a vector of right hand sides of functions depending on $\mathbf{y}$.

We need to obtain direct $\tilde{\mathbf{f}}$ of right hand sides for $\frac{\partial}{\partial t} \mathbf{y}$, which is done as the following

$`\frac{\partial}{\partial \mathbf{y}} \mathbf{g}(\mathbf{y}) \cdot \frac{\partial \mathbf{y}}{\partial t} = \mathbf{f}(\mathbf{y}) \Rightarrow \frac{\partial \mathbf{y}}{\partial t} = \left( \frac{\partial}{\partial \mathbf{y}} \mathbf{g}(\mathbf{y}) \right)^{-1} \cdot \mathbf{f}(\mathbf{y}) := \tilde{\mathbf{f}}(\mathbf{y})`$

Use `SymPy` to evaluate the Jacobian $\frac{\partial}{\partial \mathbf{y}} \mathbf{g}(\mathbf{y})$ analytically (extending `KnModel.py`)




# OBSOLETE DERIVATIONS

# FUEL - FLUID MODEL
General conservative form of the fuel fluid:
$`\frac{d \mathbf{y}}{dt} + \nabla \cdot \mathbf{F} = \mathbf{0}, \quad (0)`$
and its volume integral
$` \int \frac{d \mathbf{y}}{dt} dV + \int \nabla \cdot \mathbf{F} dV = \mathbf{0}, \quad (1)`$
where
$`\mathbf{y} = 
\begin{bmatrix} 
   m_f n_f \\
   m_f n_f \mathbf{v}_f \\
   C_V 2 n_f k_B T_f + \frac{1}{2} m_f n_f \mathbf{v}_f\cdot\mathbf{v}_f
\end{bmatrix}, \quad (2)`$
and 
$`\mathbf{F} = 
\begin{bmatrix} 
   m_f n_f (\mathbf{v}_f + \mathbf{v}_{th}^i) \\
   m_f n_f \mathbf{v}_f (\mathbf{v}_f + \mathbf{v}_{th}^i) + p_f \mathbf{I} \\ 
   (C_V 2 n_f k_B T_f) (\mathbf{v}_f + \mathbf{v}_{th}) +  (\frac{1}{2} m_f n_f \mathbf{v}_f \cdot \mathbf{v}_f + p_f) \mathbf{v}_f
\end{bmatrix}, \quad (3)`$,
where we double the fuel density (including ions and electrons $`n_f^I = n_f^e = n_f`$) in the energy equation to include both ions and electrons at the same temperature $`T_f`$, but different thermal velocity $`v_{th}^I`$ and $`v_{th}^e`$.

The second term of (1) can be written (divergence theorem https://en.wikipedia.org/wiki/Divergence_theorem) as $`\int \nabla \cdot \mathbf{F} dV = \int \mathbf{F} \cdot \mathbf{n} dS `$ ($`\mathbf{F}`$ defined by (3)), where we use $`\mathbf{v}_f(R) = \mathbf{v}_s`$ (the fuel velocity at the interface equals the shell inflight velocity $\mathbf{v}_s$)
$`
\int
\frac{d}{dt}\begin{bmatrix} 
   m_f n_f \\
   m_f n_f \mathbf{v}_f \\
   C_V 2 n_f k_B T_f + \frac{1}{2} m_f n_f \mathbf{v}_f \cdot \mathbf{v}_f
\end{bmatrix} 
dV`$
$`\overset{\mathbf{v}_f(R) = \mathbf{v}_s}{=} - \int
\begin{bmatrix} 
   m_f n_f (\mathbf{v}_s + \mathbf{v}_{th}^i) \cdot \mathbf{n} \\
   m_f n_f \mathbf{v}_s (\mathbf{v}_s + \mathbf{v}_{th}^i) \cdot \mathbf{n} + p_f \mathbf{I} \cdot \mathbf{n} \\
   (C_V 2 n_f k_B T_f) (\mathbf{v}_s + \mathbf{v}_{th}) \cdot \mathbf{n} +  (\frac{1}{2} m_f n_f \mathbf{v}_s \cdot \mathbf{v}_s + p_f) \mathbf{v}_s \cdot \mathbf{n}
\end{bmatrix} dS , \quad (4)`$.

Note that the first term of (1) can be written (Reynold's transport theorem (https://en.wikipedia.org/wiki/Reynolds_transport_theorem) as $` \int \frac{d \mathbf{y}}{dt} dV = \frac{d}{dt} (\int \mathbf{y} dV) - \int \mathbf{y} (\mathbf{v}_s \cdot \mathbf{n}) dS`$.

Using the Reynold's and divergence theorem relations, we write (1) as
$`
\frac{d}{dt} \left( \int
\begin{bmatrix} 
   m_f n_f \\
   m_f n_f \mathbf{v}_f \\
   C_V 2 n_f k_B T_f + \frac{1}{2} m_f n_f \mathbf{v}_f\cdot\mathbf{v}_f
\end{bmatrix} 
dV \right) - \int
\begin{bmatrix} 
   m_f n_f (\mathbf{v}_s \cdot \mathbf{n})\\
   m_f n_f \mathbf{v}_s (\mathbf{v}_s \cdot \mathbf{n})\\
   (C_V 2 n_f k_B T_f + \frac{1}{2} m_f n_f \mathbf{v}_s\cdot\mathbf{v}_s) (\mathbf{v}_s \cdot \mathbf{n})
\end{bmatrix} 
dS`$
$`= - \int
\begin{bmatrix} 
   m_f n_f (\mathbf{v}_s + \mathbf{v}_{th}^i) \cdot \mathbf{n} \\
   m_f n_f \mathbf{v}_s (\mathbf{v}_s + \mathbf{v}_{th}^i) \cdot \mathbf{n} + p_f \mathbf{I} \cdot \mathbf{n} \\
   (C_V 2 n_f k_B T_f) (\mathbf{v}_s + \mathbf{v}_{th}) \cdot \mathbf{n} +  (\frac{1}{2} m_f n_f \mathbf{v}_s \cdot \mathbf{v}_s + p_f) \mathbf{v}_s \cdot \mathbf{n}
\end{bmatrix} dS. (5)`$
Fuel mass $`M_f = \int m_f n_f dV`$, momentum $`\mathbf{P}_f = \int m_f n_f \mathbf{v}_f dV`$, and energy $`E_f = \int C_V 2 n_f k_B T_f + \frac{1}{2} m_f n_f \mathbf{v}_f\cdot\mathbf{v}_f dV`$ along with cancelation of lhs-vs-rhs terms allow to write (5) as a set of equations describing the time evolution of macroscopic fuel quantities 
$`\frac{d}{dt} 
\begin{bmatrix} 
   M_f \\
   \mathbf{P}_f \\
   E_f
\end{bmatrix} + \int
\begin{bmatrix} 
   m_f n_f \mathbf{v}_{th}^i \cdot \mathbf{n} \\
   m_f n_f \mathbf{v}_s \mathbf{v}_{th}^i \cdot \mathbf{n} + p_f \mathbf{I} \cdot \mathbf{n} \\
   (C_V n_f k_B T_f) (\mathbf{v}_{th}^i + \mathbf{v}_{th}^e) \cdot \mathbf{n} +  p_f \mathbf{v}_s \cdot \mathbf{n}
\end{bmatrix} dS = \mathbf{0}. \quad (6)`$

Notes:
- $`m_f (ion + electron), C_V, k_B`$ are constants
- $`p_f (ion + electron), \mathbf{v}_{th}`$ to be defined, in our case by ideal gas relations
# SHELL - NEWTON MODEL
Newton's model "m a = force" of a rigid body can be used to define the dynamics of "fyling" shell. Importantly, the momentum and energy of the shell is defined as $`\mathbf{P}_s = M_s \mathbf{v}_s`$ and $`E_s = C_M M_s T_s + \frac{1}{2} M_s \mathbf{v}_s \cdot \mathbf{v}_s`$, respectively, which allows us to define the model equations for the~shell 
$`\frac{d}{dt} 
\begin{bmatrix} 
   M_s \\
   M_s \mathbf{v}_s \\
   C_M M_s T_s + \frac{1}{2} M_s \mathbf{v}_s \cdot \mathbf{v}_s
\end{bmatrix} = \int
\begin{bmatrix} 
   m_f n_f \mathbf{v}_{th}^i \cdot \mathbf{n} \\
   m_f n_f \mathbf{v}_s \mathbf{v}_{th}^i \cdot \mathbf{n} + p_f \mathbf{I} \cdot \mathbf{n} \\
   (C_V 2 n_f k_B T_f) \mathbf{v}_{th} \cdot \mathbf{n} +  p_f \mathbf{v}_s \cdot \mathbf{n}
\end{bmatrix} dS, \quad (7)`$
arising from the conservation point of view at (6) as conservation of total mass $`M_f + M_s`$, total momentum $`\mathbf{P}_f + \mathbf{P}_s`$, and total energy $`E_f + E_s`$.

In order to track the position $`R(t)`$ of the shell in time, we add the last equation to close the system as
$`\frac{d R}{dt} = \mathbf{v}_s \cdot \mathbf{n} . (8)`$

Notes:
- $C_M$ is a constant

# $\alpha$ - MODEL OF FUSION BURN
The model of fusion burn is directly coupled to the population of "new born" $\alpha$ particles (ionized Helium),
$`\frac{d n_\alpha}{dt} = \beta T_f^2 n_f^2, (9)`$
which also affects the fuel fluid in terms of "sink" of fuel particles (2 fuel particles burn to create one $\alpha$ particle) and "source" to internal fuel energy (increase in $T_f$)
$`\mathbf{S} = 
\begin{bmatrix} - 2 \frac{d n_\alpha}{dt} \\ 0 \\ E_\alpha \frac{d n_\alpha}{dt}
\end{bmatrix}, \quad (10)`$
following the description in Lawson criterion (https://en.wikipedia.org/wiki/Lawson_criterion).

Source (10) is then incorporated into the fuel fluid model by adjusting (4) as
$` \int \frac{d \mathbf{y}}{dt} dV + \int \nabla \cdot \mathbf{F} dV = \int \mathbf{S} dV. \quad (11)`$

Notes:
- $\beta$ is a constant to be set in correct units
- $E_\alpha$ is the kinetic energy of the "new born" $\alpha$ particle at 3.5 MeV (needs to be conversed to correct units)

# COMPLETE ODE SYSTEM
Marcos, we end up using a system of 8 equations for $\frac{d n_f}{dt}, \frac{d v_f}{dt}, \frac{d T_f}{dt}, \frac{d M_s}{dt}, \frac{d v_s}{dt}, \frac{d T_s}{dt}, \frac{d R}{dt}, \frac{d n_\alpha}{dt}$ represented by (4), (7), (8), (9), and (10). 

The spherical symmetry allows for simplification $`\mathbf{v}_f \cdot \mathbf{n} = v_f, \mathbf{v}_{th} \cdot \mathbf{n} = v_{th}, \mathbf{v}_s \cdot \mathbf{n} = v_s`$ and $`\mathbf{v}_f = v_f, \mathbf{v}_s = v_s`$ and $`p_f \mathbf{I} \cdot \mathbf{n} = p_f`$ in (4), (7), and (8), leading to

$`\begin{align}
\int
\frac{d}{dt}\begin{bmatrix} 
   m_f n_f \\
   m_f n_f v_f \\
   C_V 2 n_f k_B T_f + \frac{1}{2} m_f n_f v_f^2
\end{bmatrix} 
dV &= - \int
\begin{bmatrix} 
   m_f n_f (v_s + v_{th}^i) \\
   m_f n_f v_s (v_s + v_{th}^i) + p_f \\
   (C_V 2 n_f k_B T_f) (v_s + v_{th}) +  (\frac{1}{2} m_f n_f v_s^2 + p_f) v_s
\end{bmatrix} dS + \int
\begin{bmatrix} - 2 \frac{d n_\alpha}{dt} \\ 0 \\ E_\alpha \frac{d n_\alpha}{dt}
\end{bmatrix} 
dV,
\\
\frac{d}{dt} 
\begin{bmatrix} 
   M_s \\
   M_s v_s \\
   C_M M_s T_s + \frac{1}{2} M_s v_s^2
\end{bmatrix} &= \int
\begin{bmatrix} 
   m_f n_f v_{th}^i \\
   m_f n_f v_s v_{th}^i + p_f \\
   (C_V 2 n_f k_B T_f) v_{th} +  p_f v_s
\end{bmatrix} dS, (12) 
\\
\frac{d R}{dt} &= v_s ,
\\
\frac{d n_\alpha}{dt} &= \beta T_f^2 n_f^2 .
\end{align}`$

Notes:
- $\alpha$ particles are thermalized to $T_f$, which should be reflected in the internal  and kinetic energy of the fuel as $`C_V (2 n_f + 3 n_\alpha) k_B T_f + \frac{1}{2} (m_f n_f + m_{He} n_\alpha) \mathbf{v}_f\cdot\mathbf{v}_f`$ in (2) and corresponding flux $`(C_V n_f k_B T_f) (2 \mathbf{v}_f + \mathbf{v}_{th}^i + \mathbf{v}_{th}^e) + (C_V n_\alpha k_B T_f) (3 \mathbf{v}_f + \mathbf{v}_{th}^\alpha + 2 \mathbf{v}_{th}^e) +  (\frac{1}{2} (m_f n_f + m_{He} n_\alpha) \mathbf{v}_f \cdot \mathbf{v}_f + p_f) \mathbf{v}_f`$ in (3), and also propagate this update to the whole system.
-  Initial condition for $v_f(t=0)$ can be obtained from $`v_f(t=0) = \int R v_s(t=0) dV`$, i.e. using linear profile of velocity between *zero* at the center of the capsule and $v_s(t=0)$ at the surface at the capsule at time $`t=0`$.
-  Consider completely eliminating $v_f$ by defining it as $`v_f(R) = R v_s`$, i.e. using linear profile of velocity between *zero* at the center of the capsule and $v_s$ at the surface at the capsule.

Then, the *spatial constness* and *volume & surface integration over the sphere* (the magic assumptions we used on my whiteboard) in the set of 8 equations (12) will lead you to the first order equations you need to implement in python.

# FIRST ORDER SYSTEM OF ODEs
TBD

