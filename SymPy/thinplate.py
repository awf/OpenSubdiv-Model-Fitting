from sympy import *

# Bicubic B-spline patch control points 
p00, p01, p02, p03 = symbols('p00 p01 p02 p03')
p10, p11, p12, p13 = symbols('p10 p11 p12 p13')
p20, p21, p22, p23 = symbols('p20 p21 p22 p23')
p30, p31, p32, p33 = symbols('p30 p31 p32 p33')

# Parametric coordinates
u, v = symbols('u v')

# B-spline basis functions
b0 = lambda x : (1 - x) ** 3 / 6
b1 = lambda x : (3 * x**3 - 6 * x**2 + 4) / 6
b2 = lambda x : (-3 * x**3 + 3 * x**2 + 3 * x + 1) / 6
b3 = lambda x : x**3 / 6

# Surface patch
S = (p00 * b0(u) + p10 * b1(u) + p20 * b2(u) + p30 * b3(u)) * b0(v) + \
    (p01 * b0(u) + p11 * b1(u) + p21 * b2(u) + p31 * b3(u)) * b1(v) + \
    (p02 * b0(u) + p12 * b1(u) + p22 * b2(u) + p32 * b3(u)) * b2(v) + \
    (p03 * b0(u) + p13 * b1(u) + p23 * b2(u) + p33 * b3(u)) * b3(v)

# First-order derivatives
dS_du = diff(S, u)
dS_dv = diff(S, v)

# Second-order derivatives
d2S_duu = diff(dS_du, u)
d2S_duv = diff(dS_du, v)
d2S_dvv = diff(dS_dv, v)

# Thin-plate stretching and bending energies
stretching = integrate(integrate(dS_du ** 2 + dS_dv ** 2,                        (u, 0, 1)), (v, 0, 1))
bending    = integrate(integrate(d2S_duu ** 2 + 2 * d2S_duv ** 2 + d2S_dvv ** 2, (u, 0, 1)), (v, 0, 1))

common_denom = 302400
quadm_stretching = Matrix(16, 16, lambda i, j : stretching.coeff(eval('p' + str(i // 4) + str(i % 4)) * eval('p' + str(j // 4) + str(j % 4))) / (1 if i == j else 2))
print 'Quadratic form for stretching energy is (1 / ' + str(common_denom) + ') * '
pprint(common_denom * quadm_stretching, wrap_line=False)
quadm_bending = Matrix(16, 16, lambda i, j : bending.coeff(eval('p' + str(i // 4) + str(i % 4)) * eval('p' + str(j // 4) + str(j % 4))) / (1 if i == j else 2))
print '\nQuadratic form for bending energy is (1 / ' + str(common_denom) + ') * '
pprint(common_denom * quadm_bending, wrap_line=False)
