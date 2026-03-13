import numpy as np

# Elementos orbitais
E = 3.299702031  # rad
e = 0.0484007
a = 5.20332
n = 1.0
w_deg = -85.7958
I_deg = 1.30537
Omega_deg = 100.535
 
# Calculando a posição no referencial inercial
x = a * (np.cos(E) - e)
y = a * (np.sqrt(1 - e**2)) * np.sin(E)
print("A posição no referencial Inercial é: x =", x, "e y =", y)

# Definindo as matrizes de rotação
def P1(w_deg):
    w = np.radians(w_deg)
    return np.array([
        [np.cos(w), -np.sin(w), 0],
        [np.sin(w),  np.cos(w), 0],
        [0,          0,         1]
    ])

def P2(I_deg):
    I = np.radians(I_deg)
    return np.array([
        [1,  0,          0],
        [0,  np.cos(I), -np.sin(I)],
        [0,  np.sin(I),  np.cos(I)]
    ])

def P3(Omega_deg):
    Omega = np.radians(Omega_deg)
    return np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega),  np.cos(Omega), 0],
        [0,              0,             1]
    ])

# Função para multiplicar as matrizes
def multiply_matrices(w_deg, I_deg, Omega_deg):
    P1_matrix = P1(w_deg)
    P2_matrix = P2(I_deg)
    P3_matrix = P3(Omega_deg)
    product = np.dot(np.dot(P3_matrix, P2_matrix), P1_matrix)
    return product

# Calculando o produto das matrizes
result = multiply_matrices(w_deg, I_deg, Omega_deg)
print("Produto das matrizes P1 * P2 * P3:")
print(result)

# Calculando a anomalia verdadeira e a distância
M = 3.307311666  # rad
f = M + 2 * e * np.sin(M) + (5/4) * (e**2) * np.sin(2*M) + (e**3) * ((13/12) * np.sin(3*M) - (1/4) * np.sin(M))
print(f)
r = a * (1 - (e * np.cos(E)))
cosf = (np.cos(E) - e) / (1 - e * np.cos(E))

MatrizMult = np.array([[r * cosf], [r * np.sin(f)], [0]])
CoordSinodico = np.dot(result, MatrizMult)

# Nomeando e imprimindo os elementos da matriz CoordSinodico
coord_names = ['x_sinodico', 'y_sinodico', 'z_sinodico']
for name, value in zip(coord_names, CoordSinodico):
    print(f"{name}: {value[0]}")

print("As coordenadas de posição do corpo no referencial sideral são:", CoordSinodico)

# Calculando as componentes de velocidade
dr = n * a * np.sqrt((a**2) * (e**2) - (r - a)**2)
MatrizMult2 = np.array([dr * -np.sin(f), dr * np.cos(f), 0])
VelSinodico = np.dot(result, MatrizMult2)

# Nomeando e imprimindo os elementos da matriz VelSinodico
vel_names = ['vx_sinodico', 'vy_sinodico', 'vz_sinodico']
for name, value in zip(vel_names, VelSinodico):
    print(f"{name}: {value}")

print("As componentes de velocidade do corpo no referencial sideral são:", VelSinodico)
