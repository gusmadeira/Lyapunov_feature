# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import glob
import re
import os
from scipy.optimize import newton

# ==============================================================================
# SEÇÃO 1: CARREGAMENTO DA ESTRUTURA
# ==============================================================================

def load_generic_orbitals(pattern):
    file_paths = glob.glob(pattern)
    a_vals, e_vals = [], []
    for path in file_paths:
        try:
            df_tmp = pd.read_csv(path, sep=r'\s+', comment='#', header=None, nrows=1)
            if not df_tmp.empty:
                a_vals.append(float(df_tmp.iloc[0, 1]))
                e_vals.append(float(df_tmp.iloc[0, 2]))
        except: continue
    return a_vals, e_vals

def load_white_structure_data():
    patterns = ["./1o_elmtosOrbitais_*.txt", "./3o_elmtosOrbitais_*.txt", 
                "./elmtosOrbitais_*_*.txt", "./2_ordem_elmtosOrbitais_*_*_*"]
    at, et = [], []
    for p in patterns:
        av, ev = load_generic_orbitals(p)
        at.extend(av); et.extend(ev)
    return at, et

def load_all_delimitation_data():
    file_paths = glob.glob("delimitacao_elmtosOrbitais_*_*_*.txt")
    if not file_paths: return np.array([]), np.array([])
    
    all_pts = []
    for path in file_paths:
        try:
            parts = os.path.basename(path).replace(".txt", "").split('_')
            cj = float(parts[-3])
            x_val = float(parts[-2])
            with open(path, 'r') as f:
                line = f.readline()
                if line.strip() and not line.strip().startswith('#'):
                    vals = line.split()
                    all_pts.append({'cj': cj, 'x': x_val, 'a': float(vals[1]), 'e': float(vals[2])})
        except: continue

    if not all_pts: return np.array([]), np.array([])
    
    cj_groups = {}
    for pt in all_pts:
        cj_groups.setdefault(pt['cj'], []).append(pt)
    
    r1, r2 = [], []
    for cj in sorted(cj_groups.keys(), reverse=True):
        pts = cj_groups[cj]
        if len(pts) >= 2:
            r1.append(max(pts, key=lambda p: p['x']))
            r2.append(min(pts, key=lambda p: p['x']))
    
    path_final = r1 + sorted(r2, key=lambda p: p['a'])
    return np.array([p['a'] for p in path_final]), np.array([p['e'] for p in path_final])

def plot_white_structure(ax, a_data, e_data):
    if not a_data: return
#    limites = [1.77, 1.808, 1.86, 1.98, 2.01, 2.1, 2.4, 2.5, 2.7, 2.85, 2.898, 3.03]
    limites = [1.77, 1.808, 1.87, 1.98, 2.01, 2.1, 2.4, 2.5, 2.7, 2.85, 2.9, 2.965, 3.207, 3.235, 3.6]
    for i in range(len(limites)-1):
        pts = sorted([(a, e) for a, e in zip(a_data, e_data) if limites[i] <= a < limites[i+1]], key=lambda x: x[1])
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color='blue', linestyle=':', linewidth=3.0, zorder=20)

# ==============================================================================
# SEÇÃO 2: CÁLCULO DAS RESSONÂNCIAS
# ==============================================================================

def n_freq(a, mu): return (1 / a**3)**0.5 * np.sqrt(1 + mu + (3 * mu / 4) * (1/a)**2)
def k_freq(a, mu): return (1 / a**3)**0.5 * np.sqrt(1 + mu - (3 * mu / 4) * (1/a)**2)
def varpi_dot(a, mu): return n_freq(a, mu) - k_freq(a, mu)
def f_ressonancia(a, m, j, mu, omega_corpo):
    return m * omega_corpo - (m - j) * n_freq(a, mu) - j * varpi_dot(a, mu)

def get_resonance_positions(mu, omega_corpo, a_min, a_max):
    defs = [
        ("a_sync", 1, 0), ("3:4", -4, -1), ("4:3", -3, 1), ("5:3", -3, 2),
        ("2:1", -1, 1),  ("7:3", -3, 4), ("8:3", -3, 5), ("5:4", -4, 1),  
        ("3:2", -2, 1), ("7:4", -4, 3), ("9:4", -4, 5),  ("5:2", -2, 3), 
        ("3:1", -1, 2), ("5:6", -6, -1), ("1:2", -2, -1), ("16:17", -17, -1),
        ("19:17", -18, 1), ("19:16", -16, 3), ("8:5", -5, 3), ("10:9", -9, 1), ("7:5", -5, 2)
    ]
    results = {}
    for nome, m, j in defs:
        try:
            # Aproximação Kepleriana para o chute inicial
            n_approx = m * omega_corpo / (m - j)
            a_guess = ((1 + mu) / n_approx**2)**(1/3)
            res_pos = newton(lambda a: f_ressonancia(a, m, j, mu, omega_corpo), a_guess, tol=1e-5)
            # Filtra ressonâncias que estão dentro da janela de exibição
            if a_min - 0.1 <= res_pos <= a_max + 0.1: 
                results[nome] = res_pos
        except: continue
    return results

# ==============================================================================
# SEÇÃO 3: PLOTAGEM
# ==============================================================================

def create_lyapunov_plot(filename, mu_val, lambda_val, overlay=True):
    print(f"Processando ({'com' if overlay else 'sem'}) overlay: {filename}")
    cols = ['a', 'e', 'tf', 'col', 'af', 'ef', 'lyap']
    try:
        data = pd.read_csv(filename, sep=r'\s+', comment='#', header=None, names=cols)
        is_inf_mask = data['lyap'].astype(str).str.lower().str.contains('inf')
        data['lyap'] = pd.to_numeric(data['lyap'], errors='coerce')
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}"); return

    mask_finite = (data['lyap'] > 0) & np.isfinite(data['lyap'])
    data['T_lyap'] = np.nan
    data.loc[mask_finite, 'T_lyap'] = (1.0 / data.loc[mask_finite, 'lyap']) / (2 * np.pi)

    pivot = data.pivot(index='e', columns='a', values='T_lyap')
    pivot_inf = data.assign(inf_flag=is_inf_mask).pivot(index='e', columns='a', values='inf_flag')
    
    X, Y, Z = pivot.columns.values, pivot.index.values, pivot.values
    Z_inf = pivot_inf.values

    cmap = LinearSegmentedColormap.from_list("lyap", 
           [(0.00, "#FF0000"), (0.20, "yellow"), (0.60, "purple"), (1.00, "fuchsia")], N=256)
    cmap.set_bad('white')

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('white')
    
    mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap, norm=mcolors.LogNorm(vmin=100, vmax=10000), zorder=1)
    inf_layer = np.ma.masked_where(Z_inf == False, np.ones_like(Z_inf))
    ax.pcolormesh(X, Y, inf_layer, shading='auto', cmap=ListedColormap(['black']), zorder=2)

    # TRAVA OS LIMITES nos dados de partículas
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    if overlay:
        sa, se = load_white_structure_data()
        plot_white_structure(ax, sa, se)
        
        da, de = load_all_delimitation_data()
        if da.size > 0:
            ax.plot(da, de, color='black', linestyle='-', linewidth=1.0, zorder=21)
        suffix = "overlay"
    else:
        suffix = "clean"

    # Ressonâncias (usando a nova lista extensa)
    res = get_resonance_positions(mu_val, lambda_val, X.min(), X.max())
    for label, x_v in res.items():
        ax.axvline(x=x_v, color='#4F4F4F', linestyle='--', linewidth=1.0, alpha=0.5, zorder=5)
        # Ajuste dinâmico da altura do texto para não poluir
        y_pos = Y.max() * 0.93
        txt_label = r'$a_{sync}$' if 'sync' in label else label
        ax.text(x_v, y_pos, txt_label, color='#333333', rotation=90, 
                fontsize=9, ha='right', fontweight='bold', zorder=25)

    ax.set_xlabel(r'$a/R_c$', fontsize=14); ax.set_ylabel('$e$', fontsize=14)
    plt.colorbar(mesh, label=r'Lyapunov time ($T_{rot}$)')
    
    output_name = f"mapa_lyapunov_{suffix}_mu{mu_val}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Arquivo salvo: {output_name}")

if __name__ == "__main__":
    # Parametros de entrada
    file_target = "particles_mu2.10-3_lbd0.523.txt"
    mu = 0.002
    lbd = 0.5228667405826437
    
    # Gera as duas versões
    create_lyapunov_plot(file_target, mu, lbd, overlay=True)
    create_lyapunov_plot(file_target, mu, lbd, overlay=False)